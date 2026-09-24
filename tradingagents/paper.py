"""Paper trading: act on the agents' ratings with a simulated account.

The graph stops at a rating. This module keeps a JSON ledger of cash and
positions, turns each rating into a pending order, and fills that order at the
open of the first session that starts after the decision was taken, so a fill
never uses a price that was already known when the call was made. Nothing here
talks to a broker; it exists to see how the ratings would have traded.

Sizing, as a fraction of account equity (``max_position`` defaults to 25%):
    Buy          -> max_position
    Overweight   -> at least half of max_position (an existing larger stake is kept)
    Hold         -> no order
    Underweight  -> halve the position
    Sell         -> close the position
    REVIEW       -> no order (the decision had no readable rating)
The account is long only and never borrows: buys are capped by free cash.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from tradingagents import screener
from tradingagents.agents.rating import RATINGS_5_TIER
from tradingagents.dataflows.symbols import crypto_base, normalize_symbol
from tradingagents.portfolio import PortfolioContext, Position

# Regular-session hours by exchange time zone. For any other exchange a fill
# waits for the first session dated after the decision day, which is never early.
MARKET_HOURS = {
    "America/New_York": (time(9, 30), time(16, 0)),
    "Europe/London": (time(8, 0), time(16, 30)),
}

# What each account currency is compared against in the status report.
BENCHMARKS = {"USD": "SPY", "GBP": "^FTSE"}

# Yahoo quotes London shares in pence; the ledger keeps pounds.
QUOTE_SCALE = {("GBp", "GBP"): 0.01, ("GBX", "GBP"): 0.01}

# A rebalance smaller than this share of equity is not worth a trade.
MIN_TRADE_FRACTION = 0.01

TRADED_RATINGS = ("Buy", "Overweight", "Underweight", "Sell")


class PriceSource(Protocol):
    def sessions(self, ticker: str, start: str, end: str) -> tuple[pd.DataFrame, str]: ...

    def currency(self, ticker: str) -> str: ...


class YahooPrices:
    """Daily sessions and quote currency from Yahoo Finance."""

    def sessions(self, ticker: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
        from tradingagents.dataflows.vendors.yahoo.market import get_sessions
        return get_sessions(ticker, start, end)

    def currency(self, ticker: str) -> str:
        from tradingagents.dataflows.vendors.yahoo.market import get_quote_currency
        return get_quote_currency(ticker)


def default_ledger_path(config: dict) -> Path:
    return Path(config["results_dir"]).parent / "paper" / "ledger.json"


# --------------------------------------------------------------------------- ledger


def new_ledger(cash: float, currency: str, prices: PriceSource, now: datetime,
               max_position: float = 0.25, slippage_bps: float = 10.0,
               commission: float = 0.0, whole_shares: bool = False) -> dict:
    currency = currency.upper()
    if cash <= 0:
        raise ValueError("starting cash must be positive")
    if not 0 < max_position <= 1:
        raise ValueError("max position must be between 0 and 1, e.g. 0.25 for 25%")
    if slippage_bps < 0 or commission < 0:
        raise ValueError("slippage and commission cannot be negative")
    ledger = {
        "version": 1,
        "currency": currency,
        "created": now.isoformat(timespec="seconds"),
        "starting_cash": float(cash),
        "cash": float(cash),
        "rules": {"max_position": max_position, "slippage_bps": slippage_bps,
                  "commission": commission, "whole_shares": whole_shares},
        "benchmark": None,
        "positions": {},
        "pending": [],
        "trades": [],
        "decisions": [],
    }
    benchmark = BENCHMARKS.get(currency)
    if benchmark:
        frame, _ = _recent_sessions(prices, benchmark, now)
        close = float(frame["close"].iloc[-1])
        ledger["benchmark"] = {"ticker": benchmark, "start_price": close, "last_price": close}
    return ledger


def load_ledger(path: str | Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(
            f"no paper account at {path}; create one with `tradingagents paper init`"
        ) from None


def save_ledger(ledger: dict, path: str | Path) -> None:
    """Write atomically, so an interrupted save cannot corrupt the account."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".ledger-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
    os.replace(tmp, path)


def equity(ledger: dict) -> float:
    return ledger["cash"] + sum(p["quantity"] * p["last_price"] for p in ledger["positions"].values())


def portfolio_context(ledger: dict) -> PortfolioContext:
    """The book as the agents see it, with prices in each instrument's quote units."""
    return PortfolioContext(
        cash=round(ledger["cash"], 2),
        currency=ledger["currency"],
        positions=[
            Position(ticker=t, quantity=p["quantity"],
                     average_price=p["average_price"] / p["quote_scale"])
            for t, p in ledger["positions"].items()
        ],
    )


# --------------------------------------------------------------------------- timing


def last_completed_session(frame: pd.DataFrame, tz: str, now: datetime) -> date | None:
    """The latest session whose close had passed at ``now``."""
    local = now.astimezone(ZoneInfo(tz))
    hours = MARKET_HOURS.get(tz)
    for day in reversed(list(frame.index)):
        if day < local.date():
            return day
        if day == local.date() and hours and local.time() >= hours[1]:
            return day
    return None


def fill_session(frame: pd.DataFrame, tz: str, decided_at: datetime, after: date) -> date | None:
    """The first session after ``after`` that opened after ``decided_at``."""
    local = decided_at.astimezone(ZoneInfo(tz))
    hours = MARKET_HOURS.get(tz)
    for day in frame.index:
        if day <= after:
            continue
        if day > local.date():
            return day
        if day == local.date() and hours and local.time() < hours[0]:
            return day
    return None


def _recent_sessions(prices: PriceSource, ticker: str, now: datetime,
                     since: date | None = None) -> tuple[pd.DataFrame, str]:
    today = now.astimezone(timezone.utc).date()
    start = since or today - timedelta(days=14)
    return prices.sessions(ticker, start.isoformat(), (today + timedelta(days=2)).isoformat())


def quote_scale(prices: PriceSource, ticker: str, currency: str) -> float:
    quote = prices.currency(ticker)
    if quote == currency:
        return 1.0
    scale = QUOTE_SCALE.get((quote, currency))
    if scale is None:
        raise ValueError(
            f"{ticker} is quoted in {quote}, but this paper account is in {currency}; "
            "use a separate account per currency"
        )
    return scale


# --------------------------------------------------------------------------- decisions


def target_weight(rating: str, current: float, max_position: float) -> float | None:
    """The weight a rating asks for, or None when it asks for no trade."""
    if rating == "Buy":
        return max_position
    if rating == "Overweight":
        return max(current, max_position / 2)
    if rating == "Underweight":
        return current / 2
    if rating == "Sell":
        return 0.0
    return None


def decide(ledger: dict, ticker: str, prices: PriceSource,
           decider: Callable[[str, str, PortfolioContext], str],
           clock: Callable[[], datetime]) -> dict:
    """Run the agents on ``ticker`` for its last completed session and queue the order.

    The currency is checked before the agents run, since a run can take an hour.
    """
    ticker = normalize_symbol(ticker)
    if ticker.endswith("=F"):
        raise ValueError(
            f"{ticker} is a rolling futures contract, which a paper account cannot hold "
            "honestly; use a commodity fund such as GLD (gold), SLV (silver) or USO (oil)"
        )
    scale = quote_scale(prices, ticker, ledger["currency"])
    frame, tz = _recent_sessions(prices, ticker, clock())
    analysis = last_completed_session(frame, tz, clock())
    if analysis is None:
        raise ValueError(f"{ticker} has no completed session in the last two weeks")

    rating = decider(ticker, analysis.isoformat(), portfolio_context(ledger))
    decided_at = clock().astimezone(timezone.utc)
    record = {"ticker": ticker, "analysis_date": analysis.isoformat(), "rating": rating,
              "decided_at": decided_at.isoformat(timespec="seconds")}
    ledger["decisions"].append(record)

    held = ticker in ledger["positions"]
    ledger["pending"] = [o for o in ledger["pending"] if o["ticker"] != ticker]
    if rating in ("Buy", "Overweight") or (rating in ("Underweight", "Sell") and held):
        ledger["pending"].append({**record, "quote_scale": scale})
    return record


# --------------------------------------------------------------------------- fills


def _round_quantity(quantity: float, whole: bool) -> float:
    step = 1 if whole else 1_000_000
    return math.floor(quantity * step + 1e-9) / step


def _execute(ledger: dict, order: dict, session: date, open_price: float) -> dict | None:
    rules = ledger["rules"]
    ticker = order["ticker"]
    position = ledger["positions"].get(ticker)
    if position:
        position["last_price"] = open_price
    held = position["quantity"] if position else 0.0
    total = equity(ledger)
    current = held * open_price / total
    target = target_weight(order["rating"], current, rules["max_position"])
    if target is None:
        return None
    delta = (target - current) * total
    if target > 0 and abs(delta) < MIN_TRADE_FRACTION * total:
        return None

    slip = rules["slippage_bps"] / 10_000
    commission = rules["commission"]
    if delta > 0:
        price = open_price * (1 + slip)
        spend = min(delta, ledger["cash"] - commission)
        quantity = _round_quantity(spend / price, rules["whole_shares"])
        if quantity <= 0:
            return None
        ledger["cash"] -= quantity * price + commission
        if position is None:
            position = ledger["positions"][ticker] = {
                "quantity": 0.0, "average_price": 0.0, "last_price": open_price,
                "quote_scale": order["quote_scale"]}
        cost = position["quantity"] * position["average_price"] + quantity * price
        position["quantity"] += quantity
        position["average_price"] = cost / position["quantity"]
        side, realized = "buy", None
    else:
        if position is None:
            return None
        price = open_price * (1 - slip)
        quantity = held if target == 0 else min(
            held, _round_quantity(-delta / price, rules["whole_shares"]))
        if quantity <= 0:
            return None
        ledger["cash"] += quantity * price - commission
        realized = (price - position["average_price"]) * quantity - commission
        position["quantity"] -= quantity
        if position["quantity"] <= 1e-9:
            del ledger["positions"][ticker]
        side = "sell"

    trade = {"date": session.isoformat(), "ticker": ticker, "side": side,
             "quantity": quantity, "price": round(price, 6), "rating": order["rating"],
             "analysis_date": order["analysis_date"]}
    if realized is not None:
        trade["realized"] = round(realized, 2)
    ledger["trades"].append(trade)
    return trade


def settle(ledger: dict, prices: PriceSource, now: datetime) -> list[dict]:
    """Fill every pending order whose fill session has opened; sells go first."""
    order_rank = {r: i for i, r in enumerate(RATINGS_5_TIER)}
    fills, still_pending = [], []
    for order in sorted(ledger["pending"], key=lambda o: -order_rank.get(o["rating"], 0)):
        analysis = date.fromisoformat(order["analysis_date"])
        frame, tz = _recent_sessions(prices, order["ticker"], now, since=analysis)
        session = fill_session(frame, tz, datetime.fromisoformat(order["decided_at"]), analysis)
        if session is None:
            still_pending.append(order)
            continue
        open_price = float(frame.loc[session, "open"]) * order["quote_scale"]
        trade = _execute(ledger, order, session, open_price)
        if trade:
            fills.append(trade)
    ledger["pending"] = still_pending
    return fills


def mark_to_market(ledger: dict, prices: PriceSource, now: datetime) -> None:
    """Value each position, and the benchmark, at its latest close."""
    for ticker, position in ledger["positions"].items():
        frame, _ = _recent_sessions(prices, ticker, now)
        position["last_price"] = float(frame["close"].iloc[-1]) * position["quote_scale"]
    if ledger["benchmark"]:
        frame, _ = _recent_sessions(prices, ledger["benchmark"]["ticker"], now)
        ledger["benchmark"]["last_price"] = float(frame["close"].iloc[-1])
    ledger["marked_at"] = now.astimezone(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- report


def report(ledger: dict) -> str:
    cur = ledger["currency"]
    total = equity(ledger)
    change = total / ledger["starting_cash"] - 1
    lines = [f"Paper account ({cur}), started {ledger['created'][:10]}",
             f"Equity {total:,.2f} · cash {ledger['cash']:,.2f} · return {change:+.2%}"]
    bench = ledger.get("benchmark")
    if bench:
        bench_change = bench["last_price"] / bench["start_price"] - 1
        lines.append(f"{bench['ticker']} over the same period {bench_change:+.2%} "
                     f"(you are {change - bench_change:+.2%} against it)")
    lines.append("")
    if ledger["positions"]:
        lines.append("Positions:")
        for ticker, p in sorted(ledger["positions"].items()):
            value = p["quantity"] * p["last_price"]
            gain = p["last_price"] / p["average_price"] - 1
            lines.append(f"  {ticker}: {p['quantity']:,.4g} @ {p['average_price']:,.2f} "
                         f"-> {p['last_price']:,.2f} · {value:,.2f} ({gain:+.2%}, "
                         f"{value / total:.0%} of equity)")
    else:
        lines.append("Positions: none")
    if ledger["pending"]:
        lines.append("Waiting to fill at the next open:")
        lines += [f"  {o['ticker']}: {o['rating']} (from {o['analysis_date']})"
                  for o in ledger["pending"]]
    if ledger["trades"]:
        lines.append("Recent trades:")
        for t in ledger["trades"][-10:]:
            extra = f" · realized {t['realized']:+,.2f}" if "realized" in t else ""
            lines.append(f"  {t['date']} {t['side']} {t['quantity']:,.4g} {t['ticker']} "
                         f"@ {t['price']:,.2f} ({t['rating']}){extra}")
    return "\n".join(lines)


def summary(ledger: dict) -> dict:
    """The account as data, for a UI to lay out."""
    total = equity(ledger)
    change = total / ledger["starting_cash"] - 1
    bench = ledger.get("benchmark")
    bench_change = bench["last_price"] / bench["start_price"] - 1 if bench else None
    positions = []
    for ticker, p in sorted(ledger["positions"].items()):
        value = p["quantity"] * p["last_price"]
        positions.append({"ticker": ticker, "quantity": p["quantity"],
                          "average_price": p["average_price"], "last_price": p["last_price"],
                          "value": value, "gain": p["last_price"] / p["average_price"] - 1,
                          "weight": value / total})
    return {
        "currency": ledger["currency"], "created": ledger["created"],
        "marked_at": ledger.get("marked_at"), "rules": ledger["rules"],
        "starting_cash": ledger["starting_cash"], "cash": ledger["cash"],
        "equity": total, "return": change,
        "benchmark": bench and {"ticker": bench["ticker"], "return": bench_change,
                                "excess": change - bench_change},
        "positions": positions,
        "pending": ledger["pending"],
        "trades": ledger["trades"][-25:][::-1],
        "decisions": ledger["decisions"][-25:][::-1],
    }


# --------------------------------------------------------------------------- session


def graph_decider(config: dict) -> Callable[[str, str, PortfolioContext], str]:
    """Run the full graph for a ticker, with the analysts that apply to it."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graphs: dict[tuple[str, ...], TradingAgentsGraph] = {}

    def decide(ticker: str, trade_date: str, portfolio: PortfolioContext) -> str:
        asset_type = "crypto" if crypto_base(ticker) else "stock"
        # Crypto and commodity funds have no company financials to analyze.
        no_fundamentals = asset_type == "crypto" or ticker in screener.COMMODITY_FUNDS
        analysts = ("market", "social", "news") if no_fundamentals else (
            "market", "social", "news", "fundamentals")
        if analysts not in graphs:
            graphs[analysts] = TradingAgentsGraph(list(analysts), config=config)
        _, signal = graphs[analysts].propagate(ticker, trade_date, asset_type, portfolio=portfolio)
        return signal

    return decide


def run_session(path: str | Path, tickers: list[str], *, prices: PriceSource,
                decider: Callable[[str, str, PortfolioContext], str],
                clock: Callable[[], datetime], log: Callable[[str], None],
                screen: str | None = None, top: int = 2,
                screen_fn: Callable[[str, date], list] | None = None,
                should_stop: Callable[[], bool] | None = None) -> dict:
    """One paper-trading session: fill due orders, analyze, queue, revalue.

    Analyzes the named tickers, every held position (so the account can sell
    it), and with ``screen`` the screener's ``top`` new names. The ledger is
    saved after every step, so an interrupted session loses nothing done.
    """
    ledger = load_ledger(path)
    names = [t.strip().upper() for t in tickers if t.strip()]
    names += [t for t in ledger["positions"] if t not in names]
    if screen:
        currency, _ = screener.universe(screen)
        if currency != ledger["currency"]:
            raise ValueError(f"the {screen} universe trades in {currency}, "
                             f"but this account is in {ledger['currency']}")
        # Yesterday: today's bar may still be trading, and a pick must not use it.
        as_of = (clock() - timedelta(days=1)).date()
        picks = (screen_fn or screener.screen)(screen, as_of)
        new = [p for p in picks if p.ticker not in names][:top]
        for pick in new:
            log(f"screener picked {pick.ticker} ({pick.momentum:+.1%} over 3 months)")
        names += [p.ticker for p in new]
    if not names:
        raise ValueError("No ticker to analyze; name some (e.g. AAPL,MSFT) or use the screener")

    for trade in settle(ledger, prices, clock()):
        log(f"filled: {trade['side']} {trade['quantity']:,.4g} {trade['ticker']} @ {trade['price']:,.2f}")
    save_ledger(ledger, path)

    for name in names:
        if should_stop and should_stop():
            log("stopped before analyzing the remaining tickers")
            break
        log(f"Analyzing {name}...")
        try:
            record = decide(ledger, name, prices, decider, clock)
        except Exception as exc:  # one ticker failing must not lose the others
            log(f"skipped {name}: {exc}")
            continue
        save_ledger(ledger, path)
        log(f"{record['ticker']} ({record['analysis_date']}): {record['rating']}")

    mark_to_market(ledger, prices, clock())
    save_ledger(ledger, path)
    return ledger
