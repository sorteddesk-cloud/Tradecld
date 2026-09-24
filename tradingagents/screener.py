"""Pick candidates for the agents with a price-only momentum screen.

The agents take many minutes per ticker, so they cannot choose from a hundred
names themselves. This ranks a fixed universe on price alone, in seconds, and
hands the best few to the agents. The rule is deliberately plain:

    keep a name only if its last close is above its 50-day average and it is up
    over the last 63 trading days (about three months), then rank by that gain.

The screen is itself a strategy (momentum), and it only ever looks at closes up
to ``as_of``, so a pick never depends on a later price. Do not use it to choose
the tickers of a backtest: screening today and testing the past selects names
already known to have risen.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

LOOKBACK_DAYS = 63
SMA_DAYS = 50

# Universes and the account currency each trades in. The equity lists are the
# large-cap index members as of 2025; a symbol Yahoo no longer serves is skipped.
UNIVERSES: dict[str, tuple[str, tuple[str, ...]]] = {
    "us": ("USD", (
        "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
        "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
        "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
        "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "F", "FDX", "GD", "GE",
        "GILD", "GM", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU", "ISRG",
        "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ",
        "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX",
        "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG", "PLTR", "PM", "PYPL",
        "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS",
        "TSLA", "TXN", "UBER", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC",
        "WMT", "XOM",
    )),
    "uk": ("GBP", tuple(f"{s}.L" for s in (
        "AAL", "ABF", "ADM", "AHT", "ANTO", "AUTO", "AV", "AZN", "BA", "BARC",
        "BATS", "BEZ", "BKG", "BLND", "BNZL", "BP", "BRBY", "BT-A", "CCH", "CNA",
        "CPG", "CRDA", "DCC", "DGE", "ENT", "EXPN", "FCIT", "FRES", "GLEN", "GSK",
        "HLMA", "HLN", "HSBA", "HWDN", "IAG", "ICG", "IHG", "III", "IMB", "IMI",
        "INF", "ITRK", "JD", "KGF", "LAND", "LGEN", "LLOY", "LSEG", "MKS", "MNDI",
        "MNG", "NG", "NWG", "NXT", "PHNX", "PRU", "PSH", "PSN", "PSON", "REL",
        "RIO", "RKT", "RMV", "RR", "RTO", "SBRY", "SDR", "SGE", "SGRO", "SHEL",
        "SMIN", "SMT", "SN", "SPX", "SSE", "STAN", "SVT", "TSCO", "TW", "ULVR",
        "UTG", "UU", "VOD", "WEIR", "WPP", "WTB",
    ))),
    # Commodities through exchange-traded funds, which trade like shares; a
    # rolling futures contract would book the roll gap as profit or loss.
    "commodities": ("USD", (
        "GLD", "SLV", "PPLT", "PALL", "CPER", "USO", "BNO", "UNG",
        "DBA", "DBB", "DBC", "CORN", "WEAT", "SOYB",
    )),
}

COMMODITY_FUNDS = frozenset(UNIVERSES["commodities"][1])


@dataclass(frozen=True)
class Pick:
    ticker: str
    momentum: float   # gain over the lookback, e.g. 0.12 for +12%
    close: float
    sma: float


def universe(name: str) -> tuple[str, tuple[str, ...]]:
    try:
        return UNIVERSES[name]
    except KeyError:
        raise ValueError(f"unknown universe {name!r}; choose from {', '.join(UNIVERSES)}") from None


def rank(closes: pd.DataFrame, lookback: int = LOOKBACK_DAYS, sma_days: int = SMA_DAYS) -> list[Pick]:
    """Names above their moving average and up over the lookback, best first."""
    picks = []
    for ticker in closes.columns:
        series = closes[ticker].dropna()
        if len(series) <= max(lookback, sma_days):
            continue
        close = float(series.iloc[-1])
        sma = float(series.iloc[-sma_days:].mean())
        momentum = close / float(series.iloc[-1 - lookback]) - 1
        if close > sma and momentum > 0:
            picks.append(Pick(ticker, momentum, close, sma))
    return sorted(picks, key=lambda p: p.momentum, reverse=True)


def screen(name: str, as_of: date,
           fetch: Callable[[list[str], str, str], pd.DataFrame] | None = None) -> list[Pick]:
    """Rank a universe on closes up to and including ``as_of``."""
    if fetch is None:
        from tradingagents.dataflows.vendors.yahoo.market import get_daily_closes as fetch
    _, symbols = universe(name)
    # 63 trading days need about 90 calendar days; fetch generously for holidays.
    start = as_of - timedelta(days=150)
    closes = fetch(list(symbols), start.isoformat(), (as_of + timedelta(days=1)).isoformat())
    closes = closes[[d <= as_of for d in closes.index]]
    return rank(closes)
