"""Paper trading fills at an open that came after the decision, and sizes by rating."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from tradingagents import paper

NY = "America/New_York"


class FakePrices:
    def __init__(self, bars: dict[str, dict[date, tuple[float, float]]], tz: str = NY,
                 currency: dict[str, str] | None = None):
        self.bars, self.tz, self.quotes = bars, tz, currency or {}

    def sessions(self, ticker, start, end):
        lo, hi = date.fromisoformat(start), date.fromisoformat(end)
        rows = {d: v for d, v in sorted(self.bars[ticker].items()) if lo <= d < hi}
        frame = pd.DataFrame({"open": [v[0] for v in rows.values()],
                              "close": [v[1] for v in rows.values()]}, index=list(rows))
        return frame, self.tz

    def currency(self, ticker):
        return self.quotes.get(ticker, "USD")


def _utc(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


MON, TUE, WED = date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23)


def _prices():
    return FakePrices({
        "AAPL": {MON: (100.0, 101.0), TUE: (102.0, 104.0), WED: (105.0, 103.0)},
        "SPY": {MON: (500.0, 500.0), TUE: (505.0, 505.0), WED: (510.0, 510.0)},
    })


def _account(prices, now, **rules):
    return paper.new_ledger(10_000, "usd", prices, now, **rules)


@pytest.mark.unit
def test_analysis_uses_the_last_session_whose_close_has_passed():
    frame, tz = _prices().sessions("AAPL", "2026-09-01", "2026-09-30")
    # 20:00 UTC is 16:00 in New York: Tuesday's close has just passed.
    assert paper.last_completed_session(frame, tz, _utc(2026, 9, 22, 20)) == TUE
    # 15:00 UTC is 11:00 in New York: Tuesday is still trading.
    assert paper.last_completed_session(frame, tz, _utc(2026, 9, 22, 15)) == MON


@pytest.mark.unit
def test_a_fill_never_uses_an_open_known_before_the_decision():
    frame, tz = _prices().sessions("AAPL", "2026-09-01", "2026-09-30")
    # Decided before Tuesday's 09:30 open: Tuesday's open is fair.
    assert paper.fill_session(frame, tz, _utc(2026, 9, 22, 12), MON) == TUE
    # Decided after Tuesday opened: the next open is Wednesday's.
    assert paper.fill_session(frame, tz, _utc(2026, 9, 22, 15), MON) == WED
    # No session has opened since: the order waits.
    assert paper.fill_session(frame, tz, _utc(2026, 9, 23, 15), TUE) is None


@pytest.mark.unit
def test_buy_is_queued_then_filled_at_the_next_open_with_slippage():
    prices = _prices()
    now = _utc(2026, 9, 22, 12)  # before Tuesday's open; Monday is the last close
    ledger = _account(prices, now)
    seen = []

    def decider(ticker, trade_date, portfolio):
        seen.append((ticker, trade_date, portfolio.cash))
        return "Buy"

    record = paper.decide(ledger, "aapl", prices, decider, lambda: now)
    assert seen == [("AAPL", "2026-09-21", 10_000)]
    assert record["rating"] == "Buy" and len(ledger["pending"]) == 1
    assert ledger["trades"] == []

    fills = paper.settle(ledger, prices, _utc(2026, 9, 23, 21))
    assert len(fills) == 1 and fills[0]["date"] == "2026-09-22"
    assert fills[0]["price"] == pytest.approx(102.0 * 1.001)
    position = ledger["positions"]["AAPL"]
    # 25% of 10,000 at the slipped price, in fractional shares.
    assert position["quantity"] * fills[0]["price"] == pytest.approx(2_500, abs=0.01)
    assert ledger["cash"] == pytest.approx(7_500, abs=0.01)
    assert ledger["pending"] == []


@pytest.mark.unit
@pytest.mark.parametrize("rating,current,expected", [
    ("Buy", 0.0, 0.25), ("Buy", 0.4, 0.25),
    ("Overweight", 0.0, 0.125), ("Overweight", 0.2, 0.2),
    ("Hold", 0.2, None), ("REVIEW", 0.2, None),
    ("Underweight", 0.2, 0.1), ("Sell", 0.2, 0.0),
])
def test_rating_sizes(rating, current, expected):
    assert paper.target_weight(rating, current, 0.25) == expected


@pytest.mark.unit
def test_sell_closes_the_position_and_records_the_realized_gain():
    prices = _prices()
    ledger = _account(prices, _utc(2026, 9, 22, 12), slippage_bps=0)
    ledger["cash"] = 9_000.0
    ledger["positions"]["AAPL"] = {"quantity": 10.0, "average_price": 100.0,
                                   "last_price": 101.0, "quote_scale": 1.0}
    paper.decide(ledger, "AAPL", prices, lambda *a: "Sell", lambda: _utc(2026, 9, 22, 12))
    fills = paper.settle(ledger, prices, _utc(2026, 9, 23, 21))
    assert fills[0]["side"] == "sell" and fills[0]["realized"] == pytest.approx(20.0)
    assert "AAPL" not in ledger["positions"]
    assert ledger["cash"] == pytest.approx(9_000 + 10 * 102.0)


@pytest.mark.unit
def test_bearish_calls_on_names_not_held_and_holds_queue_nothing():
    prices = _prices()
    ledger = _account(prices, _utc(2026, 9, 22, 12))
    for rating in ("Sell", "Underweight", "Hold", "REVIEW"):
        paper.decide(ledger, "AAPL", prices, lambda *a, r=rating: r, lambda: _utc(2026, 9, 22, 12))
    assert ledger["pending"] == [] and len(ledger["decisions"]) == 4


@pytest.mark.unit
def test_buys_are_capped_by_cash_and_never_borrow():
    prices = _prices()
    ledger = _account(prices, _utc(2026, 9, 22, 12), max_position=1.0)
    ledger["cash"] = 50.0
    ledger["positions"]["SPY"] = {"quantity": 19.9, "average_price": 500.0,
                                  "last_price": 500.0, "quote_scale": 1.0}
    paper.decide(ledger, "AAPL", prices, lambda *a: "Buy", lambda: _utc(2026, 9, 22, 12))
    paper.settle(ledger, prices, _utc(2026, 9, 23, 21))
    assert ledger["cash"] >= 0


@pytest.mark.unit
def test_london_prices_in_pence_are_booked_in_pounds():
    bars = {"BARC.L": {MON: (200.0, 201.0), TUE: (210.0, 205.0)},
            "^FTSE": {MON: (8000.0, 8000.0), TUE: (8050.0, 8050.0)}}
    prices = FakePrices(bars, tz="Europe/London", currency={"BARC.L": "GBp", "^FTSE": "GBP"})
    now = _utc(2026, 9, 22, 6)  # 07:00 in London, before the 08:00 open
    ledger = paper.new_ledger(1_000, "GBP", prices, now, slippage_bps=0)
    paper.decide(ledger, "BARC.L", prices, lambda *a: "Buy", lambda: now)
    fills = paper.settle(ledger, prices, _utc(2026, 9, 22, 17))
    assert fills[0]["price"] == pytest.approx(2.10)
    assert ledger["cash"] == pytest.approx(750, abs=0.01)
    # The agents see the average price in the pence the data is quoted in.
    assert paper.portfolio_context(ledger).positions[0].average_price == pytest.approx(210.0)


@pytest.mark.unit
def test_a_ticker_in_another_currency_is_refused_before_the_agents_run():
    prices = FakePrices({"BARC.L": {MON: (200.0, 201.0)}, "SPY": {MON: (500.0, 500.0)}},
                        currency={"BARC.L": "GBp"})
    ledger = _account(prices, _utc(2026, 9, 22, 12))
    with pytest.raises(ValueError, match="quoted in GBp"):
        paper.decide(ledger, "BARC.L", prices,
                     lambda *a: pytest.fail("agents ran"), lambda: _utc(2026, 9, 22, 12))


@pytest.mark.unit
def test_ledger_round_trips_and_reports(tmp_path):
    prices = _prices()
    now = _utc(2026, 9, 22, 12)
    ledger = _account(prices, now)
    paper.decide(ledger, "AAPL", prices, lambda *a: "Buy", lambda: now)
    later = now + timedelta(days=1, hours=9)
    paper.settle(ledger, prices, later)
    paper.mark_to_market(ledger, prices, later)
    path = tmp_path / "paper" / "ledger.json"
    paper.save_ledger(ledger, path)
    loaded = paper.load_ledger(path)
    assert loaded == ledger
    text = paper.report(loaded)
    assert "AAPL" in text and "SPY over the same period" in text


@pytest.mark.unit
def test_missing_ledger_says_how_to_create_one(tmp_path):
    with pytest.raises(ValueError, match="paper init"):
        paper.load_ledger(tmp_path / "none.json")


@pytest.mark.unit
def test_futures_are_refused_with_a_fund_suggested():
    prices = _prices()
    ledger = _account(prices, _utc(2026, 9, 22, 12))
    with pytest.raises(ValueError, match="GLD"):
        paper.decide(ledger, "GOLD", prices, lambda *a: pytest.fail("agents ran"),
                     lambda: _utc(2026, 9, 22, 12))
