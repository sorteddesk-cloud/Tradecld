"""The screener keeps rising names above their average, and never looks past as_of."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from tradingagents import screener

START = date(2026, 3, 2)


def _closes(**paths: list[float]) -> pd.DataFrame:
    days = [START + timedelta(days=i) for i in range(len(next(iter(paths.values()))))]
    return pd.DataFrame(paths, index=days)


def _line(first: float, last: float, n: int = 100) -> list[float]:
    return [first + (last - first) * i / (n - 1) for i in range(n)]


@pytest.mark.unit
def test_rank_keeps_rising_names_above_their_average_best_first():
    closes = _closes(UP=_line(100, 150), UPMORE=_line(100, 200), DOWN=_line(150, 100),
                     # Up over three months but just fell through its average.
                     FADING=_line(100, 180, 90) + [120.0] * 10)
    picks = screener.rank(closes)
    assert [p.ticker for p in picks] == ["UPMORE", "UP"]
    assert picks[0].momentum > picks[1].momentum > 0


@pytest.mark.unit
def test_rank_skips_names_without_enough_history():
    assert screener.rank(_closes(NEW=_line(10, 20, 40))) == []


@pytest.mark.unit
def test_screen_ignores_closes_after_as_of():
    # Flat, then a spike after as_of: a screen that peeked would pick it.
    path = [100.0] * 100 + [200.0] * 10
    frame = _closes(SPIKE=path, STEADY=_line(100, 130, 110))
    as_of = frame.index[99]
    requested = []

    def fetch(symbols, start, end):
        requested.append((start, end))
        return frame

    picks = screener.screen("us", as_of, fetch=fetch)
    assert [p.ticker for p in picks] == ["STEADY"]
    assert requested[0][1] == (as_of + timedelta(days=1)).isoformat()


@pytest.mark.unit
def test_universes_name_their_currency():
    assert screener.universe("us")[0] == "USD"
    assert screener.universe("uk")[0] == "GBP" and all(s.endswith(".L") for s in screener.universe("uk")[1])
    assert "GLD" in screener.COMMODITY_FUNDS
    with pytest.raises(ValueError, match="unknown universe"):
        screener.universe("mars")


@pytest.mark.unit
def test_batch_closes_keep_requested_names_and_drop_empty_ones(monkeypatch):
    from tradingagents.dataflows.vendors.yahoo import market

    index = pd.DatetimeIndex(["2026-09-21", "2026-09-22"], tz="America/New_York")
    columns = pd.MultiIndex.from_product([["Close", "Volume"], ["AAPL", "GC=F", "GONE"]])
    data = pd.DataFrame([[1.0, 2.0, None, 10, 20, None], [1.5, 2.5, None, 11, 21, None]],
                        index=index, columns=columns)
    monkeypatch.setattr(market.yf, "download", lambda *a, **kw: data)
    closes = market.get_daily_closes(["AAPL", "GOLD", "GONE"], "2026-09-01", "2026-09-23")
    assert list(closes.columns) == ["AAPL", "GOLD"]
    assert list(closes.index) == [date(2026, 9, 21), date(2026, 9, 22)]
