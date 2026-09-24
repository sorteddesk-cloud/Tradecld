"""The web GUI: its guard, a full analysis through its worker, and the Trading tab."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

pytest.importorskip("flask")

from gui import agent_map, app as gui_app, run_manager, trading  # noqa: E402
from tests.test_graph_end_to_end import (  # noqa: E402, F401  (offline is a fixture)
    TRADE_DATE,
    ScriptedModel,
    _Client,
    offline,
)
from tests.test_paper import _prices  # noqa: E402
from tradingagents import paper  # noqa: E402
from tradingagents.graph import trading_graph  # noqa: E402
from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS  # noqa: E402


@pytest.fixture
def client():
    return gui_app.app.test_client()


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point every framework path at a temp dir.

    Looked up at call time: other tests reload the config module, and the GUI
    reads whichever DEFAULT_CONFIG is current.
    """
    import tradingagents.default_config as default_config

    for key, sub in (("results_dir", "logs"), ("data_cache_dir", "cache"),
                     ("memory_log_path", "memory/log.md")):
        monkeypatch.setitem(default_config.DEFAULT_CONFIG, key, str(tmp_path / sub))
    return tmp_path


def _wait_for_job():
    for _ in range(200):
        if not trading.jobs.busy():
            return trading.jobs.current
        time.sleep(0.02)
    raise AssertionError("job did not finish")


@pytest.mark.unit
def test_the_page_serves_with_the_trading_tab(client):
    page = client.get("/").get_data(as_text=True)
    assert 'id="tab-trading"' in page and "/static/js/trading.js" in page
    assert "brevity-input" not in page  # not an option this framework has
    assert client.get("/static/js/trading.js").status_code == 200


@pytest.mark.unit
def test_other_sites_cannot_drive_the_server(client):
    assert client.get("/api/health", headers={"Host": "attacker.example:5000"}).status_code == 403
    cross = client.post("/api/env", json={"OPENAI_API_KEY": "x"},
                        headers={"Origin": "https://attacker.example"})
    assert cross.status_code == 403
    same = client.post("/api/jobs/stop", json={}, headers={"Origin": "http://localhost"})
    assert same.status_code == 200


@pytest.mark.unit
def test_clear_node_names_match_the_graph():
    assert {
        s.clear_node: s.agent_node for s in ANALYST_NODE_SPECS.values()} == agent_map.CLEAR_NODE_TO_DISPLAY
    assert {a["display"] for a in agent_map.ANALYSTS.values()} == {
        s.agent_node for s in ANALYST_NODE_SPECS.values()}


@pytest.mark.unit
def test_an_analysis_through_the_gui_worker_reaches_a_logged_decision(
        home, monkeypatch, offline):  # noqa: F811
    monkeypatch.setattr(trading_graph, "create_llm_client", lambda **k: _Client(ScriptedModel()))
    run = run_manager.Run({"ticker": "NVDA", "date": TRADE_DATE, "research_depth": 1,
                           "analysts": ["market", "social", "news", "fundamentals"]})
    events = []
    run.emit = events.append

    run_manager._worker(run)

    assert run.status == "completed", run.error
    assert run.rating == "Overweight"
    assert set(run.roster.values()) == {"completed"}
    # Each analyst was seen working before it finished; the first one from the start.
    opening = next(e for e in events if e.get("type") == "agents_init")["agents"]
    assert opening["Market Analyst"] == "in_progress"
    started = {e["agent"] for e in events
               if e.get("type") == "agent_update" and e["status"] == "in_progress"}
    assert {"Sentiment Analyst", "News Analyst", "Fundamentals Analyst"} <= started
    report = home / "logs" / "NVDA" / TRADE_DATE / "reports" / "complete_report.md"
    assert "Overweight" in report.read_text(encoding="utf-8")
    assert "Overweight" in (home / "memory" / "log.md").read_text(encoding="utf-8")


@pytest.mark.unit
def test_a_future_date_is_refused(home):
    run = run_manager.Run({"ticker": "NVDA", "date": "2999-01-01", "analysts": ["market"]})
    run.emit = lambda e: None
    run_manager._worker(run)
    assert run.status == "error" and "future" in run.error


@pytest.mark.unit
def test_paper_account_session_and_status(client, home, monkeypatch):
    monkeypatch.setattr(paper, "YahooPrices", _prices)
    monkeypatch.setattr(trading, "_now", lambda: datetime(2026, 9, 22, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(paper, "graph_decider", lambda config, callbacks=None: lambda *a: "Buy")

    assert client.get("/api/paper/account").get_json()["exists"] is False
    opened = client.post("/api/paper/init", json={"account": "test", "cash": 5000})
    assert opened.status_code == 200, opened.get_json()
    again = client.post("/api/paper/init", json={"account": "test"})
    assert again.status_code == 409
    assert client.post("/api/paper/init", json={"account": "../x"}).status_code == 400
    assert client.get("/api/paper/accounts").get_json()["accounts"] == ["test"]

    started = client.post("/api/paper/run", json={"account": "test", "tickers": "AAPL"})
    assert started.status_code == 200, started.get_json()
    job = _wait_for_job()
    assert job.status == "completed", job.error
    assert any("AAPL (2026-09-21): Buy" in line for line in job.log)

    account = client.get("/api/paper/account?account=test&refresh=1").get_json()
    summary = account["summary"]
    # Decided before Tuesday's open, so Tuesday's open filled the buy.
    assert [p["ticker"] for p in summary["positions"]] == ["AAPL"]
    assert summary["trades"][0]["date"] == "2026-09-22"


@pytest.mark.unit
def test_backtest_validates_before_starting(client, home):
    bad = client.post("/api/backtest", json={"tickers": "", "start": "2026-01-01", "end": "2026-02-01"})
    assert bad.status_code == 400
    backwards = client.post("/api/backtest", json={"tickers": "AAPL", "start": "2026-02-01",
                                                   "end": "2026-01-01"})
    assert backwards.status_code == 400 and "before" in backwards.get_json()["error"]
    assert client.get("/api/backtest/runs").get_json() == {"runs": []}


@pytest.mark.unit
def test_a_job_and_an_analysis_never_overlap(client, home, monkeypatch):
    release = []

    def slow(log, should_stop):
        while not release and not should_stop():
            time.sleep(0.01)

    trading.jobs.start("paper", "Slow job", slow)
    try:
        refused = client.post("/api/analyze", json={"ticker": "AAPL"})
        assert refused.status_code == 409 and "Slow job" in refused.get_json()["error"]
        assert client.post("/api/backtest", json={"tickers": "AAPL"}).status_code == 409
    finally:
        release.append(True)
        _wait_for_job()


# --------------------------------------------------------------------------- pause


@pytest.fixture
def gate():
    from gui.pause import GATE
    GATE.resume()
    yield GATE
    GATE.resume()


def _scripted_with_callbacks(monkeypatch):
    """Scripted models that carry the graph's callbacks; returns their shared call log."""
    calls: list = []

    def factory(**kwargs):
        model = ScriptedModel(callbacks=kwargs.get("callbacks"))
        model.calls = calls
        return _Client(model)
    monkeypatch.setattr(trading_graph, "create_llm_client", factory)
    return calls


@pytest.mark.unit
def test_a_paused_analysis_waits_then_finishes_on_resume(home, monkeypatch, offline, gate):  # noqa: F811
    import threading

    calls = _scripted_with_callbacks(monkeypatch)
    run = run_manager.Run({"ticker": "NVDA", "date": TRADE_DATE, "research_depth": 1,
                           "analysts": ["market"]})
    events = []
    run.emit = events.append
    gate.pause()
    worker = threading.Thread(target=run_manager._worker, args=(run,))
    worker.start()
    time.sleep(0.6)
    assert calls == [] and run.status == "running"  # held before the first model call
    assert any("Paused" in e.get("message", "") for e in events)

    gate.resume()
    worker.join(timeout=30)
    assert run.status == "completed", run.error
    assert calls and not gate.paused


@pytest.mark.unit
def test_stop_releases_a_paused_run(home, monkeypatch, offline, gate):  # noqa: F811
    import threading

    _scripted_with_callbacks(monkeypatch)
    run = run_manager.Run({"ticker": "NVDA", "date": TRADE_DATE, "analysts": ["market"]})
    run.emit = lambda e: None
    gate.pause()
    worker = threading.Thread(target=run_manager._worker, args=(run,))
    worker.start()
    time.sleep(0.3)
    run.stop()
    worker.join(timeout=30)
    assert run.status == "stopped"
    assert not gate.paused  # a finished run leaves nothing paused


@pytest.mark.unit
def test_pause_route_needs_something_running(client, gate):
    assert client.post("/api/pause", json={"paused": True}).status_code == 409
    release = []
    trading.jobs.start("paper", "Slow job", lambda log, stop: [time.sleep(0.01) for _ in iter(
        lambda: bool(release) or stop(), True)])
    try:
        assert client.post("/api/pause", json={"paused": True}).get_json()["paused"] is True
        assert client.get("/api/jobs/current").get_json()["job"]["paused"] is True
        assert client.post("/api/pause", json={"paused": False}).get_json()["paused"] is False
    finally:
        release.append(True)
        _wait_for_job()


# --------------------------------------------------------------------------- continue


@pytest.mark.unit
def test_a_stopped_backtest_can_be_continued(client, home, monkeypatch):
    import tradingagents.backtest as backtest

    swept = []

    def fake_run_backtest(tickers, dates, config, **kw):
        swept.append((tuple(tickers), tuple(dates), kw["run_id"], kw["selected_analysts"]))
        return backtest.BacktestResult(run_id=kw["run_id"], log_path=None)

    monkeypatch.setattr(backtest, "run_backtest", fake_run_backtest)
    started = client.post("/api/backtest", json={
        "tickers": "aapl", "start": "2026-06-01", "end": "2026-06-15", "every": 7,
        "analysts": ["market"]}).get_json()
    run_id = started["run_id"]
    _wait_for_job()
    assert [s[1] for s in swept] == [("2026-06-01",), ("2026-06-08",), ("2026-06-15",)]

    listed = client.get("/api/backtest/runs").get_json()["runs"]
    assert listed[0]["run_id"] == run_id and listed[0]["can_continue"] is True
    assert listed[0]["total"] == 3

    swept.clear()
    again = client.post("/api/backtest", json={"continue": run_id})
    assert again.status_code == 200
    _wait_for_job()
    assert {s[2] for s in swept} == {run_id} and swept[0][0] == ("AAPL",)
    assert swept[0][3] == ["market"]

    assert client.post("/api/backtest", json={"continue": "nope"}).status_code == 400
    assert client.post("/api/backtest", json={"continue": "../../etc"}).status_code == 400
