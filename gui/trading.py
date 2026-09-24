"""Trading tab API: the paper account, the screener, and backtests.

The page drives the same library the CLI does (tradingagents.paper,
tradingagents.screener, tradingagents.backtest); nothing here trades on its
own rules. Long work runs as a background job the page polls.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Blueprint, jsonify, request

from .jobs import JobManager
from .pause import GATE, PauseCallback

bp = Blueprint("trading", __name__)
jobs = JobManager()
_runs = None  # the analysis RunManager, so a job and an analysis never overlap

_ACCOUNT_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def init_app(app, runs) -> None:
    global _runs
    _runs = runs
    app.register_blueprint(bp)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _default_config() -> dict:
    from tradingagents.default_config import DEFAULT_CONFIG
    return DEFAULT_CONFIG


def _paper_dir() -> Path:
    from tradingagents import paper
    return paper.default_ledger_path(_default_config()).parent


def _ledger_path(account: str | None) -> Path:
    name = (account or "ledger").strip()
    if not _ACCOUNT_RE.match(name):
        raise ValueError("account names use letters, digits, - and _ only")
    return _paper_dir() / f"{name}.json"


def _job_config(llm: dict | None) -> dict:
    """The framework config for a job, from the model settings the page sends."""
    from .run_manager import _build_config
    params = dict(llm or {})
    params.setdefault("research_depth", 1)
    return _build_config(params, _default_config())


def _busy_reason() -> str | None:
    if jobs.busy():
        return f"{jobs.current.title} is still running"
    if _runs is not None and _runs.is_anything_running():
        return "An analysis is running; wait for it or stop it first"
    return None


def _pause_callbacks(log, should_stop) -> list:
    return [PauseCallback(GATE, on_hold=lambda: log("⏸ Paused before the next step."),
                          should_stop=should_stop)]


def _error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


# --------------------------------------------------------------------------- paper


@bp.route("/api/paper/accounts")
def paper_accounts():
    folder = _paper_dir()
    names = sorted(p.stem for p in folder.glob("*.json")) if folder.exists() else []
    return jsonify({"accounts": names})


@bp.route("/api/paper/account")
def paper_account():
    from tradingagents import paper
    try:
        path = _ledger_path(request.args.get("account"))
    except ValueError as exc:
        return _error(str(exc))
    if not path.exists():
        return jsonify({"ok": True, "exists": False})
    ledger = paper.load_ledger(path)
    warning = None
    # A running session is writing this ledger; show it as saved rather than race it.
    if request.args.get("refresh") and not jobs.busy():
        try:
            prices = paper.YahooPrices()
            paper.settle(ledger, prices, _now())
            paper.mark_to_market(ledger, prices, _now())
            paper.save_ledger(ledger, path)
        except Exception as exc:
            warning = f"Prices could not be refreshed ({exc}); showing the last saved values."
            ledger = paper.load_ledger(path)
    return jsonify({"ok": True, "exists": True, "account": path.stem,
                    "summary": paper.summary(ledger), "warning": warning})


@bp.route("/api/paper/init", methods=["POST"])
def paper_init():
    from tradingagents import paper
    body = request.get_json(silent=True) or {}
    try:
        path = _ledger_path(body.get("account"))
        if path.exists() and not body.get("force"):
            return _error(f"The account {path.stem!r} already exists", 409)
        if jobs.busy():
            return _error("Wait for the running job to finish first", 409)
        ledger = paper.new_ledger(
            float(body.get("cash") or 10_000), str(body.get("currency") or "USD"),
            paper.YahooPrices(), _now(),
            max_position=float(body.get("max_position") or 0.25),
            slippage_bps=float(body.get("slippage_bps") if body.get("slippage_bps") is not None else 10),
            commission=float(body.get("commission") or 0),
            whole_shares=bool(body.get("whole_shares")),
        )
    except Exception as exc:
        return _error(str(exc))
    paper.save_ledger(ledger, path)
    return jsonify({"ok": True, "account": path.stem, "summary": paper.summary(ledger)})


@bp.route("/api/paper/screen")
def paper_screen():
    from tradingagents import screener
    universe = request.args.get("universe") or "us"
    top = int(request.args.get("top") or 10)
    try:
        currency, _ = screener.universe(universe)
        picks = screener.screen(universe, (_now() - timedelta(days=1)).date())
    except Exception as exc:
        return _error(str(exc))
    return jsonify({"ok": True, "universe": universe, "currency": currency,
                    "picks": [asdict(p) for p in picks[:top]]})


@bp.route("/api/paper/run", methods=["POST"])
def paper_run():
    from tradingagents import paper, screener
    body = request.get_json(silent=True) or {}
    reason = _busy_reason()
    if reason:
        return _error(reason, 409)
    try:
        path = _ledger_path(body.get("account"))
    except ValueError as exc:
        return _error(str(exc))
    if not path.exists():
        return _error("Open a paper account first")
    tickers = body.get("tickers") or []
    if isinstance(tickers, str):
        tickers = tickers.split(",")
    screen = body.get("screen") or None
    top = int(body.get("top") or 2)
    config = _job_config(body.get("llm"))

    def work(log, should_stop):
        ledger = paper.run_session(
            path, tickers, prices=paper.YahooPrices(),
            decider=paper.graph_decider(config, callbacks=_pause_callbacks(log, should_stop)),
            clock=_now, log=log, screen=screen, top=top, screen_fn=screener.screen,
            should_stop=should_stop,
        )
        return {"summary": paper.summary(ledger)}

    job = jobs.start("paper", f"Paper session ({path.stem})", work)
    return jsonify({"ok": True, "job": job.snapshot()})


# --------------------------------------------------------------------------- backtest


def _summary_dict(log_path: Path) -> dict:
    from tradingagents.backtest import summarize
    s = summarize(log_path)
    return {
        "resolved": s.resolved, "pending": s.pending, "unscored": s.unscored,
        "holding": s.holding, "text": s.render(),
        "by_rating": {r: asdict(score) for r, score in s.by_rating.items()},
    }


def _backtest_dir(run_id: str) -> Path:
    if not _ACCOUNT_RE.match(run_id or ""):
        raise ValueError("unknown backtest")
    return Path(_default_config()["results_dir"]) / "backtest" / run_id


def _read_plan(folder: Path) -> dict | None:
    try:
        return json.loads((folder / "plan.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@bp.route("/api/backtest", methods=["POST"])
def backtest_start():
    """Start a sweep, or with ``continue`` finish one that was stopped.

    The grid is saved beside the sweep's log as plan.json, so a stopped sweep
    can be finished later; cells already in its log are skipped.
    """
    from tradingagents.backtest import iter_grid, run_backtest
    body = request.get_json(silent=True) or {}
    reason = _busy_reason()
    if reason:
        return _error(reason, 409)
    try:
        if body.get("continue"):
            run_id = str(body["continue"])
            plan = _read_plan(_backtest_dir(run_id))
            if plan is None:
                raise ValueError("This backtest has no saved plan (it was started from the "
                                 "command line); continue it there with --run-id")
            tickers, dates, analysts = plan["tickers"], plan["dates"], plan["analysts"]
        else:
            tickers = body.get("tickers") or []
            if isinstance(tickers, str):
                tickers = tickers.split(",")
            tickers = [t.strip().upper() for t in tickers if t.strip()]
            analysts = body.get("analysts") or ["market", "news", "fundamentals"]
            if not tickers:
                raise ValueError("Name at least one ticker")
            dates = iter_grid(str(body.get("start")), str(body.get("end")),
                              int(body.get("every") or 7))
            if not dates:
                raise ValueError("That range has no dates to analyze")
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    except (KeyError, ValueError) as exc:
        return _error(str(exc))

    config = _job_config(body.get("llm"))
    folder = Path(config["results_dir"]) / "backtest" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    if not body.get("continue"):
        plan = {"tickers": tickers, "dates": dates, "analysts": analysts}
        (folder / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    log_path = folder / "trading_memory.md"

    def work(log, should_stop):
        log(f"{len(tickers)} ticker(s) x {len(dates)} date(s) = {len(tickers) * len(dates)} runs")
        callbacks = _pause_callbacks(log, should_stop)
        # One date per call, so Stop takes effect between dates; the shared
        # run id makes each call continue the same sweep and skip done cells.
        for date in dates:
            if should_stop():
                log("Stopped. Press Continue on this backtest to finish it later.")
                break
            log(f"Analyzing {', '.join(tickers)} as of {date}...")
            result = run_backtest(tickers, [date], config, selected_analysts=analysts,
                                  run_id=run_id, callbacks=callbacks)
            if result.skipped:
                log(f"{result.skipped} already done, skipped")
            for ticker, day, why in result.failures:
                log(f"failed {ticker} {day}: {why}")
        if not log_path.exists():
            raise RuntimeError("No run completed, so there is nothing to score")
        summary = _summary_dict(log_path)
        log(summary["text"])
        return {"run_id": run_id, "summary": summary}

    verb = "Continue" if body.get("continue") else "Backtest"
    job = jobs.start("backtest", f"{verb} {','.join(tickers)}", work)
    return jsonify({"ok": True, "job": job.snapshot(), "run_id": run_id})


@bp.route("/api/backtest/runs")
def backtest_runs():
    base = Path(_default_config()["results_dir"]) / "backtest"
    out = []
    if base.exists():
        for folder in sorted(base.iterdir(), reverse=True)[:30]:
            log = folder / "trading_memory.md"
            if not log.is_file():
                plan = _read_plan(folder)
                if plan:
                    total = len(plan["tickers"]) * len(plan["dates"])
                    out.append({"run_id": folder.name, "tickers": plan["tickers"], "total": total,
                                "done": 0, "can_continue": True, "resolved": 0, "pending": 0,
                                "unscored": 0, "by_rating": {}, "holding": ""})
                continue
            plan = _read_plan(folder)
            try:
                entry = {"run_id": folder.name, **_summary_dict(log)}
            except Exception as exc:
                out.append({"run_id": folder.name, "error": str(exc)})
                continue
            done = entry["resolved"] + entry["pending"] + entry["unscored"]
            if plan:
                total = len(plan["tickers"]) * len(plan["dates"])
                entry.update(tickers=plan["tickers"], total=total, done=done,
                             can_continue=done < total)
            else:
                entry.update(total=None, done=done, can_continue=False)
            out.append(entry)
    return jsonify({"runs": out})


# --------------------------------------------------------------------------- jobs


@bp.route("/api/jobs/current")
def jobs_current():
    job = jobs.current
    since = int(request.args.get("since") or 0)
    return jsonify({"job": job.snapshot(since) if job else None})


@bp.route("/api/jobs/stop", methods=["POST"])
def jobs_stop():
    return jsonify({"ok": jobs.stop()})


# --------------------------------------------------------------------------- pause


@bp.route("/api/pause", methods=["GET"])
def pause_get():
    return jsonify({"paused": GATE.paused, "since": GATE.paused_at})


@bp.route("/api/pause", methods=["POST"])
def pause_set():
    """Pause (``{"paused": true}``) or resume whatever is running."""
    want = bool((request.get_json(silent=True) or {}).get("paused"))
    if want:
        if not jobs.busy() and not (_runs is not None and _runs.is_anything_running()):
            return _error("Nothing is running to pause", 409)
        GATE.pause()
    else:
        GATE.resume()
    return jsonify({"ok": True, "paused": GATE.paused})
