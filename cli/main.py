import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import typer

from cli.display import console
from cli.run import run_analysis
from tradingagents import paper, screener
from tradingagents.backtest import iter_grid, run_backtest, summarize
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.portfolio import load_portfolio

# prompt_toolkit's win32 output module is importable only on Windows (it asserts
# the platform at import time), so gate on the platform rather than catching the
# failure — that way a genuinely broken prompt_toolkit on Windows still surfaces
# instead of silently disabling the handler below. Off Windows this stays an
# empty tuple, which `except` accepts and never matches (#1138).
if sys.platform == "win32":  # pragma: no cover - platform dependent
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError

    _NO_CONSOLE_ERRORS: tuple[type[BaseException], ...] = (NoConsoleScreenBufferError,)
else:
    _NO_CONSOLE_ERRORS = ()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


@app.callback(invoke_without_command=True)
def analyze(
    ctx: typer.Context,
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
    portfolio: str = typer.Option(
        None,
        "--portfolio",
        help="JSON file with current holdings and cash, so the trader, risk and "
        "portfolio agents size against your actual position.",
    ),
):
    """Run an analysis. This is what a bare `tradingagents` does."""
    if ctx.invoked_subcommand is not None:
        return
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    portfolio_context = None
    if portfolio:
        try:
            portfolio_context = load_portfolio(portfolio)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None

    try:
        run_analysis(checkpoint=checkpoint, portfolio=portfolio_context)
    except _NO_CONSOLE_ERRORS:
        # A terminal with no console buffer cannot host the interactive prompts.
        # Emit one actionable line on stderr instead of a prompt_toolkit
        # traceback; plain text, since rich may not render here either (#1138).
        typer.echo(
            "Error: no Windows console available. The interactive CLI needs a real "
            "console buffer — run it from Windows Terminal, PowerShell, or cmd.exe "
            "rather than a piped or embedded terminal.",
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command()
def backtest(
    tickers: str = typer.Argument(..., help="Comma-separated tickers, e.g. NVDA,AAPL"),
    start: str = typer.Option(..., "--start", help="First analysis date, YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="Last analysis date, YYYY-MM-DD"),
    every: int = typer.Option(7, "--every", help="Days between analysis dates"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts to run; omit for all four"
    ),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock or crypto"),
    portfolio: str = typer.Option(
        None, "--portfolio", help="JSON file with holdings and cash, held constant across the grid"
    ),
    run_id: str = typer.Option(
        None, "--run-id", help="Continue an earlier sweep: its cells are skipped and its log reused"
    ),
):
    """Score past decisions over a grid of tickers and dates."""

    try:
        dates = iter_grid(start, end, every)
        book = load_portfolio(portfolio) if portfolio else None
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    names = [t.strip() for t in tickers.split(",") if t.strip()]
    if not names:
        console.print("[red]No ticker to analyze; pass them comma-separated, e.g. NVDA,AAPL[/red]")
        raise typer.Exit(code=1)

    kwargs = {"asset_type": asset_type, "portfolio": book, "run_id": run_id}
    if analysts:
        kwargs["selected_analysts"] = [a.strip().lower() for a in analysts.split(",") if a.strip()]

    try:
        result = run_backtest(names, dates, DEFAULT_CONFIG, **kwargs)
    except Exception as exc:  # a missing key or an unknown analyst is a setup error
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(summarize(result).render())
    console.print(f"\nRan {result.cells_run} cells, skipped {result.skipped}. Log: {result.log_path}")
    for ticker, date, reason in result.failures:
        console.print(f"[yellow]failed:[/yellow] {ticker} {date}: {reason}")
    for ticker, reason in result.settlement_failures:
        console.print(f"[yellow]unsettled:[/yellow] {ticker}: {reason}")


paper_app = typer.Typer(help="Paper trading: act on the ratings with a simulated account.")
app.add_typer(paper_app, name="paper")

_LEDGER_OPTION = typer.Option(None, "--ledger", help="Ledger file; defaults to ~/.tradingagents/paper/ledger.json")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ledger_path(ledger: str | None):
    return ledger or paper.default_ledger_path(DEFAULT_CONFIG)


@paper_app.command("init")
def paper_init(
    cash: float = typer.Option(10_000.0, "--cash", help="Starting cash"),
    currency: str = typer.Option("USD", "--currency", help="USD for US stocks and crypto, GBP for London (.L)"),
    max_position: float = typer.Option(0.25, "--max-position", help="Largest share of equity in one name"),
    slippage_bps: float = typer.Option(10.0, "--slippage-bps", help="Price penalty per fill, in basis points"),
    commission: float = typer.Option(0.0, "--commission", help="Flat fee per trade"),
    whole_shares: bool = typer.Option(False, "--whole-shares", help="Trade whole shares only"),
    force: bool = typer.Option(False, "--force", help="Replace an existing account"),
    ledger: str = _LEDGER_OPTION,
):
    """Open a paper account."""
    path = _ledger_path(ledger)
    if Path(path).exists() and not force:
        console.print(f"[red]A paper account already exists at {path}; pass --force to replace it.[/red]")
        raise typer.Exit(code=1)
    try:
        book = paper.new_ledger(cash, currency, paper.YahooPrices(), _now(), max_position,
                                slippage_bps, commission, whole_shares)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    paper.save_ledger(book, path)
    console.print(f"Opened a {book['currency']} paper account with {cash:,.2f} at {path}")


_SCREEN_TOP = 2


def _screen_as_of():
    # Yesterday: today's bar may still be trading, and a pick must not use it.
    return (_now() - timedelta(days=1)).date()


@paper_app.command("screen")
def paper_screen(
    universe: str = typer.Option("us", "--universe", help="us, uk or commodities"),
    top: int = typer.Option(10, "--top", help="How many to show"),
):
    """Rank a universe on 3-month momentum, above the 50-day average (price only, no AI)."""
    try:
        picks = screener.screen(universe, _screen_as_of())
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    if not picks:
        console.print(f"Nothing in {universe} is both above its 50-day average and up over 3 months.")
    for i, pick in enumerate(picks[:top], 1):
        console.print(f"{i:>2}. {pick.ticker}: {pick.momentum:+.1%} over 3 months, "
                      f"{pick.close / pick.sma - 1:+.1%} above its 50-day average")


@paper_app.command("run")
def paper_run(
    tickers: str = typer.Argument("", help="Comma-separated tickers, e.g. AAPL,MSFT; optional with --screen"),
    screen: str = typer.Option(None, "--screen", help="Also let the screener pick: us, uk or commodities"),
    top: int = typer.Option(_SCREEN_TOP, "--top", help="How many new names the screener adds"),
    ledger: str = _LEDGER_OPTION,
):
    """Fill due orders, then analyze each ticker and queue its order for the next open.

    Held positions are always analyzed, so the account can decide to sell them.
    """
    try:
        book = paper.run_session(
            _ledger_path(ledger), tickers.split(","), prices=paper.YahooPrices(),
            decider=paper.graph_decider(DEFAULT_CONFIG), clock=_now, log=console.print,
            screen=screen, top=top, screen_fn=screener.screen,
        )
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print("")
    console.print(paper.report(book))


@paper_app.command("status")
def paper_status(ledger: str = _LEDGER_OPTION):
    """Fill due orders and show the account."""
    path = _ledger_path(ledger)
    prices = paper.YahooPrices()
    try:
        book = paper.load_ledger(path)
        paper.settle(book, prices, _now())
        paper.mark_to_market(book, prices, _now())
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    paper.save_ledger(book, path)
    console.print(paper.report(book))


if __name__ == "__main__":
    app()
