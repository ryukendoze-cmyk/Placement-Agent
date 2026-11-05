from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_scraper.pipeline import EXTRACTOR_REGISTRY, PortalResult, run_pipeline  # noqa: E402

app = typer.Typer(
    help="Advanced India-focused job scraper (stealth_harvester).",
    no_args_is_help=True,
)
console = Console()


def _keywords(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


@app.command()
def scrape(
    keywords: str = typer.Option(
        "python",
        "--keywords",
        "-k",
        help="Comma-separated search terms, e.g. 'Python, FastAPI, Full Stack'",
    ),
    location: str = typer.Option("India", "--location", "-l", help="City or 'India'/'Remote'"),
    days_old: int = typer.Option(7, "--days-old", "-d", help="Freshness window in days"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max jobs per portal"),
    portals: str = typer.Option("", "--portals", "-p", help="Comma-separated portal subset (default: all)"),
    output: Path = typer.Option(Path("output"), "--output", "-o", help="Output directory"),
) -> None:
    """Scrape all (or a subset of) Indian job portals and write JSON outputs."""
    kw = _keywords(keywords)
    portal_list = [p.strip() for p in portals.split(",") if p.strip()] or None
    rows: list[tuple[str, str, str, str]] = []

    def progress(portal: str, result: PortalResult) -> None:
        status = f"[red]error: {result.error}[/red]" if result.error else "[green]ok[/green]"
        rows.append((portal, str(len(result.jobs)), f"{result.duration_s:.1f}s", status))
        console.print(
            f"  [cyan]{portal}[/cyan]: {len(result.jobs)} jobs "
            f"({result.duration_s:.1f}s) - {result.error or 'ok'}"
        )

    with console.status("[bold green]Scraping portals...[/bold green]"):
        out = asyncio.run(
            run_pipeline(
                keywords=kw,
                location=location,
                days_old=days_old,
                limit=limit,
                portals=portal_list,
                output_dir=output,
                progress=progress,
            )
        )

    table = Table(title="Stealth Harvester - India Job Scrape")
    table.add_column("Portal", style="cyan")
    table.add_column("Jobs", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Status")
    for row in rows:
        table.add_row(*row)
    console.print(table)

    c = out["counts"]
    console.rule(characters="-")
    summary = Table(title="Summary")
    summary.add_row("Keywords", ", ".join(kw))
    summary.add_row("Location", location)
    summary.add_row("Freshness", f"{days_old} days")
    summary.add_row("Raw jobs", str(c["raw"]))
    summary.add_row("Normalized (deduped)", str(c["normalized"]))
    summary.add_row("Deduped away", str(c["deduped_away"]))
    summary.add_row("Portals ok / failed", f"{c['portals_ok']} / {c['portals_failed']}")
    summary.add_row("Raw output", out["outputs"]["raw"])
    summary.add_row("Normalized output", out["outputs"]["normalized"])
    console.print(summary)

    failed = [r.portal for r in out["results"] if r.error]
    if failed:
        console.print(f"[yellow]Portals with errors: {', '.join(failed)}[/yellow]")
    if c["normalized"] == 0:
        raise typer.Exit(code=1)


@app.command()
def portals() -> None:
    """List available portal extractors."""
    table = Table(title="Available Portals")
    table.add_column("Portal")
    table.add_row("naukri")
    for name in sorted(set(EXTRACTOR_REGISTRY) - {"naukri"}):
        table.add_row(name)
    console.print(table)


if __name__ == "__main__":
    app()
