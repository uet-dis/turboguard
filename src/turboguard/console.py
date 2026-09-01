"""Shared Rich console for TurboGuard pipeline output.

Provides a singleton ``Console`` and helper functions that produce
styled, consistent terminal output across all dataset scripts.

Usage::

    from turboguard.console import console, header, step, done, metric

    header("Train TurboGuard", dataset="unsw")
    step("Training VQ-VAE")
    done("Model saved", path=run_dir)
    metric("FPR", 1.23)
"""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
"""Singleton console instance — import this everywhere."""


def header(title: str, dataset: str = "") -> None:
    """Prints a prominent section header.

    Args:
        title: Section title (e.g. "Train Baselines").
        dataset: Dataset name shown as subtitle.
    """
    subtitle = f"[dim]{dataset.upper()}[/dim]" if dataset else None
    console.print()
    console.print(
        Panel(
            f"[bold white]{title}[/bold white]",
            subtitle=subtitle,
            border_style="cyan",
            width=72,
        )
    )


def step(msg: str) -> None:
    """Prints a pipeline step message.

    Args:
        msg: Step description (e.g. "Training XGBoost").
    """
    console.print(f"  [cyan]▸[/cyan] {msg}")


def substep(msg: str) -> None:
    """Prints an indented sub-step message.

    Args:
        msg: Sub-step description.
    """
    console.print(f"    [dim]→[/dim] {msg}")


def done(msg: str, path: Path | str | None = None) -> None:
    """Prints a success message with optional path.

    Args:
        msg: Completion message.
        path: Optional output path to display.
    """
    if path:
        console.print(f"  [green]✓[/green] {msg}: [bold]{path}[/bold]")
    else:
        console.print(f"  [green]✓[/green] {msg}")


def warn(msg: str) -> None:
    """Prints a warning message.

    Args:
        msg: Warning text.
    """
    console.print(f"  [yellow]⚠[/yellow] {msg}")


def metric(name: str, value: float, unit: str = "%") -> None:
    """Prints a single metric value.

    Args:
        name: Metric name (e.g. "FPR").
        value: Metric value.
        unit: Unit suffix (default "%").
    """
    console.print(f"  [bold]{name}:[/bold] {value:.2f}{unit}")


def metrics_line(**kwargs: float) -> None:
    """Prints multiple metrics on one line.

    Args:
        **kwargs: Metric name/value pairs.
    """
    parts = [f"[bold]{k}:[/bold] {v:.1f}%" for k, v in kwargs.items()]
    console.print("  " + "  │  ".join(parts))


def sector_stats(name: str, total: int, benign: int, attack: int) -> None:
    """Prints sector split statistics.

    Args:
        name: Sector name (e.g. "A", "B", "C").
        total: Total samples.
        benign: Benign sample count.
        attack: Attack sample count.
    """
    console.print(
        f"  Sector [bold]{name}[/bold]: {total:,} "
        f"([green]benign={benign:,}[/green], [red]attack={attack:,}[/red])"
    )


def results_table(title: str, rows: list[dict], columns: list[str]) -> None:
    """Prints a formatted results table.

    Args:
        title: Table title.
        rows: List of dicts, each containing column values.
        columns: Column names to display.
    """
    table = Table(title=title, show_lines=False, border_style="dim")
    for col in columns:
        justify = "left" if col == columns[0] else "right"
        table.add_column(col, justify=justify)
    for row in rows:
        table.add_row(*[str(row.get(c, "")) for c in columns])
    console.print(table)
