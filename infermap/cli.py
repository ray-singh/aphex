"""CLI entry point — Typer app with Rich output."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

app = typer.Typer(
    name="infermap",
    help="Hardware-aware ML deployment optimization and recommendation.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True, style="bold red")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@app.command()
def analyze(
    model_path: Path = typer.Argument(..., help="Path to a saved PyTorch model (.pt / .pkl)"),
) -> None:
    """Inspect a model and profile the current hardware."""
    from infermap.inspector import inspect_model
    from infermap.profiler import profile_hardware

    with console.status("[bold green]Profiling hardware..."):
        hw = profile_hardware()

    with console.status("[bold green]Inspecting model..."):
        info = inspect_model(model_path)

    _print_hardware(hw)
    _print_model_info(info)


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


@app.command()
def preflight(
    model_path: Path = typer.Argument(..., help="Path to a saved PyTorch model"),
    dtype: str = typer.Option("fp32", help="Precision to check: fp32, fp16, bf16, int8, int4"),
    target_throughput: Optional[float] = typer.Option(
        None, "--target-throughput", help="Required throughput in req/s"
    ),
) -> None:
    """Run a fast feasibility check before benchmarking."""
    from infermap.inspector import inspect_model
    from infermap.profiler import profile_hardware
    from infermap.preflight import run_preflight

    with console.status("[bold green]Profiling hardware..."):
        hw = profile_hardware()

    with console.status("[bold green]Inspecting model..."):
        info = inspect_model(model_path)

    result = run_preflight(info, hw, dtype=dtype, target_throughput=target_throughput)
    _print_preflight(result)

    if not result.feasible or result.category == "impossible":
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@app.command()
def benchmark(
    model_path: Path = typer.Argument(..., help="Path to a saved PyTorch model"),
    input_shape: str = typer.Option(
        "3,224,224", "--input-shape", help="Input tensor shape (no batch dim), e.g. 3,224,224"
    ),
    batch_size: int = typer.Option(1, help="Batch size for benchmarking"),
    warmup: int = typer.Option(10, help="Warm-up iterations"),
    iters: int = typer.Option(100, help="Measurement iterations"),
) -> None:
    """Benchmark all candidate deployment strategies."""
    import torch
    from infermap.inspector import inspect_model
    from infermap.profiler import profile_hardware
    from infermap.preflight import run_preflight
    from infermap.candidates import generate_candidates
    from infermap.benchmark import benchmark_candidate

    shape = [int(x) for x in input_shape.split(",")]

    with console.status("[bold green]Profiling hardware..."):
        hw = profile_hardware()
    with console.status("[bold green]Inspecting model..."):
        info = inspect_model(model_path, input_shape=shape)

    pf = run_preflight(info, hw)
    _print_preflight(pf)
    if pf.category == "impossible":
        raise typer.Exit(code=1)

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    candidates = generate_candidates(info, hw)
    console.print(f"\n[bold]Running {len(candidates)} candidates...[/bold]\n")

    results = []
    for cand in candidates:
        with console.status(f"  Benchmarking [cyan]{cand.description}[/cyan]..."):
            r = benchmark_candidate(cand, model, info, shape, batch_size, warmup, iters)
        results.append(r)
        if r.ok:
            console.print(
                f"  [green]✓[/green] {cand.description:45s} "
                f"p50={r.latency_p50_ms:7.2f} ms  "
                f"{r.throughput_rps:8.0f} req/s  "
                f"{r.memory_mb:7.1f} MB"
            )
        else:
            console.print(f"  [red]✗[/red] {cand.description:45s} ERROR: {r.error}")

    _print_results_table(results)


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------


@app.command()
def optimize(
    model_path: Path = typer.Argument(..., help="Path to a saved PyTorch model"),
    objective: str = typer.Option("latency", help="Optimization goal: latency, throughput, memory"),
    input_shape: str = typer.Option("3,224,224", "--input-shape", help="Input shape (no batch)"),
    batch_size: int = typer.Option(1, help="Batch size"),
    max_latency_ms: Optional[float] = typer.Option(None, "--max-latency-ms"),
    max_memory_mb: Optional[float] = typer.Option(None, "--max-memory-mb"),
    min_throughput_rps: Optional[float] = typer.Option(None, "--min-throughput-rps"),
) -> None:
    """Benchmark all candidates and recommend the optimal deployment strategy."""
    import torch
    from infermap.inspector import inspect_model
    from infermap.profiler import profile_hardware
    from infermap.preflight import run_preflight
    from infermap.candidates import generate_candidates
    from infermap.benchmark import benchmark_candidate
    from infermap.recommender import recommend

    shape = [int(x) for x in input_shape.split(",")]

    with console.status("[bold green]Profiling hardware..."):
        hw = profile_hardware()
    with console.status("[bold green]Inspecting model..."):
        info = inspect_model(model_path, input_shape=shape)

    pf = run_preflight(info, hw)
    _print_preflight(pf)
    if pf.category == "impossible":
        raise typer.Exit(code=1)

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    candidates = generate_candidates(info, hw)
    console.print(f"\n[bold]Benchmarking {len(candidates)} candidates...[/bold]\n")

    results = []
    for cand in candidates:
        with console.status(f"  [cyan]{cand.description}[/cyan]..."):
            r = benchmark_candidate(cand, model, info, shape, batch_size)
        results.append(r)

    rec = recommend(
        results,
        objective=objective,
        max_latency_ms=max_latency_ms,
        max_memory_mb=max_memory_mb,
        min_throughput_rps=min_throughput_rps,
    )

    _print_recommendation(rec)


# ---------------------------------------------------------------------------
# Rich output helpers
# ---------------------------------------------------------------------------


def _print_hardware(hw: object) -> None:
    from infermap.profiler import HardwareProfile

    assert isinstance(hw, HardwareProfile)
    t = Table(title="Hardware Profile", box=box.ROUNDED, show_header=False)
    t.add_column("Key", style="bold cyan")
    t.add_column("Value")
    t.add_row("CPU", f"{hw.cpu.name} ({hw.cpu.physical_cores}C/{hw.cpu.logical_cores}T)")
    t.add_row("RAM", f"{hw.cpu.ram_gb:.1f} GB")
    t.add_row("Accelerator", hw.accelerator.kind.upper())
    if hw.accelerator.kind != "none":
        t.add_row("Device", hw.accelerator.name)
        t.add_row("Memory", f"{hw.accelerator.memory_gb:.1f} GB")
        t.add_row("BF16", "Yes" if hw.accelerator.bf16 else "No")
    console.print(t)


def _print_model_info(info: object) -> None:
    from infermap.inspector import ModelInfo

    assert isinstance(info, ModelInfo)
    t = Table(title="Model Profile", box=box.ROUNDED, show_header=False)
    t.add_column("Key", style="bold cyan")
    t.add_column("Value")
    t.add_row("Framework", info.framework.capitalize())
    t.add_row("Family", info.family.capitalize())
    t.add_row("Parameters", f"{info.parameters:,}")
    t.add_row("Memory (FP32)", f"{info.estimated_memory_fp32_gb:.2f} GB")
    t.add_row("Memory (FP16)", f"{info.estimated_memory_fp16_gb:.2f} GB")
    console.print(t)


def _print_preflight(result: object) -> None:
    from infermap.preflight import PreflightResult

    assert isinstance(result, PreflightResult)

    icon_map = {"ok": "✓", "tight": "⚠", "unlikely": "⚠", "impossible": "✗"}
    style_map = {"ok": "green", "tight": "yellow", "unlikely": "yellow", "impossible": "red"}
    icon = icon_map[result.category]
    style = style_map[result.category]

    title = f"[{style}]{icon} Pre-flight: {result.category.upper()}[/{style}]"
    body = result.message

    if result.suggestions:
        body += "\n\n[bold]Suggestions:[/bold]"
        for s in result.suggestions:
            body += f"\n  → {s}"

    body += (
        f"\n\n[dim]Estimated model memory: {result.estimated_memory_gb:.2f} GB  |  "
        f"Available: {result.available_memory_gb:.2f} GB[/dim]"
    )
    if result.estimated_throughput_range:
        lo, hi = result.estimated_throughput_range
        body += f"\n[dim]Estimated throughput range: {lo:.0f}–{hi:.0f} req/s[/dim]"

    console.print(Panel(body, title=title, border_style=style))


def _print_results_table(results: list) -> None:
    from infermap.benchmark import BenchmarkResult

    t = Table(title="Benchmark Results", box=box.ROUNDED)
    t.add_column("Backend", style="cyan")
    t.add_column("p50 (ms)", justify="right")
    t.add_column("p95 (ms)", justify="right")
    t.add_column("p99 (ms)", justify="right")
    t.add_column("Throughput", justify="right")
    t.add_column("Memory (MB)", justify="right")
    t.add_column("Status")

    for r in results:
        assert isinstance(r, BenchmarkResult)
        if r.ok:
            t.add_row(
                r.candidate.description,
                f"{r.latency_p50_ms:.2f}",
                f"{r.latency_p95_ms:.2f}",
                f"{r.latency_p99_ms:.2f}",
                f"{r.throughput_rps:.0f} req/s",
                f"{r.memory_mb:.1f}",
                "[green]OK[/green]",
            )
        else:
            t.add_row(
                r.candidate.description, "—", "—", "—", "—", "—",
                f"[red]ERR: {r.error[:40]}[/red]",
            )

    console.print(t)


def _print_recommendation(rec: object) -> None:
    from infermap.recommender import Recommendation

    assert isinstance(rec, Recommendation)
    r = rec.result

    body = (
        f"[bold cyan]{r.candidate.description}[/bold cyan]\n\n"
        f"  p50 latency :  {r.latency_p50_ms:.2f} ms\n"
        f"  p95 latency :  {r.latency_p95_ms:.2f} ms\n"
        f"  Throughput  :  {r.throughput_rps:.0f} req/s\n"
        f"  Memory      :  {r.memory_mb:.1f} MB\n\n"
        f"[dim]{rec.rationale}[/dim]"
    )

    if len(rec.pareto_frontier) > 1:
        body += f"\n\n[dim]Pareto frontier contains {len(rec.pareto_frontier)} non-dominated options.[/dim]"

    console.print(Panel(body, title="[bold green]Recommended Strategy[/bold green]", border_style="green"))


if __name__ == "__main__":
    app()
