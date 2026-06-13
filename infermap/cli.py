"""CLI entry point — Typer app with Rich output."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Must be set before libomp is loaded (i.e. before any torch import).
# Prevents crash when torch and onnxruntime each ship their own libomp.dylib on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table

app = typer.Typer(
    name="aphex",
    help="Hardware-aware ML deployment optimization and recommendation.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True, style="bold red")

_RANK_STYLES = ["bold green", "green", "yellow", "dim yellow", "dim", "dim", "dim"]


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

    _print_header("analyze")

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

    _print_header("preflight")

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
    timeout: float = typer.Option(180.0, "--timeout", help="Per-candidate timeout in seconds (0 = no limit)"),
    calibration_data: Optional[Path] = typer.Option(
        None, "--calibration-data",
        help="Path to a .pt file with calibration inputs for INT8 accuracy measurement",
    ),
) -> None:
    """Benchmark all candidate deployment strategies."""
    from infermap.inspector import inspect_model
    from infermap.profiler import profile_hardware
    from infermap.preflight import run_preflight
    from infermap.candidates import generate_candidates

    shape = [int(x) for x in input_shape.split(",")]
    timeout_s = timeout if timeout > 0 else None

    _print_header("benchmark")

    with console.status("[bold green]Profiling hardware..."):
        hw = profile_hardware()
    with console.status("[bold green]Inspecting model..."):
        info = inspect_model(model_path, input_shape=shape)

    pf = run_preflight(info, hw)
    _print_preflight(pf)
    if pf.category == "impossible":
        raise typer.Exit(code=1)
    if pf.category == "unlikely":
        _prompt_unlikely_or_abort()

    from infermap.inspector import _load_model
    model = _load_model(model_path)
    model.eval()

    calib = _load_calibration(calibration_data)
    candidates = generate_candidates(info, hw)
    results = _run_candidates(candidates, model, info, shape, batch_size, warmup, iters, timeout_s, calib)
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
    timeout: float = typer.Option(180.0, "--timeout", help="Per-candidate timeout in seconds (0 = no limit)"),
    calibration_data: Optional[Path] = typer.Option(
        None, "--calibration-data",
        help="Path to a .pt file with calibration inputs for INT8 accuracy measurement",
    ),
) -> None:
    """Benchmark all candidates and recommend the optimal deployment strategy."""
    from infermap.inspector import inspect_model
    from infermap.profiler import profile_hardware
    from infermap.preflight import run_preflight
    from infermap.candidates import generate_candidates
    from infermap.recommender import recommend

    shape = [int(x) for x in input_shape.split(",")]
    timeout_s = timeout if timeout > 0 else None

    _print_header("optimize")

    with console.status("[bold green]Profiling hardware..."):
        hw = profile_hardware()
    with console.status("[bold green]Inspecting model..."):
        info = inspect_model(model_path, input_shape=shape)

    pf = run_preflight(info, hw)
    _print_preflight(pf)
    if pf.category == "impossible":
        raise typer.Exit(code=1)
    if pf.category == "unlikely":
        _prompt_unlikely_or_abort()

    from infermap.inspector import _load_model
    model = _load_model(model_path)
    model.eval()

    calib = _load_calibration(calibration_data)
    candidates = generate_candidates(info, hw)
    results = _run_candidates(candidates, model, info, shape, batch_size, 10, 100, timeout_s, calib)
    _print_results_table(results)

    rec = recommend(
        results,
        objective=objective,
        max_latency_ms=max_latency_ms,
        max_memory_mb=max_memory_mb,
        min_throughput_rps=min_throughput_rps,
    )
    _print_recommendation(rec)


# ---------------------------------------------------------------------------
# Shared benchmark runner
# ---------------------------------------------------------------------------


def _load_calibration(path: Optional[Path]) -> list | None:
    if path is None:
        return None
    import torch
    raw = torch.load(path, weights_only=False)
    if isinstance(raw, torch.Tensor):
        return [raw[i : i + 1] for i in range(min(raw.size(0), 32))]
    if isinstance(raw, list):
        return raw[:32]
    return None


def _run_candidates(
    candidates: list,
    model: object,
    model_info: object,
    shape: list[int],
    batch_size: int,
    warmup: int,
    iters: int,
    timeout_s: float | None,
    calibration_inputs: list | None = None,
) -> list:
    from infermap.benchmark import benchmark_candidate

    results = []
    console.print()

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[cyan]{task.description}"),
        TextColumn("[dim]·[/dim]"),
        MofNCompleteColumn(),
        TextColumn("[dim]·[/dim]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"racing {len(candidates)} configurations",
            total=len(candidates),
        )
        for cand in candidates:
            progress.update(task, description=cand.description)
            r = benchmark_candidate(  # type: ignore[arg-type]
                cand, model, model_info, shape, batch_size, warmup, iters,
                timeout_s=timeout_s, calibration_inputs=calibration_inputs,
            )
            results.append(r)
            if r.ok:
                progress.console.print(
                    f"  [green]✓[/green]  {cand.description:<44}"
                    f"  [bold]{r.latency_p50_ms:>8.2f} ms[/bold]"
                    f"  [dim]{r.throughput_rps:>7.0f} req/s[/dim]"
                )
            else:
                progress.console.print(
                    f"  [red]✗[/red]  [dim]{cand.description:<44}[/dim]"
                    f"  [red]{(r.error or '')[:55]}[/red]"
                )
            progress.advance(task)

    console.print()
    return results


# ---------------------------------------------------------------------------
# Rich output helpers
# ---------------------------------------------------------------------------


def _print_header(label: str) -> None:
    console.print()
    console.print(Rule(f"[bold]aphex[/bold] [dim]·[/dim] [dim]{label}[/dim]", style="dim", align="left"))
    console.print()


def _prompt_unlikely_or_abort() -> None:
    console.print()
    console.print("  [yellow][1][/yellow] Benchmark anyway")
    console.print("  [dim][2] Abort[/dim]")
    choice = typer.prompt("\nChoice", default="1")
    if choice.strip() != "1":
        raise typer.Exit(code=1)
    console.print()


def _print_hardware(hw: object) -> None:
    from infermap.profiler import HardwareProfile

    assert isinstance(hw, HardwareProfile)
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("Key", style="dim", min_width=14)
    t.add_column("Value", style="bold")
    t.add_row("cpu", f"{hw.cpu.name} ({hw.cpu.physical_cores}C / {hw.cpu.logical_cores}T)")
    t.add_row("ram", f"{hw.cpu.ram_gb:.1f} GB")
    t.add_row("accelerator", hw.accelerator.kind.upper() if hw.accelerator.kind != "none" else "none")
    if hw.accelerator.kind != "none":
        t.add_row("device", hw.accelerator.name)
        t.add_row("vram", f"{hw.accelerator.memory_gb:.1f} GB")
        t.add_row("bf16", "yes" if hw.accelerator.bf16 else "no")
    console.print(Rule("[dim]hardware[/dim]", style="dim"))
    console.print(t)


def _print_model_info(info: object) -> None:
    from infermap.inspector import ModelInfo

    assert isinstance(info, ModelInfo)
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("Key", style="dim", min_width=14)
    t.add_column("Value", style="bold")
    t.add_row("framework", info.framework.capitalize())
    t.add_row("family", info.family.capitalize())
    t.add_row("parameters", f"{info.parameters:,}")
    t.add_row("memory fp32", f"{info.estimated_memory_fp32_gb:.2f} GB")
    t.add_row("memory fp16", f"{info.estimated_memory_fp16_gb:.2f} GB")
    console.print(Rule("[dim]model[/dim]", style="dim"))
    console.print(t)


def _print_preflight(result: object) -> None:
    from infermap.preflight import PreflightResult

    assert isinstance(result, PreflightResult)

    cfg: dict[str, tuple[str, str]] = {
        "ok":         ("✓", "green"),
        "tight":      ("⚠", "yellow"),
        "unlikely":   ("⚠", "yellow"),
        "impossible": ("✗", "red"),
    }
    icon, style = cfg[result.category]

    title = f"[{style}]{icon}  preflight: {result.category}[/{style}]"
    body = result.message

    if result.suggestions:
        body += "\n"
        for s in result.suggestions:
            body += f"\n  [dim]→[/dim] {s}"

    body += (
        f"\n\n[dim]model {result.estimated_memory_gb:.2f} GB  ·  "
        f"available {result.available_memory_gb:.2f} GB[/dim]"
    )
    if result.estimated_throughput_range:
        lo, hi = result.estimated_throughput_range
        body += f"  [dim]·  est. {lo:.0f}–{hi:.0f} req/s[/dim]"

    console.print(Panel(body, title=title, border_style=style, padding=(1, 2)))


def _speed_bar(value: float, max_value: float, width: int = 18) -> str:
    if max_value == 0:
        return "░" * width
    filled = round((value / max_value) * width)
    return "█" * filled + "░" * (width - filled)


def _print_results_table(results: list) -> None:
    from infermap.benchmark import BenchmarkResult

    ok = sorted(
        [r for r in results if r.ok],
        key=lambda r: r.latency_p50_ms,
    )
    failed = [r for r in results if not r.ok]

    if not ok:
        console.print("[red]All candidates failed.[/red]")
        return

    max_tput = max(r.throughput_rps for r in ok)
    has_accuracy = any(r.accuracy_drop is not None for r in results)

    t = Table(box=box.SIMPLE, show_header=True, header_style="dim", padding=(0, 1))
    t.add_column("", width=3, justify="right")           # rank
    t.add_column("backend", min_width=34)
    t.add_column("p50", justify="right")
    t.add_column("p95", justify="right")
    t.add_column("req/s", justify="right")
    t.add_column("mb", justify="right")
    if has_accuracy:
        t.add_column("acc drop", justify="right")
    t.add_column("speed", min_width=20)

    for i, r in enumerate(ok):
        assert isinstance(r, BenchmarkResult)
        s = _RANK_STYLES[min(i, len(_RANK_STYLES) - 1)]
        bar = _speed_bar(r.throughput_rps, max_tput)
        row: list[str] = [
            f"[{s}]#{i + 1}[/{s}]",
            f"[{s}]{r.candidate.description}[/{s}]",
            f"[{s}]{r.latency_p50_ms:.2f} ms[/{s}]",
            f"[dim]{r.latency_p95_ms:.2f} ms[/dim]",
            f"[dim]{r.throughput_rps:.0f}[/dim]",
            f"[dim]{r.memory_mb:.0f}[/dim]",
        ]
        if has_accuracy:
            acc = f"{r.accuracy_drop * 100:.2f}%" if r.accuracy_drop is not None else "—"
            row.append(f"[dim]{acc}[/dim]")
        row.append(f"[{s}]{bar}[/{s}]")
        t.add_row(*row)

    for r in failed:
        assert isinstance(r, BenchmarkResult)
        row = [
            "[dim]—[/dim]",
            f"[dim]{r.candidate.description}[/dim]",
            "[dim]—[/dim]", "[dim]—[/dim]", "[dim]—[/dim]", "[dim]—[/dim]",
        ]
        if has_accuracy:
            row.append("[dim]—[/dim]")
        row.append(f"[red]{(r.error or '')[:28]}[/red]")
        t.add_row(*row)

    console.print(Rule("[dim]results[/dim]", style="dim"))
    console.print(t)


def _print_recommendation(rec: object) -> None:
    from infermap.recommender import Recommendation

    assert isinstance(rec, Recommendation)
    r = rec.result

    acc_line = (
        f"  [dim]acc drop[/dim]  {r.accuracy_drop * 100:.2f}%\n"
        if r.accuracy_drop is not None
        else ""
    )
    stats = (
        f"  [dim]p50[/dim]    [bold]{r.latency_p50_ms:.2f} ms[/bold]\n"
        f"  [dim]p95[/dim]    {r.latency_p95_ms:.2f} ms\n"
        f"  [dim]req/s[/dim]  [bold]{r.throughput_rps:.0f}[/bold]\n"
        f"  [dim]mb[/dim]     {r.memory_mb:.0f}\n"
        + acc_line
    )

    frontier_note = (
        f"\n[dim]{len(rec.pareto_frontier)} options on the Pareto frontier.[/dim]"
        if len(rec.pareto_frontier) > 1
        else ""
    )

    body = (
        f"[bold white]{r.candidate.description}[/bold white]\n\n"
        f"{stats}\n"
        f"[dim]{rec.rationale}[/dim]"
        f"{frontier_note}"
    )

    console.print()
    console.print(Rule("[bold green]recommendation[/bold green]", style="green"))
    console.print(Panel(body, border_style="green", padding=(1, 2)))
    console.print()


if __name__ == "__main__":
    app()
