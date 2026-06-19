"""CLI entry point — Typer app with Rich output."""

from __future__ import annotations
import json
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
    model_path: Path = typer.Argument(..., help="Path to a saved model (.pt / .pkl / ...)"),
) -> None:
    """Inspect a model and profile the current hardware."""
    from infermap.profiler import profile_hardware
    from infermap.registry import get_plugin

    _print_header("analyze")

    with console.status("[bold green]Profiling hardware..."):
        hw = profile_hardware()

    with console.status("[bold green]Inspecting model..."):
        plugin = get_plugin(model_path)
        info = plugin.inspect(model_path)

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
    from infermap.profiler import profile_hardware
    from infermap.preflight import run_preflight
    from infermap.registry import get_plugin

    _print_header("preflight")

    with console.status("[bold green]Profiling hardware..."):
        hw = profile_hardware()

    with console.status("[bold green]Inspecting model..."):
        plugin = get_plugin(model_path)
        info = plugin.inspect(model_path)

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
    batch_sizes: str = typer.Option("1,2,4,8", "--batch-sizes", help="Comma-separated batch sizes to sweep, e.g. 1,2,4,8"),
    warmup: int = typer.Option(10, help="Warm-up iterations"),
    iters: int = typer.Option(100, help="Measurement iterations"),
    timeout: float = typer.Option(180.0, "--timeout", help="Per-candidate timeout in seconds (0 = no limit)"),
    calibration_data: Optional[Path] = typer.Option(
        None, "--calibration-data",
        help="Path to a .pt file (or image directory) with unlabelled calibration inputs for int8 quantization.",
    ),
    eval_data: Optional[Path] = typer.Option(
        None, "--eval",
        help="Path to labelled eval data (.pt, .parquet, .csv, or image dir) for real accuracy measurement.",
    ),
    eval_label_col: str = typer.Option(
        "label", "--eval-label-col",
        help="Column name for labels when --eval points to a .parquet or .csv file.",
    ),
    infer_fn: Optional[str] = typer.Option(
        None, "--infer-fn",
        help="Inference callable as 'path/to/module.py:function'. Required with --eval. "
             "Function signature: predict(inputs: list) -> np.ndarray.",
    ),
    max_quality_loss: Optional[float] = typer.Option(
        None, "--max-quality-loss",
        help="Maximum acceptable quality loss vs FP32 baseline (0.0–1.0). Requires --calibration-data or --eval.",
    ),
    format_: str = typer.Option("table", "--format", help="Output format: table | json"),
) -> None:
    """Benchmark all candidate deployment strategies."""
    from infermap.profiler import profile_hardware
    from infermap.preflight import run_preflight
    from infermap.registry import get_plugin

    shape = [int(x) for x in input_shape.split(",")]
    bs_list = [int(x) for x in batch_sizes.split(",")]
    timeout_s = timeout if timeout > 0 else None
    json_mode = format_ == "json"

    if not json_mode:
        _print_header("benchmark")

    if max_quality_loss is not None and calibration_data is None and eval_data is None:
        err_console.print(
            "[yellow]Warning: --max-quality-loss has no effect without --calibration-data or --eval.[/yellow]"
        )

    with console.status("[bold green]Profiling hardware..."):
        hw = profile_hardware()
    with console.status("[bold green]Inspecting model..."):
        plugin = get_plugin(model_path)
        info = plugin.inspect(model_path, input_shape=shape)

    pf = run_preflight(info, hw)
    if not json_mode:
        _print_preflight(pf)
    if pf.category == "impossible":
        if json_mode:
            print(json.dumps({"error": pf.message, "category": "impossible"}, indent=2))
        raise typer.Exit(code=1)
    if pf.category == "unlikely" and not json_mode:
        _prompt_unlikely_or_abort()  # no constraints to relax in bare benchmark

    if eval_data and not infer_fn:
        err_console.print("[yellow]Warning: --eval has no effect without --infer-fn.[/yellow]")

    model = plugin.load(model_path)
    eval_dataset = _load_eval(eval_data, info, eval_label_col)
    fn = _load_infer_fn(infer_fn)
    calib = _load_calibration(calibration_data, shape) or (eval_dataset.inputs if eval_dataset else None)
    candidates = plugin.generate_candidates(info, hw)
    candidates, cost_estimates = _prune_with_cost_model(candidates, info, hw, bs_list[0], json_mode)
    results = _run_candidates(plugin, candidates, model, info, shape, bs_list, warmup, iters, timeout_s, calib, json_mode=json_mode)
    _run_eval(eval_dataset, fn, json_mode, results=results, model=model, info=info)
    if json_mode:
        _emit_json(results)
    else:
        _print_results_table(results)


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------


@app.command()
def optimize(
    model_path: Path = typer.Argument(..., help="Path to a saved PyTorch model"),
    objective: str = typer.Option("latency", help="Optimization goal: latency, throughput, memory"),
    input_shape: str = typer.Option("3,224,224", "--input-shape", help="Input shape (no batch)"),
    batch_sizes: str = typer.Option("1,2,4,8", "--batch-sizes", help="Comma-separated batch sizes to sweep, e.g. 1,2,4,8"),
    target: Optional[str] = typer.Option(
        None, "--target",
        help="Target cloud instance type or GPU alias, e.g. ml.g4dn.xlarge, a2-highgpu-1g, a100. "
             "Skips local hardware profiling and optimises for the specified hardware. "
             "Run 'aphex targets' to see all options.",
    ),
    remote: Optional[str] = typer.Option(
        None, "--remote",
        help="Run benchmarks on a remote machine via SSH, e.g. user@gpu-host. "
             "Requires aphex installed on the remote machine.",
    ),
    max_latency_ms: Optional[float] = typer.Option(None, "--max-latency-ms"),
    max_memory_mb: Optional[float] = typer.Option(None, "--max-memory-mb"),
    min_throughput_rps: Optional[float] = typer.Option(None, "--min-throughput-rps"),
    timeout: float = typer.Option(180.0, "--timeout", help="Per-candidate timeout in seconds (0 = no limit)"),
    calibration_data: Optional[Path] = typer.Option(
        None, "--calibration-data",
        help="Path to a .pt file (or image directory) with unlabelled calibration inputs for int8 quantization.",
    ),
    eval_data: Optional[Path] = typer.Option(
        None, "--eval",
        help="Path to labelled eval data (.pt, .parquet, .csv, or image dir) for real accuracy measurement.",
    ),
    eval_label_col: str = typer.Option(
        "label", "--eval-label-col",
        help="Column name for labels when --eval points to a .parquet or .csv file.",
    ),
    infer_fn: Optional[str] = typer.Option(
        None, "--infer-fn",
        help="Inference callable as 'path/to/module.py:function'. Required with --eval. "
             "Function signature: predict(inputs: list) -> np.ndarray.",
    ),
    max_quality_loss: Optional[float] = typer.Option(
        None, "--max-quality-loss",
        help="Maximum acceptable quality loss vs FP32 baseline (0.0–1.0). Requires --calibration-data or --eval.",
    ),
    output: Optional[Path] = typer.Option(
        Path("deployment.yaml"), "--output",
        help="Write deployment config to this path (default: deployment.yaml). Pass '' to skip.",
    ),
    serving: Optional[str] = typer.Option(
        None, "--serving",
        help="Generate serving configs: triton, torchserve, bentoml, fastapi",
    ),
    format_: str = typer.Option("table", "--format", help="Output format: table | json"),
    report: Optional[Path] = typer.Option(None, "--report", help="Write HTML benchmark report to this path"),
    metrics: Optional[Path] = typer.Option(None, "--metrics", help="Write benchmark metrics JSON to this path"),
) -> None:
    """Benchmark all candidates and recommend the optimal deployment strategy."""
    from infermap.profiler import profile_hardware
    from infermap.preflight import run_preflight
    from infermap.recommender import recommend
    from infermap.registry import get_plugin

    shape = [int(x) for x in input_shape.split(",")]
    bs_list = [int(x) for x in batch_sizes.split(",")]
    timeout_s = timeout if timeout > 0 else None
    json_mode = format_ == "json"

    # ── remote path ───────────────────────────────────────────────────────────
    if remote is not None:
        _remote_optimize(
            host=remote,
            model_path=model_path,
            input_shape=input_shape,
            objective=objective,
            batch_sizes=batch_sizes,
            max_latency_ms=max_latency_ms,
            max_memory_mb=max_memory_mb,
            min_throughput_rps=min_throughput_rps,
            max_quality_loss=max_quality_loss,
            timeout=timeout,
            output=output,
            serving=serving,
            format_=format_,
            target=target,
            calibration_data=calibration_data,
            eval_data=eval_data,
            infer_fn=infer_fn,
            report=report,
            metrics=metrics,
        )
        return

    if not json_mode:
        _print_header("optimize")

    if max_quality_loss is not None and calibration_data is None and eval_data is None:
        err_console.print(
            "[yellow]Warning: --max-quality-loss has no effect without --calibration-data or --eval.[/yellow]"
        )

    if target is not None:
        from infermap.cloud.instances import get_instance_profile
        try:
            hw = get_instance_profile(target)
        except ValueError as exc:
            err_console.print(str(exc))
            raise typer.Exit(code=1)
        console.print(
            f"  [dim]target[/dim]  [bold]{target}[/bold]  "
            f"[dim]({hw.accelerator.name}, {hw.accelerator.memory_gb:.0f} GB VRAM)[/dim]\n"
        )
    else:
        with console.status("[bold green]Profiling hardware..."):
            hw = profile_hardware()

    with console.status("[bold green]Inspecting model..."):
        plugin = get_plugin(model_path)
        info = plugin.inspect(model_path, input_shape=shape)

    pf = run_preflight(info, hw, target_throughput=min_throughput_rps)
    if not json_mode:
        _print_preflight(pf)
    if pf.category == "impossible":
        if json_mode:
            print(json.dumps({"error": pf.message, "category": "impossible"}, indent=2))
        raise typer.Exit(code=1)
    if pf.category == "unlikely" and not json_mode:
        if _prompt_unlikely_or_abort():
            min_throughput_rps = None

    if eval_data and not infer_fn:
        err_console.print("[yellow]Warning: --eval has no effect without --infer-fn.[/yellow]")

    model = plugin.load(model_path)
    eval_dataset = _load_eval(eval_data, info, eval_label_col)
    fn = _load_infer_fn(infer_fn)
    calib = _load_calibration(calibration_data, shape) or (eval_dataset.inputs if eval_dataset else None)
    candidates = plugin.generate_candidates(info, hw)
    candidates = _select_with_fingerprint(candidates, info, hw, json_mode)
    results = _run_candidates(plugin, candidates, model, info, shape, bs_list, 10, 100, timeout_s, calib, json_mode=json_mode)
    eval_result = _run_eval(eval_dataset, fn, json_mode, results=results, model=model, info=info)

    rec = recommend(
        results,
        objective=objective,
        max_latency_ms=max_latency_ms,
        max_memory_mb=max_memory_mb,
        min_throughput_rps=min_throughput_rps,
        max_quality_loss=max_quality_loss,
    )

    write_output = bool(output and str(output) != "")
    serving_dir: Path | None = None

    if write_output or serving:
        from infermap.deployment import build_config, write_yaml
        from infermap.system_recommender import recommend_system_config
        sys_cfg = recommend_system_config(rec, info, hw, objective=objective)
        cfg = build_config(
            rec, info, hw,
            objective=objective,
            max_latency_ms=max_latency_ms,
            max_memory_mb=max_memory_mb,
            min_throughput_rps=min_throughput_rps,
            max_quality_loss=max_quality_loss,
            system=sys_cfg,
            eval_metric=eval_result[0] if eval_result else None,
            eval_score=eval_result[1] if eval_result else None,
            input_shape=shape,
        )
        if write_output:
            write_yaml(cfg, output)  # type: ignore[arg-type]

        if serving:
            from infermap.serving import generate_serving_config, SUPPORTED_FRAMEWORKS
            if serving not in SUPPORTED_FRAMEWORKS:
                err_console.print(
                    f"Unknown serving framework {serving!r}. "
                    f"Supported: {', '.join(SUPPORTED_FRAMEWORKS)}"
                )
                raise typer.Exit(code=1)
            base = output.parent if (output and str(output) != "") else Path(".")
            serving_dir = base / "serving" / serving
            generate_serving_config(serving, cfg, serving_dir)

    if json_mode:
        _emit_json(results, rec)
    else:
        _print_results_table(results)
        _print_recommendation(rec, eval_result)
        if write_output:
            console.print(f"  [dim]deployment config →[/dim] [bold]{output}[/bold]")
        if serving_dir is not None:
            console.print(f"  [dim]serving config ({serving}) →[/dim] [bold]{serving_dir}[/bold]")
        if write_output or serving_dir is not None:
            console.print()

    if metrics or report:
        payload = _build_metrics_payload(results, rec)
        if metrics:
            metrics.write_text(json.dumps(payload, indent=2))
            if not json_mode:
                console.print(f"  [dim]metrics →[/dim] [bold]{metrics}[/bold]")
        if report:
            from infermap.report import render_report
            render_report(payload, report)
            if not json_mode:
                console.print(f"  [dim]report →[/dim] [bold]{report}[/bold]")
    if (metrics or report) and not json_mode:
        console.print()


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


@app.command()
def convert(
    model_path: Path = typer.Argument(..., help="Path to a saved model (.pt / .pkl / ...)"),
    backend: Optional[str] = typer.Option(
        None, "--backend",
        help="Target backend, e.g. onnx_int8_cpu, pytorch_fp16, tensorrt_fp16.",
    ),
    input_shape: Optional[str] = typer.Option(
        None, "--input-shape",
        help="Input tensor shape (no batch dim), e.g. 3,224,224. Required unless --from-config is used.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output",
        help="Output artifact path. Auto-derived from model name and backend if omitted.",
    ),
    from_config: Optional[Path] = typer.Option(
        None, "--from-config",
        help="Read backend and input-shape from a deployment.yaml written by aphex optimize.",
    ),
    calibration_data: Optional[Path] = typer.Option(
        None, "--calibration-data",
        help="Calibration inputs (.pt file or image dir) required for int8 backends.",
    ),
) -> None:
    """Convert a model to its recommended deployment format.

    \b
    Examples
    --------
    # Use the recommendation from a prior 'aphex optimize' run:
      aphex convert model.pt --from-config deployment.yaml

    # Explicit backend + shape:
      aphex convert model.pt --backend onnx_int8_cpu --input-shape 3,224,224

    # TensorRT with calibration data:
      aphex convert model.pt --backend tensorrt_int8 --input-shape 3,224,224 \\
            --calibration-data calib.pt --output model_int8.engine
    """
    from infermap.converter import ALL_CONVERTIBLE, convert as _convert, default_output_path, read_deployment_yaml
    from infermap.registry import get_plugin

    _print_header("convert")

    # ── resolve backend + shape from --from-config ────────────────────────────
    resolved_backend = backend
    resolved_shape_str = input_shape

    if from_config is not None:
        if not from_config.exists():
            err_console.print(f"Config file not found: {from_config}")
            raise typer.Exit(code=1)
        try:
            cfg_data = read_deployment_yaml(from_config)
        except Exception as exc:
            err_console.print(f"Could not parse {from_config}: {exc}")
            raise typer.Exit(code=1)

        if resolved_backend is None:
            resolved_backend = (cfg_data.get("recommendation") or {}).get("backend")
        if resolved_shape_str is None:
            resolved_shape_str = (cfg_data.get("model") or {}).get("input_shape")

    # ── validate ──────────────────────────────────────────────────────────────
    if resolved_backend is None:
        err_console.print(
            "No backend specified. Use --backend or --from-config deployment.yaml."
        )
        raise typer.Exit(code=1)

    if resolved_backend not in ALL_CONVERTIBLE:
        err_console.print(
            f"Backend {resolved_backend!r} cannot be converted to a file artifact. "
            f"Convertible backends: {', '.join(sorted(ALL_CONVERTIBLE))}"
        )
        raise typer.Exit(code=1)

    if resolved_shape_str is None:
        err_console.print(
            "No input shape specified. Use --input-shape 3,224,224 or --from-config."
        )
        raise typer.Exit(code=1)

    try:
        shape = [int(x) for x in resolved_shape_str.split(",")]
    except ValueError:
        err_console.print(f"Invalid input shape: {resolved_shape_str!r}. Expected comma-separated ints.")
        raise typer.Exit(code=1)

    out_path = output or default_output_path(model_path, resolved_backend)

    # ── load model + calibration ──────────────────────────────────────────────
    with console.status("[bold green]Loading model..."):
        plugin = get_plugin(model_path)
        model = plugin.load(model_path)

    calib = _load_calibration(calibration_data, shape) if calibration_data else None

    # ── convert ───────────────────────────────────────────────────────────────
    console.print(
        f"  [dim]converting[/dim] [bold]{model_path.name}[/bold] "
        f"[dim]→[/dim] [bold cyan]{resolved_backend}[/bold cyan]"
    )
    try:
        with console.status(f"[bold green]Building {resolved_backend}..."):
            written = _convert(model, resolved_backend, shape, out_path, calib)
    except Exception as exc:
        err_console.print(f"Conversion failed: {exc}")
        raise typer.Exit(code=1)

    console.print()
    for p in written:
        size_mb = p.stat().st_size / 1e6
        console.print(f"  [green]✓[/green]  {p}  [dim]({size_mb:.1f} MB)[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


@app.command()
def check(
    model_path: Path = typer.Argument(..., help="Path to the saved model"),
    from_config: Path = typer.Option(..., "--from-config", help="deployment.yaml written by aphex optimize"),
    max_latency_delta: Optional[float] = typer.Option(
        10.0, "--max-latency-delta",
        help="Max allowed latency increase in percent (default 10). Pass 0 to disable.",
    ),
    min_throughput_delta: Optional[float] = typer.Option(
        None, "--min-throughput-delta",
        help="Max allowed throughput drop in percent, e.g. 10 means throughput may not drop >10%%.",
    ),
    max_memory_delta: Optional[float] = typer.Option(
        None, "--max-memory-delta",
        help="Max allowed memory increase in percent.",
    ),
    max_accuracy_drop: Optional[float] = typer.Option(
        None, "--max-accuracy-drop",
        help="Max allowed absolute accuracy drop vs baseline eval_score (requires --eval + --infer-fn).",
    ),
    warmup: int = typer.Option(5, help="Warm-up iterations (default lower than optimize for speed)"),
    iters: int = typer.Option(50, help="Measurement iterations"),
    eval_data: Optional[Path] = typer.Option(None, "--eval", help="Labelled eval data for accuracy re-check"),
    eval_label_col: str = typer.Option("label", "--eval-label-col"),
    infer_fn: Optional[str] = typer.Option(None, "--infer-fn", help="Inference callable 'module.py:fn'"),
) -> None:
    """Regression-check a model against a saved deployment.yaml baseline.

    \b
    Benchmarks the recommended backend from a prior 'aphex optimize' run and
    compares results against the saved performance numbers. Exits 1 if any
    configured threshold is exceeded — safe to use in CI pipelines.

    \b
    Examples
    --------
      aphex check model.pt --from-config deployment.yaml
      aphex check model.pt --from-config deployment.yaml --max-latency-delta 15
      aphex check model.pt --from-config deployment.yaml \\
            --eval val.pt --infer-fn infer.py:predict --max-accuracy-drop 0.02
    """
    from infermap.candidates import DeploymentCandidate
    from infermap.checker import run_check
    from infermap.converter import read_deployment_yaml
    from infermap.registry import get_plugin

    _print_header("check")

    if not from_config.exists():
        err_console.print(f"Config not found: {from_config}")
        raise typer.Exit(code=1)
    try:
        cfg = read_deployment_yaml(from_config)
    except Exception as exc:
        err_console.print(f"Could not parse {from_config}: {exc}")
        raise typer.Exit(code=1)

    rec = cfg.get("recommendation") or {}
    perf = cfg.get("performance") or {}
    model_cfg = cfg.get("model") or {}

    backend = rec.get("backend")
    if not backend:
        err_console.print("deployment.yaml has no recommendation.backend — run aphex optimize first.")
        raise typer.Exit(code=1)

    shape_str = model_cfg.get("input_shape")
    if not shape_str:
        err_console.print("deployment.yaml has no model.input_shape — re-run aphex optimize.")
        raise typer.Exit(code=1)
    try:
        shape = [int(x) for x in str(shape_str).split(",")]
    except ValueError:
        err_console.print(f"Invalid input_shape in config: {shape_str!r}")
        raise typer.Exit(code=1)

    batch_size = int(rec.get("batch_size") or 1)
    description = rec.get("description") or backend
    device = rec.get("device") or "cpu"
    dtype = rec.get("dtype") or "fp32"

    console.print(
        f"  [dim]checking[/dim] [bold]{description}[/bold] "
        f"[dim]against[/dim] [bold]{from_config}[/bold]"
    )
    console.print()

    with console.status("[bold green]Loading model..."):
        plugin = get_plugin(model_path)
        model = plugin.load(model_path)
        info = plugin.inspect(model_path, input_shape=shape)

    cand = DeploymentCandidate(
        backend=backend,
        dtype=dtype,
        description=description,
        requires_export=False,
        device=device,
    )

    with console.status(f"[bold green]Benchmarking {backend}..."):
        result = plugin.benchmark(cand, model, info, shape, batch_size, warmup, iters, None, None)

    if not result.ok:
        err_console.print(f"Benchmark failed: {result.error}")
        raise typer.Exit(code=1)

    # ── optional accuracy re-check ────────────────────────────────────────────
    observed_eval: float | None = None
    baseline_eval: float | None = perf.get("eval_score")

    if eval_data and infer_fn:
        eval_dataset = _load_eval(eval_data, info, eval_label_col)
        fn = _load_infer_fn(infer_fn)
        if eval_dataset and fn:
            from infermap.evaluator import evaluate_baseline
            try:
                with console.status("[bold green]Evaluating accuracy..."):
                    observed_eval = evaluate_baseline(fn, eval_dataset)
            except Exception as exc:
                err_console.print(f"[yellow]Warning: accuracy eval failed: {exc}[/yellow]")
    elif max_accuracy_drop is not None:
        err_console.print(
            "[yellow]Warning: --max-accuracy-drop requires --eval + --infer-fn. Skipping accuracy check.[/yellow]"
        )

    # ── compare ───────────────────────────────────────────────────────────────
    lat_threshold = max_latency_delta if (max_latency_delta and max_latency_delta > 0) else None
    check_results = run_check(
        perf,
        result.latency_p50_ms,
        result.latency_p95_ms,
        result.throughput_rps,
        result.memory_mb,
        max_latency_delta_pct=lat_threshold,
        min_throughput_delta_pct=min_throughput_delta,
        max_memory_delta_pct=max_memory_delta,
        baseline_eval_score=baseline_eval,
        observed_eval_score=observed_eval,
        max_accuracy_drop=max_accuracy_drop,
    )

    _print_check_results(check_results)

    failed = [r for r in check_results if not r.passed]
    if failed:
        console.print(f"  [red]✗[/red]  [bold red]{len(failed)} check(s) failed[/bold red]\n")
        raise typer.Exit(code=1)
    else:
        console.print("  [green]✓[/green]  [bold green]all checks passed[/bold green]\n")


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


@app.command()
def targets() -> None:
    """List all supported cloud instance types and GPU aliases for --target."""
    from infermap.cloud.instances import list_targets

    _print_header("targets")

    groups = list_targets()

    sections = [
        ("AWS EC2 / SageMaker", groups["aws"]),
        ("GCP Compute Engine / Vertex AI", groups["gcp"]),
        ("Azure", groups["azure"]),
        ("Short GPU aliases", groups["aliases"]),
    ]

    for title, items in sections:
        console.print(Rule(f"[dim]{title}[/dim]", style="dim"))
        for item in items:
            console.print(f"  [dim]·[/dim]  {item}")
        console.print()


# ---------------------------------------------------------------------------
# push / pull / ls
# ---------------------------------------------------------------------------


@app.command()
def push(
    files: list[Path] = typer.Argument(..., help="Files to upload: deployment.yaml and/or artifact"),
    name: str = typer.Option(..., "--name", help="Model name in the registry, e.g. resnet50"),
    version: Optional[str] = typer.Option(
        None, "--version",
        help="Version tag (default: auto-generated timestamp, e.g. v20260616-120000)",
    ),
    registry: Optional[str] = typer.Option(
        None, "--registry", help="Override registry URI (e.g. s3://bucket/aphex)"
    ),
) -> None:
    """Upload a model artifact (and deployment.yaml) to the cloud registry.

    \b
    Examples
    --------
      aphex push deployment.yaml model.onnx --name resnet50
      aphex push deployment.yaml model.engine --name resnet50 --version v2
    """
    import datetime

    from infermap.cloud.config import get_registry_uri
    from infermap.cloud.registry import push as _push
    from infermap.cloud.storage import get_backend

    _print_header("push")

    missing = [f for f in files if not f.exists()]
    if missing:
        for f in missing:
            err_console.print(f"File not found: {f}")
        raise typer.Exit(code=1)

    try:
        uri = registry or get_registry_uri()
        backend = get_backend(uri)
    except (RuntimeError, ValueError) as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1)

    ver = version or datetime.datetime.utcnow().strftime("v%Y%m%d-%H%M%S")

    console.print(f"  [dim]registry[/dim]  {uri}")
    console.print(f"  [dim]pushing[/dim]   [bold]{name}@{ver}[/bold]")
    console.print()

    try:
        with console.status(f"[bold green]Uploading {name}@{ver}..."):
            _push(name, ver, list(files), backend)
    except Exception as exc:
        err_console.print(f"Push failed: {exc}")
        raise typer.Exit(code=1)

    for f in files:
        size_mb = f.stat().st_size / 1e6
        console.print(
            f"  [green]✓[/green]  {name}/{ver}/{f.name}  [dim]({size_mb:.1f} MB)[/dim]"
        )
    console.print()
    console.print(f"  [dim]latest →[/dim] [bold]{ver}[/bold]")
    console.print()


@app.command()
def pull(
    name: str = typer.Argument(
        ..., help="Model name, optionally with version: resnet50 or resnet50@v1"
    ),
    out: Path = typer.Option(
        Path("."), "--out", help="Directory to write downloaded files (default: .)"
    ),
    registry: Optional[str] = typer.Option(
        None, "--registry", help="Override registry URI"
    ),
) -> None:
    """Download a model artifact from the cloud registry.

    \b
    Examples
    --------
      aphex pull resnet50                    # latest version
      aphex pull resnet50@v2                 # specific version
      aphex pull resnet50 --out ./models/
    """
    from infermap.cloud.config import get_registry_uri
    from infermap.cloud.registry import pull as _pull
    from infermap.cloud.storage import get_backend

    _print_header("pull")

    try:
        uri = registry or get_registry_uri()
        backend = get_backend(uri)
    except (RuntimeError, ValueError) as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1)

    console.print(f"  [dim]registry[/dim]  {uri}")
    console.print(f"  [dim]pulling[/dim]   [bold]{name}[/bold]")
    console.print()

    try:
        with console.status(f"[bold green]Downloading {name}..."):
            written = _pull(name, out, backend)
    except ValueError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1)
    except Exception as exc:
        err_console.print(f"Pull failed: {exc}")
        raise typer.Exit(code=1)

    for p in written:
        size_mb = p.stat().st_size / 1e6
        console.print(f"  [green]✓[/green]  {p}  [dim]({size_mb:.1f} MB)[/dim]")
    console.print()


@app.command()
def ls(
    name: Optional[str] = typer.Argument(
        None, help="Model name to list versions (omit to list all models)"
    ),
    registry: Optional[str] = typer.Option(
        None, "--registry", help="Override registry URI"
    ),
) -> None:
    """List models and versions in the cloud registry.

    \b
    Examples
    --------
      aphex ls                  # all models
      aphex ls resnet50         # versions of resnet50
    """
    from infermap.cloud.config import get_registry_uri
    from infermap.cloud.registry import list_models, list_versions
    from infermap.cloud.storage import get_backend

    _print_header("ls")

    try:
        uri = registry or get_registry_uri()
        backend = get_backend(uri)
    except (RuntimeError, ValueError) as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1)

    console.print(f"  [dim]registry[/dim]  {uri}\n")

    if name:
        versions = list_versions(name, backend)
        if not versions:
            console.print(f"  [dim]no versions found for {name!r}[/dim]")
        else:
            for v in versions:
                console.print(f"  [dim]·[/dim]  {name}[dim]@[/dim][bold]{v}[/bold]")
    else:
        models = list_models(backend)
        if not models:
            console.print("  [dim]no models in registry[/dim]")
        else:
            for m in models:
                versions = list_versions(m, backend)
                latest = versions[0] if versions else "—"
                console.print(
                    f"  [dim]·[/dim]  [bold]{m}[/bold]  "
                    f"[dim]({len(versions)} version(s), latest: {latest})[/dim]"
                )
    console.print()


# ---------------------------------------------------------------------------
# Remote optimize helper
# ---------------------------------------------------------------------------


def _remote_optimize(
    host: str,
    model_path: Path,
    input_shape: str,
    objective: str,
    batch_sizes: str,
    max_latency_ms: Optional[float],
    max_memory_mb: Optional[float],
    min_throughput_rps: Optional[float],
    max_quality_loss: Optional[float],
    timeout: float,
    output: Optional[Path],
    serving: Optional[str],
    format_: str,
    target: Optional[str],
    calibration_data: Optional[Path],
    eval_data: Optional[Path],
    infer_fn: Optional[str],
    report: Optional[Path] = None,
    metrics: Optional[Path] = None,
) -> None:
    from infermap.cloud.remote import check_remote_aphex, run_remote_optimize

    _print_header("optimize · remote")

    if target is not None:
        err_console.print(
            "[yellow]Warning: --target is ignored with --remote "
            "(the remote machine profiles its own hardware).[/yellow]"
        )

    for flag, val in [("--calibration-data", calibration_data),
                      ("--eval", eval_data), ("--infer-fn", infer_fn)]:
        if val is not None:
            err_console.print(
                f"[yellow]Warning: {flag} references a local path and is not "
                f"supported with --remote. Skipping.[/yellow]"
            )

    console.print(f"  [dim]remote[/dim]   [bold]{host}[/bold]")
    console.print(f"  [dim]model[/dim]    [bold]{model_path}[/bold]\n")

    with console.status("[bold green]Checking remote aphex installation..."):
        if not check_remote_aphex(host):
            err_console.print(
                f"aphex not found on {host}.\n"
                "Install it with: ssh {host} 'pip install aphex'"
            )
            raise typer.Exit(code=1)

    args: list[str] = [
        "--input-shape", input_shape,
        "--objective", objective,
        "--batch-sizes", batch_sizes,
        "--timeout", str(timeout),
        "--format", format_,
    ]
    if max_latency_ms is not None:
        args += ["--max-latency-ms", str(max_latency_ms)]
    if max_memory_mb is not None:
        args += ["--max-memory-mb", str(max_memory_mb)]
    if min_throughput_rps is not None:
        args += ["--min-throughput-rps", str(min_throughput_rps)]
    if max_quality_loss is not None:
        args += ["--max-quality-loss", str(max_quality_loss)]
    if serving is not None:
        args += ["--serving", serving]

    local_output = output if (output and str(output) != "") else Path("deployment.yaml")

    console.print(f"  [dim]output[/dim]   [bold]{local_output}[/bold]\n")

    import tempfile
    tmp_metrics: Path | None = None
    if report is not None and metrics is None:
        tmp_metrics = Path(tempfile.mktemp(suffix=".json"))
    local_metrics = metrics or tmp_metrics

    try:
        exit_code = run_remote_optimize(host, model_path, args, local_output, local_metrics)
        if exit_code != 0:
            raise typer.Exit(code=exit_code)

        if local_metrics and local_metrics.exists():
            if metrics:
                console.print(f"  [dim]metrics →[/dim] [bold]{metrics}[/bold]")
            if report:
                from infermap.report import render_report
                payload = json.loads(local_metrics.read_text())
                render_report(payload, report)
                console.print(f"  [dim]report →[/dim] [bold]{report}[/bold]")
            console.print()
    finally:
        if tmp_metrics and tmp_metrics.exists():
            tmp_metrics.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Shared benchmark runner
# ---------------------------------------------------------------------------


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_CALIB_MAX_SAMPLES = 32


def _load_calibration(path: Optional[Path], input_shape: list[int]) -> list | None:
    if path is None:
        return None
    if path.is_dir():
        return _load_calibration_dir(path, input_shape)
    import torch
    raw = torch.load(path, weights_only=False)
    if isinstance(raw, torch.Tensor):
        return [raw[i : i + 1] for i in range(min(raw.size(0), _CALIB_MAX_SAMPLES))]
    if isinstance(raw, list):
        return raw[:_CALIB_MAX_SAMPLES]
    return None


def _load_calibration_dir(path: Path, input_shape: list[int]) -> list | None:
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image

    if len(input_shape) != 3:
        err_console.print(
            f"[yellow]Warning: --calibration-data directory requires a 3D input shape "
            f"(C H W); got {input_shape}. Provide a .pt file for non-image inputs.[/yellow]"
        )
        return None

    c, h, w = input_shape
    files = sorted(f for f in path.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS)
    files = files[:_CALIB_MAX_SAMPLES]

    if not files:
        err_console.print(
            f"[yellow]Warning: no image files found in {path} "
            f"({', '.join(sorted(_IMAGE_EXTENSIONS))}).[/yellow]"
        )
        return None

    tensors = []
    for f in files:
        img = Image.open(f).convert("RGB" if c == 3 else "L")
        t = TF.to_tensor(img)                           # (C, H, W), values in [0, 1]
        t = TF.resize(t, [h, w], antialias=True)        # resize to target spatial dims
        if c == 3:
            t = TF.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD)
        tensors.append(t.unsqueeze(0))                  # (1, C, H, W)

    return tensors


def _load_eval(path: Optional[Path], info: object, label_col: str) -> object:
    if path is None:
        return None
    from infermap.evaluator import load_eval_data
    from infermap.inspector import ModelInfo
    assert isinstance(info, ModelInfo)
    try:
        return load_eval_data(path, info, label_col=label_col)
    except Exception as exc:
        err_console.print(f"[yellow]Warning: could not load eval data: {exc}[/yellow]")
        return None


def _load_infer_fn(spec: Optional[str]) -> object:
    if spec is None:
        return None
    from infermap.evaluator import load_infer_fn
    return load_infer_fn(spec)


def _run_eval(
    eval_dataset: object,
    infer_fn: object,
    json_mode: bool,
    results: list | None = None,
    model: object | None = None,
    info: object | None = None,
) -> tuple[str, float] | None:
    """Run accuracy evaluation.

    Path 1 — infer_fn provided: calls evaluate_baseline, returns (metric, score).
    Path 2 — no infer_fn but results+model+info provided: calls fill_accuracy_drop
              to mutate BenchmarkResult.accuracy_drop in-place, returns None.
    """
    from infermap.evaluator import EvalDataset
    if eval_dataset is None:
        return None
    assert isinstance(eval_dataset, EvalDataset)

    if infer_fn is not None:
        from infermap.evaluator import evaluate_baseline
        try:
            if not json_mode:
                console.print(
                    f"  [dim]evaluating accuracy ({eval_dataset.metric}) "
                    f"on {len(eval_dataset.inputs)} samples...[/dim]"
                )
            score = evaluate_baseline(infer_fn, eval_dataset)
            return eval_dataset.metric, score
        except Exception as exc:
            err_console.print(f"[yellow]Warning: accuracy eval failed: {exc}[/yellow]")
            return None

    if results is not None and model is not None and info is not None:
        from infermap.evaluator import fill_accuracy_drop
        try:
            if not json_mode:
                console.print(
                    f"  [dim]measuring accuracy drop ({eval_dataset.metric}) "
                    f"on {len(eval_dataset.inputs)} samples...[/dim]"
                )
            fill_accuracy_drop(results, model, info, eval_dataset)
        except Exception as exc:
            err_console.print(f"[yellow]Warning: accuracy drop measurement failed: {exc}[/yellow]")

    return None


def _select_with_fingerprint(
    candidates: list,
    info: object,
    hw: object,
    json_mode: bool,
) -> list:
    from infermap.inspector import ModelInfo
    from infermap.profiler import HardwareProfile
    from infermap.selector import select_candidates

    assert isinstance(info, ModelInfo)
    assert isinstance(hw, HardwareProfile)

    selected, rationale = select_candidates(candidates, info, hw, k=4)
    if not json_mode:
        console.print(f"  [dim]{rationale}[/dim]\n")
    return selected


def _prune_with_cost_model(
    candidates: list,
    info: object,
    hw: object,
    batch_size: int,
    json_mode: bool,
) -> tuple[list, list]:
    from infermap.cost_model import prune_candidates
    from infermap.inspector import ModelInfo
    from infermap.profiler import HardwareProfile

    assert isinstance(info, ModelInfo)
    assert isinstance(hw, HardwareProfile)

    kept, estimates = prune_candidates(candidates, info, hw, batch_size=batch_size)
    pruned = len(candidates) - len(kept)
    if pruned > 0 and not json_mode:
        console.print(
            f"  [dim]cost model pruned {pruned} candidate(s) predicted "
            f">10× slower than the best estimate[/dim]\n"
        )
    return kept, estimates


def _run_candidates(
    plugin: object,
    candidates: list,
    model: object,
    model_info: object,
    shape: list[int],
    batch_sizes: list[int],
    warmup: int,
    iters: int,
    timeout_s: float | None,
    calibration_inputs: list | None = None,
    json_mode: bool = False,
) -> list:
    from infermap.plugin import ModelPlugin
    assert isinstance(plugin, ModelPlugin)

    results = []
    total = len(candidates) * len(batch_sizes)

    if json_mode:
        for cand in candidates:
            for bs in batch_sizes:
                r = plugin.benchmark(  # type: ignore[arg-type]
                    cand, model, model_info, shape, bs, warmup, iters,
                    timeout_s, calibration_inputs,
                )
                results.append(r)
        return results

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
            f"racing {len(candidates)} backends × {len(batch_sizes)} batch sizes",
            total=total,
        )
        for cand in candidates:
            for bs in batch_sizes:
                progress.update(task, description=f"{cand.description}  bs={bs}")
                r = plugin.benchmark(  # type: ignore[arg-type]
                    cand, model, model_info, shape, bs, warmup, iters,
                    timeout_s, calibration_inputs,
                )
                results.append(r)
                if r.ok:
                    progress.console.print(
                        f"  [green]✓[/green]  {cand.description:<44}"
                        f"  [dim]bs={bs:<3}[/dim]"
                        f"  [bold]{r.latency_p50_ms:>8.2f} ms[/bold]"
                        f"  [dim]{r.throughput_rps:>7.0f} req/s[/dim]"
                    )
                else:
                    progress.console.print(
                        f"  [red]✗[/red]  [dim]{cand.description:<44}  bs={bs}[/dim]"
                        f"  [red]{(r.error or '')[:55]}[/red]"
                    )
                progress.advance(task)

    console.print()
    return results


# ---------------------------------------------------------------------------
# JSON output helpers
# ---------------------------------------------------------------------------


def _result_to_dict(r: object) -> dict:
    from infermap.benchmark import BenchmarkResult
    assert isinstance(r, BenchmarkResult)
    return {
        "backend": r.candidate.backend,
        "dtype": r.candidate.dtype,
        "device": r.candidate.device,
        "description": r.candidate.description,
        "batch_size": r.batch_size,
        "ok": r.ok,
        "latency_p50_ms": r.latency_p50_ms if r.ok else None,
        "latency_p95_ms": r.latency_p95_ms if r.ok else None,
        "latency_p99_ms": r.latency_p99_ms if r.ok else None,
        "throughput_rps": r.throughput_rps if r.ok else None,
        "memory_mb": r.memory_mb if r.ok else None,
        "accuracy_drop": r.accuracy_drop,
        "error": r.error,
    }


def _build_metrics_payload(results: list, recommendation: object = None) -> dict:
    payload: dict = {"results": [_result_to_dict(r) for r in results]}
    if recommendation is not None:
        from infermap.recommender import Recommendation
        assert isinstance(recommendation, Recommendation)
        r = recommendation.result
        payload["recommendation"] = {
            **_result_to_dict(r),
            "rationale": recommendation.rationale,
            "pareto_frontier_count": len(recommendation.pareto_frontier),
        }
    return payload


def _emit_json(results: list, recommendation: object = None) -> None:
    print(json.dumps(_build_metrics_payload(results, recommendation), indent=2))


# ---------------------------------------------------------------------------
# Rich output helpers
# ---------------------------------------------------------------------------


def _print_header(label: str) -> None:
    console.print()
    console.print(Rule(f"[bold]aphex[/bold] [dim]·[/dim] [dim]{label}[/dim]", style="dim", align="left"))
    console.print()


def _prompt_unlikely_or_abort() -> bool:
    """Prompt on 'unlikely' preflight. Returns True if the user chose to relax constraints."""
    console.print()
    console.print("  [yellow][1][/yellow] Benchmark anyway")
    console.print("  [yellow][2][/yellow] Relax constraint and benchmark")
    console.print("  [dim][3] Abort[/dim]")
    choice = typer.prompt("\nChoice", default="1")
    choice = choice.strip()
    if choice == "3":
        raise typer.Exit(code=1)
    console.print()
    return choice == "2"


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
        if hw.accelerator.memory_bandwidth_gbps is not None:
            t.add_row("mem bandwidth", f"{hw.accelerator.memory_bandwidth_gbps:.0f} GB/s")
        if hw.accelerator.fp16_tflops is not None:
            t.add_row("fp16 tflops", f"{hw.accelerator.fp16_tflops:.1f}")
        if hw.accelerator.int8_tops is not None and hw.accelerator.int8_tops > 0:
            t.add_row("int8 tops", f"{hw.accelerator.int8_tops:.0f}")
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
    # Architecture fingerprint (V2)
    if info.num_layers is not None:
        t.add_row("layers", str(info.num_layers))
    if info.num_attention_heads is not None:
        t.add_row("attn heads", str(info.num_attention_heads))
    if info.hidden_size is not None:
        t.add_row("hidden size", str(info.hidden_size))
    if info.vocab_size is not None:
        t.add_row("vocab size", f"{info.vocab_size:,}")
    if info.max_sequence_length is not None:
        t.add_row("max seq len", str(info.max_sequence_length))
    if info.num_conv_layers is not None:
        t.add_row("conv layers", str(info.num_conv_layers))
    if info.num_trees is not None:
        t.add_row("trees", str(info.num_trees))
    if info.max_depth is not None:
        t.add_row("max depth", str(info.max_depth))
    if info.num_features is not None:
        t.add_row("features", str(info.num_features))
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
    t.add_column("backend", min_width=30)
    t.add_column("bs", justify="right", width=4)
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
            f"[dim]{r.batch_size}[/dim]",
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
            f"[dim]{r.batch_size}[/dim]",
            "[dim]—[/dim]", "[dim]—[/dim]", "[dim]—[/dim]", "[dim]—[/dim]",
        ]
        if has_accuracy:
            row.append("[dim]—[/dim]")
        row.append(f"[red]{(r.error or '')[:28]}[/red]")
        t.add_row(*row)

    console.print(Rule("[dim]results[/dim]", style="dim"))
    console.print(t)


def _print_check_results(results: list) -> None:
    from infermap.checker import CheckResult

    console.print(Rule("[dim]results[/dim]", style="dim"))

    label_w = max(len(r.label) for r in results)
    for r in results:
        assert isinstance(r, CheckResult)
        sign = "+" if r.delta_pct >= 0 else ""
        delta_str = f"Δ {sign}{r.delta_pct:.1f}%"

        if r.metric in ("latency_p50_ms", "latency_p95_ms"):
            base_str = f"{r.baseline:.2f} ms"
            obs_str = f"{r.observed:.2f} ms"
        elif r.metric == "throughput_rps":
            base_str = f"{r.baseline:.0f}/s"
            obs_str = f"{r.observed:.0f}/s"
        elif r.metric == "memory_mb":
            base_str = f"{r.baseline:.0f} MB"
            obs_str = f"{r.observed:.0f} MB"
        else:
            base_str = f"{r.baseline:.4f}"
            obs_str = f"{r.observed:.4f}"

        if r.threshold_pct is None:
            status = "[dim]—[/dim]"
            threshold_str = ""
        elif r.passed:
            status = "[green]✓[/green]"
            threshold_str = f"[dim](≤{r.threshold_pct:.0f}%)[/dim]"
        else:
            status = "[red]✗[/red]"
            threshold_str = f"[red](≤{r.threshold_pct:.0f}%)[/red]"

        delta_style = "red" if not r.passed else ("dim" if r.delta_pct <= 0 else "")
        delta_fmt = f"[{delta_style}]{delta_str}[/{delta_style}]" if delta_style else delta_str

        console.print(
            f"  {status}  {r.label:<{label_w}}  "
            f"[dim]{base_str}[/dim] [dim]→[/dim] {obs_str:<12}"
            f"  {delta_fmt:<14}  {threshold_str}"
        )

    console.print()


def _print_recommendation(rec: object, eval_result: tuple | None = None) -> None:
    from infermap.recommender import Recommendation

    assert isinstance(rec, Recommendation)
    r = rec.result

    acc_line = (
        f"  [dim]acc drop[/dim]  {r.accuracy_drop * 100:.2f}%\n"
        if r.accuracy_drop is not None
        else ""
    )
    eval_line = (
        f"  [dim]{eval_result[0]}[/dim]   [bold]{eval_result[1]:.4f}[/bold]\n"
        if eval_result is not None
        else ""
    )
    stats = (
        f"  [dim]batch[/dim]  {r.batch_size}\n"
        f"  [dim]p50[/dim]    [bold]{r.latency_p50_ms:.2f} ms[/bold]\n"
        f"  [dim]p95[/dim]    {r.latency_p95_ms:.2f} ms\n"
        f"  [dim]req/s[/dim]  [bold]{r.throughput_rps:.0f}[/bold]\n"
        f"  [dim]mb[/dim]     {r.memory_mb:.0f}\n"
        + acc_line
        + eval_line
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
