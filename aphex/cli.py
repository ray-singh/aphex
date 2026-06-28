"""CLI entry point — Typer app with Rich output."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Must be set before libomp is loaded (i.e. before any torch import).
# Prevents crash when torch and onnxruntime each ship their own libomp.dylib on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if TYPE_CHECKING:
    from aphex.evaluator import EvalDataset  # noqa: E402

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
    from aphex.profiler import profile_hardware
    from aphex.registry import get_plugin

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
    target_throughput: float | None = typer.Option(
        None, "--target-throughput", help="Required throughput in req/s"
    ),
) -> None:
    """Run a fast feasibility check before benchmarking."""
    from aphex.preflight import run_preflight
    from aphex.profiler import profile_hardware
    from aphex.registry import get_plugin

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
        "3,224,224", "--input-shape",
        help="Input tensor shape (no batch dim), e.g. 3,224,224. "
             "Multi-input: 'input_ids:128:long;attention_mask:128:long'.",
    ),
    batch_sizes: str = typer.Option("1,2,4,8", "--batch-sizes", help="Comma-separated batch sizes to sweep, e.g. 1,2,4,8"),
    warmup: int = typer.Option(10, help="Warm-up iterations"),
    iters: int = typer.Option(100, help="Measurement iterations"),
    timeout: float = typer.Option(180.0, "--timeout", help="Per-candidate timeout in seconds (0 = no limit)"),
    calibration_data: Path | None = typer.Option(
        None, "--calibration-data",
        help="Path to a .pt file (or image directory) with unlabelled calibration inputs for int8 quantization.",
    ),
    eval_data: str | None = typer.Option(
        None, "--eval",
        help="Path to labelled eval data (.pt, .parquet, .csv, or image dir), or a cloud URI (s3://, gs://). For real accuracy measurement.",
    ),
    eval_label_col: str = typer.Option(
        "label", "--eval-label-col",
        help="Column name for labels when --eval points to a .parquet or .csv file.",
    ),
    infer_fn: str | None = typer.Option(
        None, "--infer-fn",
        help="Inference callable as 'path/to/module.py:function'. Required with --eval. "
             "Function signature: predict(inputs: list) -> np.ndarray.",
    ),
    max_quality_loss: float | None = typer.Option(
        None, "--max-quality-loss",
        help="Maximum acceptable quality loss vs original model baseline (0.0–1.0). Requires --calibration-data or --eval.",
    ),
    format_: str = typer.Option("table", "--format", help="Output format: table | json"),
    jobs: int = typer.Option(
        1, "--jobs", "-j", min=1,
        help="Number of (candidate, batch_size) pairs to benchmark in parallel. "
             "Keep at 1 for accurate latency/memory; raise only on systems with "
             "many isolated CPU cores and no shared accelerator.",
    ),
) -> None:
    """Benchmark all candidate deployment strategies."""
    from aphex.preflight import run_preflight
    from aphex.profiler import profile_hardware
    from aphex.registry import get_plugin

    input_spec = _parse_input_spec(input_shape)
    shape = input_spec.primary_shape
    bs_list = _parse_batch_sizes(batch_sizes)
    timeout_s = timeout if timeout > 0 else None
    json_mode = format_ == "json"

    if not json_mode:
        _print_header("benchmark")

    if max_quality_loss is not None and calibration_data is None and eval_data is None:
        err_console.print(
            "--max-quality-loss requires --calibration-data or --eval so a quality score can be measured."
        )
        raise typer.Exit(code=1)

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
    calib = _resolve_calibration(calibration_data, shape, eval_dataset)
    candidates = plugin.generate_candidates(info, hw)
    candidates, cost_estimates = _prune_with_cost_model(candidates, info, hw, bs_list[0], json_mode)
    results = _run_candidates(plugin, candidates, model, info, shape, bs_list, warmup, iters, timeout_s, calib, json_mode=json_mode, jobs=jobs, input_spec=input_spec)
    if eval_dataset is not None:
        _run_eval(eval_dataset, fn, json_mode, results=results, model=model, info=info)
    if json_mode:
        _emit_json(results, hw=hw)
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
    target: str | None = typer.Option(
        None, "--target",
        help="Target cloud instance type or GPU alias, e.g. ml.g4dn.xlarge, a2-highgpu-1g, a100. "
             "Skips local hardware profiling and optimises for the specified hardware. "
             "Run 'aphex targets' to see all options.",
    ),
    remote: str | None = typer.Option(
        None, "--remote",
        help="Run benchmarks on a remote machine via SSH, e.g. user@gpu-host. "
             "Requires aphex installed on the remote machine.",
    ),
    max_latency_ms: float | None = typer.Option(None, "--max-latency-ms"),
    max_memory_mb: float | None = typer.Option(None, "--max-memory-mb"),
    min_throughput_rps: float | None = typer.Option(None, "--min-throughput-rps"),
    timeout: float = typer.Option(180.0, "--timeout", help="Per-candidate timeout in seconds (0 = no limit)"),
    calibration_data: Path | None = typer.Option(
        None, "--calibration-data",
        help="Path to a .pt file (or image directory) with unlabelled calibration inputs for int8 quantization.",
    ),
    eval_data: str | None = typer.Option(
        None, "--eval",
        help="Path to labelled eval data (.pt, .parquet, .csv, or image dir), or a cloud URI (s3://, gs://). Required for quality measurement.",
    ),
    eval_label_col: str = typer.Option(
        "label", "--eval-label-col",
        help="Column name for labels when --eval points to a .parquet or .csv file.",
    ),
    infer_fn: str | None = typer.Option(
        None, "--infer-fn",
        help="Inference callable as 'path/to/module.py:function'. Required with --eval. "
             "Function signature: predict(inputs: list) -> np.ndarray.",
    ),
    max_accuracy_loss: float | None = typer.Option(
        None, "--max-accuracy-loss",
        help="Max acceptable accuracy drop vs original model baseline (0.0–1.0). For classification models.",
    ),
    max_f1_loss: float | None = typer.Option(
        None, "--max-f1-loss",
        help="Max acceptable macro-F1 drop vs original model baseline (0.0–1.0). For classification models.",
    ),
    max_mae_loss: float | None = typer.Option(
        None, "--max-mae-loss",
        help="Max acceptable relative MAE increase vs original model baseline (0.0–1.0). For regression models.",
    ),
    max_rmse_loss: float | None = typer.Option(
        None, "--max-rmse-loss",
        help="Max acceptable relative RMSE increase vs original model baseline (0.0–1.0). For regression models.",
    ),
    output: Path | None = typer.Option(
        Path("deployment.yaml"), "--output",
        help="Write deployment config to this path (default: deployment.yaml). Pass '' to skip.",
    ),
    serving: str | None = typer.Option(
        None, "--serving",
        help="Generate serving configs: triton, torchserve, bentoml, fastapi",
    ),
    format_: str = typer.Option("table", "--format", help="Output format: table | json"),
    report: Path | None = typer.Option(None, "--report", help="Write HTML benchmark report to this path"),
    metrics: Path | None = typer.Option(None, "--metrics", help="Write benchmark metrics JSON to this path"),
    jobs: int = typer.Option(
        1, "--jobs", "-j", min=1,
        help="Number of (candidate, batch_size) pairs to benchmark in parallel. "
             "Keep at 1 for accurate latency/memory.",
    ),
) -> None:
    """Benchmark all candidates and recommend the optimal deployment strategy."""
    from aphex.preflight import run_preflight
    from aphex.profiler import profile_hardware
    from aphex.recommender import recommend
    from aphex.registry import get_plugin

    input_spec = _parse_input_spec(input_shape)
    shape = input_spec.primary_shape
    bs_list = _parse_batch_sizes(batch_sizes)
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
            max_accuracy_loss=max_accuracy_loss,
            max_f1_loss=max_f1_loss,
            max_mae_loss=max_mae_loss,
            max_rmse_loss=max_rmse_loss,
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

    # Resolve dedicated quality-loss flags into a single (metric, threshold) pair.
    _metric_flags: dict[str, float | None] = {
        "accuracy": max_accuracy_loss,
        "f1_macro": max_f1_loss,
        "mae": max_mae_loss,
        "rmse": max_rmse_loss,
    }
    _specified = {k: v for k, v in _metric_flags.items() if v is not None}
    if len(_specified) > 1:
        err_console.print(
            "Specify at most one quality-loss flag. Got: "
            + ", ".join(f"--max-{k}-loss" for k in _specified)
        )
        raise typer.Exit(code=1)
    metric_override: str | None = next(iter(_specified), None)
    max_quality_loss: float | None = next(iter(_specified.values()), None)

    if eval_data is None:
        err_console.print(
            "--eval is required: provide a labelled test dataset so aphex can measure "
            "quality loss from quantization.\n"
            "  Example: --eval val.pt --max-accuracy-loss 0.02"
        )
        raise typer.Exit(code=1)

    if target is not None:
        from aphex.cloud.instances import get_instance_profile
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
    if pf.category == "unlikely" and not json_mode and _prompt_unlikely_or_abort():
        min_throughput_rps = None

    model = plugin.load(model_path)
    eval_dataset = _load_eval(eval_data, info, eval_label_col, metric_override=metric_override, required=True)
    fn = _load_infer_fn(infer_fn)
    calib = _resolve_calibration(calibration_data, shape, eval_dataset)
    candidates = plugin.generate_candidates(info, hw)
    candidates = _select_with_fingerprint(candidates, info, hw, json_mode)
    results = _run_candidates(plugin, candidates, model, info, shape, bs_list, 10, 100, timeout_s, calib, json_mode=json_mode, jobs=jobs, input_spec=input_spec)
    eval_result = _run_eval(eval_dataset, fn, json_mode, results=results, model=model, info=info) if eval_dataset is not None else None

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
        from aphex.deployment import build_config, write_yaml
        from aphex.system_recommender import recommend_system_config
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
            from aphex.serving import SUPPORTED_FRAMEWORKS, generate_serving_config
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
        _emit_json(results, rec, hw=hw)
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
        payload = _build_metrics_payload(results, rec, hw=hw)
        if metrics:
            metrics.write_text(json.dumps(payload, indent=2))
            if not json_mode:
                console.print(f"  [dim]metrics →[/dim] [bold]{metrics}[/bold]")
        if report:
            from aphex.report import render_report
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
    backend: str | None = typer.Option(
        None, "--backend",
        help="Target backend, e.g. onnx_int8_cpu, pytorch_fp16, tensorrt_fp16.",
    ),
    input_shape: str | None = typer.Option(
        None, "--input-shape",
        help="Input tensor shape (no batch dim), e.g. 3,224,224. Multi-input: "
             "'input_ids:128:long;attention_mask:128:long'. Required unless --from-config is used.",
    ),
    output: Path | None = typer.Option(
        None, "--output",
        help="Output artifact path. Auto-derived from model name and backend if omitted.",
    ),
    from_config: Path | None = typer.Option(
        None, "--from-config",
        help="Read backend and input-shape from a deployment.yaml written by aphex optimize.",
    ),
    calibration_data: Path | None = typer.Option(
        None, "--calibration-data",
        help="Calibration inputs (.pt file or image dir) required for int8 backends.",
    ),
    verify: bool = typer.Option(
        True, "--verify/--no-verify",
        help="After conversion, check the artifact reproduces the source model's "
             "outputs. Exits non-zero on a parity failure.",
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
    from aphex.converter import ALL_CONVERTIBLE, default_output_path, read_deployment_yaml
    from aphex.converter import convert as _convert
    from aphex.registry import get_plugin

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

    input_spec = _parse_input_spec(resolved_shape_str)
    shape = input_spec.primary_shape

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
            written = _convert(model, resolved_backend, shape, out_path, calib, input_spec)
    except Exception as exc:
        err_console.print(f"Conversion failed: {exc}")
        raise typer.Exit(code=1)

    console.print()
    for p in written:
        size_mb = p.stat().st_size / 1e6
        console.print(f"  [green]✓[/green]  {p}  [dim]({size_mb:.1f} MB)[/dim]")
    console.print()

    if verify:
        from aphex.parity import verify_artifact

        with console.status("[bold green]Verifying artifact parity..."):
            result = verify_artifact(
                model, resolved_backend, written, shape, sample_inputs=calib,
                input_spec=input_spec,
            )

        if result.skipped:
            console.print(f"  [dim]parity check skipped — {result.reason}[/dim]")
        elif result.passed:
            console.print(f"  [green]✓[/green]  {result.summary()}")
        else:
            err_console.print(f"  ✗  {result.summary()}")
            err_console.print(
                f"     Artifact outputs diverge from the source model beyond tolerance "
                f"(cosine ≥ {result.cosine_threshold}). The conversion is likely broken; "
                f"do not deploy this artifact."
            )
            raise typer.Exit(code=1)
        console.print()


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


@app.command()
def check(
    model_path: Path = typer.Argument(..., help="Path to the saved model"),
    from_config: Path = typer.Option(..., "--from-config", help="deployment.yaml written by aphex optimize"),
    max_latency_delta: float | None = typer.Option(
        10.0, "--max-latency-delta",
        help="Max allowed latency increase in percent (default 10). Pass 0 to disable.",
    ),
    min_throughput_delta: float | None = typer.Option(
        None, "--min-throughput-delta",
        help="Max allowed throughput drop in percent, e.g. 10 means throughput may not drop >10%%.",
    ),
    max_memory_delta: float | None = typer.Option(
        None, "--max-memory-delta",
        help="Max allowed memory increase in percent.",
    ),
    max_accuracy_drop: float | None = typer.Option(
        None, "--max-accuracy-drop",
        help="Max allowed absolute accuracy drop vs baseline eval_score (requires --eval + --infer-fn).",
    ),
    warmup: int = typer.Option(5, help="Warm-up iterations (default lower than optimize for speed)"),
    iters: int = typer.Option(50, help="Measurement iterations"),
    eval_data: str | None = typer.Option(None, "--eval", help="Labelled eval data for accuracy re-check (path or cloud URI)"),
    eval_label_col: str = typer.Option("label", "--eval-label-col"),
    infer_fn: str | None = typer.Option(None, "--infer-fn", help="Inference callable 'module.py:fn'"),
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
    from aphex.candidates import DeploymentCandidate
    from aphex.checker import run_check
    from aphex.converter import read_deployment_yaml
    from aphex.registry import get_plugin

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
            from aphex.evaluator import evaluate_baseline
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
# distill
# ---------------------------------------------------------------------------


@app.command()
def distill(
    model_path: Path = typer.Argument(..., help="Teacher model path (.pt / .pkl)"),
    student: str = typer.Option(
        ..., "--student",
        help="Student factory as 'path/to/module.py:make_student'. The callable "
             "must take no arguments and return a torch.nn.Module.",
    ),
    eval_data: str = typer.Option(
        ..., "--eval",
        help="Labelled dataset for distillation (.pt / .csv / .parquet / image dir / s3:// / gs://).",
    ),
    eval_label_col: str = typer.Option("label", "--eval-label-col"),
    epochs: int = typer.Option(3, "--epochs", min=1),
    batch_size: int = typer.Option(32, "--batch-size", min=1),
    lr: float = typer.Option(1e-3, "--lr"),
    temperature: float = typer.Option(
        4.0, "--temperature",
        help="KD softmax temperature. Higher = softer teacher distribution.",
    ),
    alpha: float = typer.Option(
        0.7, "--alpha", min=0.0, max=1.0,
        help="Weight on KD loss; (1 - alpha) goes to hard-label CE (classification).",
    ),
    task: str = typer.Option(
        "classification", "--task",
        help="'classification' or 'regression'. Determines the loss formulation.",
    ),
    device: str = typer.Option(
        "cpu", "--device",
        help="Training device: cpu / cuda / mps.",
    ),
    output: Path = typer.Option(
        Path("student.pt"), "--output",
        help="Where to write the distilled student state_dict.",
    ),
    report_path: Path | None = typer.Option(
        None, "--report",
        help="Optional JSON report with losses, parameter counts, and scores.",
    ),
) -> None:
    """Knowledge-distill a teacher model into a smaller student.

    \b
    Distillation is the only aphex command that performs gradient updates: it
    trains *student* to match *teacher*'s soft outputs (KL divergence on
    temperature-scaled logits) plus a hard-label term for classification.

    \b
    The student is a user-supplied nn.Module — aphex doesn't guess architecture.
    Provide a factory function and aphex handles the training loop and scoring.

    \b
    Examples
    --------
      aphex distill teacher.pt --student make_student.py:tiny_mlp \\
            --eval val.pt --epochs 5

      aphex distill resnet50.pt --student make_student.py:resnet18 \\
            --eval imagenet_subset/ --task classification \\
            --temperature 3 --alpha 0.8 --device cuda
    """
    from aphex.distillation import DistillConfig, load_student_factory, score
    from aphex.distillation import distill as _distill_fn
    from aphex.evaluator import load_eval_data
    from aphex.registry import get_plugin

    _print_header("distill")

    cfg = DistillConfig(
        epochs=epochs, batch_size=batch_size, lr=lr,
        temperature=temperature, alpha=alpha, task=task, device=device,
    )

    with console.status("[bold green]Loading teacher..."):
        plugin = get_plugin(model_path)
        info = plugin.inspect(model_path)
        teacher = plugin.load(model_path)

    with console.status("[bold green]Building student..."):
        try:
            factory = load_student_factory(student)
            student_model = factory()
        except Exception as exc:
            err_console.print(f"Failed to load student factory: {exc}")
            raise typer.Exit(code=1)

    with console.status("[bold green]Loading eval data..."):
        eval_dataset = load_eval_data(eval_data, info, label_col=eval_label_col)

    console.print(
        f"  [dim]teacher params[/dim]  [bold]{sum(p.numel() for p in teacher.parameters()):,}[/bold]"
    )
    console.print(
        f"  [dim]student params[/dim]  [bold]{sum(p.numel() for p in student_model.parameters()):,}[/bold]"
    )
    console.print(
        f"  [dim]epochs={epochs}  batch={batch_size}  lr={lr}  "
        f"temp={temperature}  alpha={alpha}  task={task}  device={device}[/dim]\n"
    )

    try:
        with console.status("[bold green]Distilling..."):
            distilled, report = _distill_fn(
                teacher, student_model,
                eval_dataset.inputs, eval_dataset.labels, cfg,
            )
    except Exception as exc:
        err_console.print(f"Distillation failed: {exc}")
        raise typer.Exit(code=1)

    try:
        teacher_score = score(teacher, eval_dataset.inputs, eval_dataset.labels, task, device)
        student_score = score(distilled, eval_dataset.inputs, eval_dataset.labels, task, device)
    except Exception as exc:
        err_console.print(f"[yellow]Warning: scoring failed: {exc}[/yellow]")
        teacher_score = None
        student_score = None
    report.teacher_score = teacher_score
    report.student_score = student_score
    report.metric = "accuracy" if task == "classification" else "rmse"

    import torch as _torch
    _torch.save(distilled.state_dict(), output)

    console.print(Rule("[dim]results[/dim]", style="dim"))
    console.print(f"  [dim]compression[/dim]  [bold]{report.compression_ratio:.1f}×[/bold]")
    console.print(
        f"  [dim]final loss[/dim]   [bold]{report.epoch_losses[-1]:.4f}[/bold]  "
        f"[dim](first epoch {report.epoch_losses[0]:.4f})[/dim]"
    )
    if teacher_score is not None and student_score is not None:
        console.print(
            f"  [dim]{report.metric}[/dim]       teacher [bold]{teacher_score:.4f}[/bold]  "
            f"→  student [bold]{student_score:.4f}[/bold]"
        )
    console.print(f"\n  [green]✓[/green]  student state_dict → [bold]{output}[/bold]\n")

    if report_path is not None:
        payload = {
            "config": {
                "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr,
                "temperature": cfg.temperature, "alpha": cfg.alpha,
                "task": cfg.task, "device": cfg.device,
            },
            "teacher_params": report.teacher_params,
            "student_params": report.student_params,
            "compression_ratio": report.compression_ratio,
            "epoch_losses": report.epoch_losses,
            "metric": report.metric,
            "teacher_score": report.teacher_score,
            "student_score": report.student_score,
        }
        report_path.write_text(json.dumps(payload, indent=2))
        console.print(f"  [dim]report →[/dim] [bold]{report_path}[/bold]\n")


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


@app.command()
def targets() -> None:
    """List all supported cloud instance types and GPU aliases for --target."""
    from aphex.cloud.instances import list_targets

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
    version: str | None = typer.Option(
        None, "--version",
        help="Version tag (default: auto-generated timestamp, e.g. v20260616-120000)",
    ),
    registry: str | None = typer.Option(
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

    from aphex.cloud.config import get_registry_uri
    from aphex.cloud.registry import push as _push
    from aphex.cloud.storage import get_backend

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

    ver = version or datetime.datetime.now(datetime.UTC).strftime("v%Y%m%d-%H%M%S")

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
    registry: str | None = typer.Option(
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
    from aphex.cloud.config import get_registry_uri
    from aphex.cloud.registry import pull as _pull
    from aphex.cloud.storage import get_backend

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
    name: str | None = typer.Argument(
        None, help="Model name to list versions (omit to list all models)"
    ),
    registry: str | None = typer.Option(
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
    from aphex.cloud.config import get_registry_uri
    from aphex.cloud.registry import list_models, list_versions
    from aphex.cloud.storage import get_backend

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
    max_latency_ms: float | None,
    max_memory_mb: float | None,
    min_throughput_rps: float | None,
    max_accuracy_loss: float | None,
    max_f1_loss: float | None,
    max_mae_loss: float | None,
    max_rmse_loss: float | None,
    timeout: float,
    output: Path | None,
    serving: str | None,
    format_: str,
    target: str | None,
    calibration_data: Path | None,
    eval_data: str | None,
    infer_fn: str | None,
    report: Path | None = None,
    metrics: Path | None = None,
) -> None:
    from aphex.cloud.remote import check_remote_aphex, run_remote_optimize

    _print_header("optimize · remote")

    if target is not None:
        err_console.print(
            "[yellow]Warning: --target is ignored with --remote "
            "(the remote machine profiles its own hardware).[/yellow]"
        )

    if eval_data is None:
        err_console.print(
            "--eval is required: provide a labelled test dataset so aphex can measure "
            "quality loss from quantization.\n"
            "  Example: --eval val.pt --max-accuracy-loss 0.02"
        )
        raise typer.Exit(code=1)

    for flag, val in [("--calibration-data", calibration_data), ("--infer-fn", infer_fn)]:
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
                f"Install it with: ssh {host} 'pip install aphex-ml'"
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
    if max_accuracy_loss is not None:
        args += ["--max-accuracy-loss", str(max_accuracy_loss)]
    if max_f1_loss is not None:
        args += ["--max-f1-loss", str(max_f1_loss)]
    if max_mae_loss is not None:
        args += ["--max-mae-loss", str(max_mae_loss)]
    if max_rmse_loss is not None:
        args += ["--max-rmse-loss", str(max_rmse_loss)]
    if serving is not None:
        args += ["--serving", serving]

    local_output = output if (output and str(output) != "") else Path("deployment.yaml")

    console.print(f"  [dim]output[/dim]   [bold]{local_output}[/bold]\n")

    import tempfile
    tmp_metrics: Path | None = None
    if report is not None and metrics is None:
        # NamedTemporaryFile(delete=False) avoids the mktemp race; we manage
        # cleanup ourselves in the finally below.
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            tmp_metrics = Path(fh.name)
    local_metrics = metrics or tmp_metrics

    # Pass cloud URIs as strings; convert local paths to Path so scp upload works.
    from aphex.cloud.remote import _is_eval_uri
    eval_remote_arg: Path | str | None = (
        eval_data if (eval_data is None or _is_eval_uri(eval_data)) else Path(eval_data)
    )

    try:
        exit_code = run_remote_optimize(host, model_path, args, local_output, local_metrics, eval_remote_arg)
        if exit_code != 0:
            raise typer.Exit(code=exit_code)

        if local_metrics and local_metrics.exists():
            if metrics:
                console.print(f"  [dim]metrics →[/dim] [bold]{metrics}[/bold]")
            if report:
                from aphex.report import render_report
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


def _ensure_type[T](value: object, expected: type[T], name: str) -> T:
    """Internal invariant check that raises TypeError and narrows the type."""
    if not isinstance(value, expected):
        raise TypeError(
            f"{name} must be {expected.__name__}, got {type(value).__name__}"
        )
    return value


def _gate_jobs(jobs: int, candidates: list, json_mode: bool) -> int:
    """Downgrade --jobs to 1 when any candidate targets a single accelerator.

    Running two benchmarks concurrently on one GPU / MPS device contends for
    compute and VRAM, distorting latency and memory measurements. We force
    serial execution in that case and tell the user once.
    """
    if jobs <= 1:
        return jobs
    accelerator_devices = {
        getattr(c, "device", "cpu") for c in candidates
    } - {"cpu", None, ""}
    if not accelerator_devices:
        return jobs
    if not json_mode:
        err_console.print(
            f"[yellow]--jobs={jobs} requested but candidates target "
            f"{sorted(accelerator_devices)}; downgrading to --jobs=1 so latency "
            "and memory measurements stay accurate. Re-run with --jobs=1 "
            "explicitly to silence this warning.[/yellow]"
        )
    return 1


def _parse_input_spec(s: str) -> Any:
    """Parse --input-shape into an InputSpec (single or multi-input).

    Single: '3,224,224'. Multi: 'input_ids:128:long;attention_mask:128:long'.
    """
    from aphex.inputspec import InputSpec

    try:
        spec = InputSpec.parse(s)
    except ValueError as exc:
        err_console.print(f"Invalid --input-shape {s!r}: {exc}")
        raise typer.Exit(code=1)
    for ts in spec.tensors:
        if not ts.shape or any(d <= 0 for d in ts.shape):
            err_console.print(
                f"Invalid --input-shape {s!r}: all dimensions must be positive integers."
            )
            raise typer.Exit(code=1)
    return spec


def _parse_shape(s: str) -> list[int]:
    """Parse a comma-separated input shape, e.g. '3,224,224' → [3,224,224]."""
    try:
        parts = [int(x) for x in s.split(",")]
    except ValueError:
        err_console.print(
            f"Invalid --input-shape {s!r}: expected comma-separated positive integers "
            "like '3,224,224'."
        )
        raise typer.Exit(code=1)
    if not parts or any(p <= 0 for p in parts):
        err_console.print(
            f"Invalid --input-shape {s!r}: all dimensions must be positive integers."
        )
        raise typer.Exit(code=1)
    return parts


def _parse_batch_sizes(s: str) -> list[int]:
    try:
        parts = [int(x) for x in s.split(",")]
    except ValueError:
        err_console.print(f"Invalid --batch-sizes {s!r}: expected comma-separated integers.")
        raise typer.Exit(code=1)
    if not parts or any(p <= 0 for p in parts):
        err_console.print(f"Invalid --batch-sizes {s!r}: all batch sizes must be positive.")
        raise typer.Exit(code=1)
    return parts


def _resolve_calibration(
    path: Path | None, shape: list[int], eval_dataset: object
) -> list | None:
    """Return calibration inputs, falling back to eval inputs with a clear warning."""
    calib = _load_calibration(path, shape)
    if calib is not None:
        return calib
    if eval_dataset is None:
        return None
    eval_inputs = getattr(eval_dataset, "inputs", None)
    if not eval_inputs:
        return None
    err_console.print(
        "[yellow]Note: no --calibration-data supplied; using eval inputs for int8 "
        "calibration. This can leak the eval set into quantization (the calibrated "
        "model is then scored on the same data). Pass --calibration-data for a clean "
        "split.[/yellow]"
    )
    return eval_inputs[:_CALIB_MAX_SAMPLES]


def _load_calibration(path: Path | None, input_shape: list[int]) -> list | None:
    if path is None:
        return None
    if path.is_dir():
        return _load_calibration_dir(path, input_shape)
    import torch

    from aphex.evaluator import _torch_load
    raw = _torch_load(path)
    if isinstance(raw, torch.Tensor):
        return [raw[i : i + 1] for i in range(min(raw.size(0), _CALIB_MAX_SAMPLES))]
    if isinstance(raw, list):
        return raw[:_CALIB_MAX_SAMPLES]
    return None


def _load_calibration_dir(path: Path, input_shape: list[int]) -> list | None:
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


def _load_eval(
    path: str | None,
    info: object,
    label_col: str,
    metric_override: str | None = None,
    required: bool = False,
) -> EvalDataset | None:
    if path is None:
        return None
    from aphex.evaluator import load_eval_data
    from aphex.inspector import ModelInfo
    if not isinstance(info, ModelInfo):
        raise TypeError(f"_load_eval expected ModelInfo, got {type(info).__name__}")
    try:
        return load_eval_data(path, info, label_col=label_col, metric_override=metric_override)
    except ValueError as exc:
        err_console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=1)
    except Exception as exc:
        if required:
            err_console.print(f"[red]Error loading eval data: {exc}[/red]")
            raise typer.Exit(code=1)
        err_console.print(f"[yellow]Warning: could not load eval data: {exc}[/yellow]")
        return None


def _load_infer_fn(spec: str | None) -> Callable[..., Any] | None:
    if spec is None:
        return None
    from aphex.evaluator import load_infer_fn
    return load_infer_fn(spec)


def _run_eval(
    eval_dataset: EvalDataset,
    infer_fn: Callable[..., Any] | None,
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
    from aphex.evaluator import EvalDataset
    if eval_dataset is None:
        return None
    if not isinstance(eval_dataset, EvalDataset):
        raise TypeError(
            f"_run_eval expected EvalDataset, got {type(eval_dataset).__name__}"
        )

    if infer_fn is not None:
        from aphex.evaluator import evaluate_baseline
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
        from aphex.evaluator import fill_accuracy_drop
        from aphex.inspector import ModelInfo
        try:
            if not json_mode:
                console.print(
                    f"  [dim]measuring accuracy drop ({eval_dataset.metric}) "
                    f"on {len(eval_dataset.inputs)} samples...[/dim]"
                )
            fill_accuracy_drop(results, model, _ensure_type(info, ModelInfo, "info"), eval_dataset)
        except Exception as exc:
            err_console.print(f"[yellow]Warning: accuracy drop measurement failed: {exc}[/yellow]")

    return None


def _select_with_fingerprint(
    candidates: list,
    info: object,
    hw: object,
    json_mode: bool,
) -> list:
    from aphex.inspector import ModelInfo
    from aphex.profiler import HardwareProfile
    from aphex.selector import select_candidates

    _info = _ensure_type(info, ModelInfo, "info")
    _hw = _ensure_type(hw, HardwareProfile, "hw")

    selected, rationale = select_candidates(candidates, _info, _hw, k=4)
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
    from aphex.cost_model import prune_candidates
    from aphex.inspector import ModelInfo
    from aphex.profiler import HardwareProfile

    _info = _ensure_type(info, ModelInfo, "info")
    _hw = _ensure_type(hw, HardwareProfile, "hw")

    kept, estimates = prune_candidates(candidates, _info, _hw, batch_size=batch_size)
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
    jobs: int = 1,
    input_spec: object = None,
) -> list:
    from aphex.candidates import DeploymentCandidate
    from aphex.inspector import ModelInfo
    from aphex.plugin import ModelPlugin
    if not isinstance(plugin, ModelPlugin):
        raise TypeError(f"plugin must be ModelPlugin, got {type(plugin).__name__}")
    _plugin = plugin
    _model_info = _ensure_type(model_info, ModelInfo, "model_info")

    jobs = _gate_jobs(jobs, candidates, json_mode)

    pairs: list[tuple[object, int]] = [(c, bs) for c in candidates for bs in batch_sizes]
    total = len(pairs)

    def _bench(pair: tuple[object, int]) -> object:
        cand, bs = pair
        return _plugin.benchmark(
            _ensure_type(cand, DeploymentCandidate, "cand"),
            model, _model_info, shape, bs, warmup, iters,
            timeout_s, calibration_inputs, input_spec,
        )

    if json_mode:
        return _run_pairs_parallel(pairs, _bench, jobs) if jobs > 1 else [_bench(p) for p in pairs]

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
        task_id = progress.add_task(
            f"racing {len(candidates)} backends × {len(batch_sizes)} batch sizes"
            + (f"  [dim](jobs={jobs})[/dim]" if jobs > 1 else ""),
            total=total,
        )

        def _print_result(cand: object, bs: int, r: Any) -> None:
            if r.ok:
                progress.console.print(
                    f"  [green]✓[/green]  {cand.description:<44}"  # type: ignore[attr-defined]
                    f"  [dim]bs={bs:<3}[/dim]"
                    f"  [bold]{r.latency_p50_ms:>8.2f} ms[/bold]"
                    f"  [dim]{r.throughput_rps:>7.0f} req/s[/dim]"
                )
            else:
                progress.console.print(
                    f"  [red]✗[/red]  [dim]{cand.description:<44}  bs={bs}[/dim]"  # type: ignore[attr-defined]
                    f"  [red]{(r.error or '')[:55]}[/red]"
                )

        if jobs <= 1:
            results = []
            for cand, bs in pairs:
                progress.update(task_id, description=f"{cand.description}  bs={bs}")  # type: ignore[attr-defined]
                r = _bench((cand, bs))
                results.append(r)
                _print_result(cand, bs, r)
                progress.advance(task_id)
        else:
            results = []
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futures = {ex.submit(_bench, p): p for p in pairs}
                for fut in as_completed(futures):
                    cand, bs = futures[fut]
                    r = fut.result()
                    results.append(r)
                    _print_result(cand, bs, r)
                    progress.advance(task_id)

    console.print()
    return results


def _run_pairs_parallel(
    pairs: list[tuple[object, int]],
    bench_fn: Callable[[tuple[object, int]], object],
    jobs: int,
) -> list:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(bench_fn, pairs))


# ---------------------------------------------------------------------------
# JSON output helpers
# ---------------------------------------------------------------------------


def _result_to_dict(r: object) -> dict:
    from aphex.benchmark import BenchmarkResult
    _r = _ensure_type(r, BenchmarkResult, "result")
    return {
        "backend": _r.candidate.backend,
        "dtype": _r.candidate.dtype,
        "device": _r.candidate.device,
        "description": _r.candidate.description,
        "batch_size": _r.batch_size,
        "ok": _r.ok,
        "latency_p50_ms": _r.latency_p50_ms if _r.ok else None,
        "latency_p95_ms": _r.latency_p95_ms if _r.ok else None,
        "latency_p99_ms": _r.latency_p99_ms if _r.ok else None,
        "latency_std_ms": _r.latency_std_ms if _r.ok else None,
        "latency_cv": _r.latency_cv if _r.ok else None,
        "stability": _r.stability if _r.ok else None,
        "throughput_rps": _r.throughput_rps if _r.ok else None,
        "memory_mb": _r.memory_mb if _r.ok else None,
        "accuracy_drop": _r.accuracy_drop,
        "error": _r.error,
    }


def _build_metrics_payload(results: list, recommendation: object = None, hw: object = None) -> dict:
    payload: dict = {"results": [_result_to_dict(r) for r in results]}
    if recommendation is not None:
        from aphex.recommender import Recommendation
        _rec = _ensure_type(recommendation, Recommendation, "recommendation")
        r = _rec.result
        payload["recommendation"] = {
            **_result_to_dict(r),
            "rationale": _rec.rationale,
            "notes": _rec.notes or [],
            "pareto_frontier_count": len(_rec.pareto_frontier),
        }
    if hw is not None:
        from aphex.profiler import HardwareProfile, collect_environment
        _hw = _ensure_type(hw, HardwareProfile, "hw")
        payload["environment"] = collect_environment(_hw)
    return payload


def _emit_json(results: list, recommendation: object = None, hw: object = None) -> None:
    print(json.dumps(_build_metrics_payload(results, recommendation, hw=hw), indent=2))


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
    from aphex.profiler import HardwareProfile

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
    from aphex.inspector import ModelInfo

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
    from aphex.preflight import PreflightResult

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
    from aphex.benchmark import BenchmarkResult

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
    from aphex.checker import CheckResult

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
    from aphex.recommender import Recommendation

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

    notes_block = ""
    if rec.notes:
        lines = "\n".join(f"  ⚠ {note}" for note in rec.notes)
        notes_block = f"\n\n[yellow]{lines}[/yellow]"

    body = (
        f"[bold white]{r.candidate.description}[/bold white]\n\n"
        f"{stats}\n"
        f"[dim]{rec.rationale}[/dim]"
        f"{frontier_note}"
        f"{notes_block}"
    )

    console.print()
    console.print(Rule("[bold green]recommendation[/bold green]", style="green"))
    console.print(Panel(body, border_style="green", padding=(1, 2)))
    console.print()


if __name__ == "__main__":
    app()
