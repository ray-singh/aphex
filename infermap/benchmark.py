"""Benchmark engine — measures latency, throughput, and memory per deployment candidate."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from infermap.candidates import DeploymentCandidate
from infermap.inspector import ModelInfo

# Warm-up iterations before timing starts
_WARMUP_ITERS = 10
# Measurement iterations
_MEASURE_ITERS = 100


@dataclass
class BenchmarkResult:
    candidate: DeploymentCandidate
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_rps: float  # requests/sec at batch_size=1
    memory_mb: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _worker(
    queue: Any,
    candidate: DeploymentCandidate,
    model: nn.Module,
    model_info: ModelInfo,
    input_shape: list[int],
    batch_size: int,
    warmup_iters: int,
    measure_iters: int,
) -> None:
    try:
        result = _run_benchmark(
            candidate, model, model_info, input_shape, batch_size, warmup_iters, measure_iters
        )
    except Exception as exc:
        result = BenchmarkResult(
            candidate=candidate,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            throughput_rps=0.0,
            memory_mb=0.0,
            error=str(exc),
        )
    queue.put(result)


def benchmark_candidate(
    candidate: DeploymentCandidate,
    model: nn.Module,
    model_info: ModelInfo,
    input_shape: list[int],
    batch_size: int = 1,
    warmup_iters: int = _WARMUP_ITERS,
    measure_iters: int = _MEASURE_ITERS,
    timeout_s: float | None = 60.0,
) -> BenchmarkResult:
    import multiprocessing as mp

    if timeout_s is None:
        return _worker_inline(candidate, model, model_info, input_shape, batch_size, warmup_iters, measure_iters)

    ctx = mp.get_context("spawn")
    queue: mp.Queue[BenchmarkResult] = ctx.Queue()
    proc = ctx.Process(
        target=_worker,
        args=(queue, candidate, model, model_info, input_shape, batch_size, warmup_iters, measure_iters),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
        return BenchmarkResult(
            candidate=candidate,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            throughput_rps=0.0,
            memory_mb=0.0,
            error=f"timed out after {timeout_s:.0f}s",
        )

    if not queue.empty():
        return queue.get_nowait()

    return BenchmarkResult(
        candidate=candidate,
        latency_p50_ms=0.0,
        latency_p95_ms=0.0,
        latency_p99_ms=0.0,
        throughput_rps=0.0,
        memory_mb=0.0,
        error="subprocess exited without result",
    )


def _worker_inline(
    candidate: DeploymentCandidate,
    model: nn.Module,
    model_info: ModelInfo,
    input_shape: list[int],
    batch_size: int,
    warmup_iters: int,
    measure_iters: int,
) -> BenchmarkResult:
    try:
        return _run_benchmark(
            candidate, model, model_info, input_shape, batch_size, warmup_iters, measure_iters
        )
    except Exception as exc:
        return BenchmarkResult(
            candidate=candidate,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            throughput_rps=0.0,
            memory_mb=0.0,
            error=str(exc),
        )


def _run_benchmark(
    candidate: DeploymentCandidate,
    model: nn.Module,
    model_info: ModelInfo,
    input_shape: list[int],
    batch_size: int,
    warmup_iters: int,
    measure_iters: int,
) -> BenchmarkResult:
    device = torch.device(candidate.device)
    prepared_model, dummy_input = _prepare(candidate, model, input_shape, batch_size, device)

    timings_ms = _time_model(prepared_model, dummy_input, device, warmup_iters, measure_iters)
    memory_mb = _measure_memory(prepared_model, dummy_input, device)

    import statistics

    timings_ms_sorted = sorted(timings_ms)
    n = len(timings_ms_sorted)
    p50 = timings_ms_sorted[int(n * 0.50)]
    p95 = timings_ms_sorted[int(n * 0.95)]
    p99 = timings_ms_sorted[int(n * 0.99)]
    throughput = 1000.0 / p50 * batch_size  # req/sec

    return BenchmarkResult(
        candidate=candidate,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        throughput_rps=throughput,
        memory_mb=memory_mb,
    )


def _prepare(
    candidate: DeploymentCandidate,
    model: nn.Module,
    input_shape: list[int],
    batch_size: int,
    device: torch.device,
) -> tuple[Any, torch.Tensor]:
    import copy

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Deep-copy so each candidate gets independent weights (dtype casts are in-place).
    m = copy.deepcopy(model).to(device)
    m.eval()

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(candidate.dtype, torch.float32)

    if candidate.dtype in ("fp16", "bf16"):
        m = m.to(torch_dtype)

    if candidate.backend == "torch_compile_fp32":
        m = torch.compile(m)

    dummy = torch.randn(batch_size, *input_shape, dtype=torch_dtype, device=device)

    if candidate.backend in ("onnx_cpu", "onnx_cuda", "onnx_coreml"):
        m, dummy = _prepare_onnx(candidate, m, dummy, device)

    return m, dummy


def _prepare_onnx(
    candidate: DeploymentCandidate,
    model: Any,
    dummy: torch.Tensor,
    device: torch.device,
) -> tuple[Any, torch.Tensor]:
    import io
    import onnxruntime as ort

    buffer = io.BytesIO()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            model,
            dummy.to("cpu"),
            buffer,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,
        )
    buffer.seek(0)

    providers: list[str]
    if candidate.backend == "onnx_cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif candidate.backend == "onnx_coreml":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(buffer.read(), providers=providers)
    # Return session as "model" and move dummy to CPU for ONNX inference
    return sess, dummy.to("cpu").float()


def _time_model(
    model: Any,
    dummy: torch.Tensor,
    device: torch.device,
    warmup_iters: int,
    measure_iters: int,
) -> list[float]:
    import onnxruntime as ort

    is_onnx = isinstance(model, ort.InferenceSession)

    def run() -> None:
        if is_onnx:
            model.run(None, {"input": dummy.numpy()})
        else:
            with torch.no_grad():
                model(dummy)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()  # type: ignore[attr-defined]

    # Warm-up
    for _ in range(warmup_iters):
        run()
    sync()

    timings_ms: list[float] = []

    if device.type == "cuda" and not is_onnx:
        # Use CUDA events for higher-precision GPU timing
        for _ in range(measure_iters):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            run()
            end_evt.record()
            torch.cuda.synchronize()
            timings_ms.append(start_evt.elapsed_time(end_evt))
    else:
        for _ in range(measure_iters):
            t0 = time.perf_counter()
            run()
            sync()
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

    return timings_ms


def _measure_memory(model: Any, dummy: torch.Tensor, device: torch.device) -> float:
    """Return peak memory usage in MB for one forward pass."""
    import onnxruntime as ort

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(dummy)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / 1e6

    if device.type == "mps":
        # MPS does not expose peak memory APIs; use psutil RAM delta as a proxy
        import psutil

        proc = psutil.Process()
        before = proc.memory_info().rss
        with torch.no_grad():
            model(dummy)
        torch.mps.synchronize()  # type: ignore[attr-defined]
        after = proc.memory_info().rss
        return max(0.0, (after - before) / 1e6)

    # CPU / ONNX
    import psutil

    proc = psutil.Process()
    before = proc.memory_info().rss
    if isinstance(model, ort.InferenceSession):
        model.run(None, {"input": dummy.numpy()})
    else:
        with torch.no_grad():
            model(dummy)
    after = proc.memory_info().rss
    return max(0.0, (after - before) / 1e6)
