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

_WARMUP_ITERS = 10
_MEASURE_ITERS = 100
_QUANT_BACKENDS = {"pytorch_int8_dynamic", "onnx_int8_cpu"}
# Backends that require an eager nn.Module (not a ScriptModule): compile needs the
# graph to be traceable by dynamo, and quantize_dynamic rewrites Linear submodules.
_EAGER_BACKENDS = {"torch_compile_fp32", "pytorch_int8_dynamic"}


def _ensure_quantization_engine() -> None:
    """Set the torch quantization engine if it hasn't been configured.

    macOS ARM defaults to 'none' (NoQEngine) which crashes quantize_dynamic.
    QNNPACK works on ARM; fbgemm works on x86.
    """
    import platform

    if torch.backends.quantized.engine in ("none", ""):
        if platform.machine() in ("arm64", "aarch64"):
            torch.backends.quantized.engine = "qnnpack"
        else:
            torch.backends.quantized.engine = "fbgemm"


class _CompatDynamicLinear(torch.ao.nn.quantized.dynamic.Linear):
    """DynamicQuantizedLinear with .weight and .bias as dequantized-tensor properties.

    Some architectures (e.g. Swin-T shifted-window attention) bypass nn.Linear.forward
    and call F.linear(x, submodule.weight, submodule.bias) directly. The base class
    exposes both as bound methods (not tensors) and the packed weight is qint8, neither
    of which F.linear accepts. Returning a dequantized float tensor lets those call
    sites work. Modules called through forward() still use the INT8 kernel path.
    """

    @property  # type: ignore[override]
    def weight(self) -> torch.Tensor:
        return torch.dequantize(self._packed_params._weight_bias()[0])

    @property  # type: ignore[override]
    def bias(self) -> torch.Tensor | None:
        return self._packed_params._weight_bias()[1]


def _safe_quantize_dynamic(model: nn.Module) -> nn.Module:
    """Quantize all nn.Linear modules, then patch DynamicQuantizedLinear.weight
    to be a property so models that access .weight directly (e.g. Swin-T) still work.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qm = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    # Change class in-place on every DynamicQuantizedLinear instance so .weight
    # returns a Tensor rather than a bound method.
    base_cls = torch.ao.nn.quantized.dynamic.Linear
    for mod in qm.modules():
        if type(mod) is base_cls:
            mod.__class__ = _CompatDynamicLinear

    return qm


@dataclass
class BenchmarkResult:
    candidate: DeploymentCandidate
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_rps: float  # requests/sec (batch_size items / p50 latency)
    memory_mb: float
    batch_size: int = 1
    error: str | None = None
    accuracy_drop: float | None = None  # cosine-similarity drop vs FP32 baseline; None if not measured

    @property
    def ok(self) -> bool:
        return self.error is None


def _serialize_model(
    model: nn.Module,
    input_shape: list[int] | None = None,
    candidate: Any = None,
) -> tuple[Any, ...]:
    """Serialize a model to bytes for cross-process transfer.

    Returns a tuple whose first element is a kind tag:
      "script"      — torch.jit ScriptModule; load with torch.jit.load
      "module"      — regular nn.Module; load with torch.load
      "stub_module" — nn.Module built from stub classes; includes stub specs
                      as a third element so workers can recreate them before
                      unpickling (avoids ModuleNotFoundError in spawned procs)

    For backends that require an eager nn.Module (torch.compile, quantize_dynamic)
    we cannot trace to a ScriptModule, so stub models go via "stub_module".
    All other stub models are traced to a self-contained ScriptModule so workers
    don't need the original class definitions.
    """
    import io
    from infermap.inspector import _STUB_REGISTRY

    if isinstance(model, torch.jit.ScriptModule):
        buf = io.BytesIO()
        torch.jit.save(model, buf)
        return ("script", buf.getvalue())

    needs_eager = (
        candidate is not None
        and hasattr(candidate, "backend")
        and candidate.backend in _EAGER_BACKENDS
    )

    # Stub models sent to eager backends need the stub specs shipped alongside
    # so worker processes can recreate the missing external classes.
    if _STUB_REGISTRY and needs_eager:
        buf = io.BytesIO()
        torch.save(model, buf)
        return ("stub_module", buf.getvalue(), list(_STUB_REGISTRY))

    if input_shape is not None:
        try:
            dummy = torch.zeros(1, *input_shape)
            with torch.no_grad():
                # check_trace=False: nn.MultiheadAttention and similar modules
                # produce different JIT IR variable names across invocations
                # (the sanity check re-runs the forward and diffs the graphs).
                # The computation is identical; only internal numbering varies.
                traced = torch.jit.trace(model, dummy, strict=False, check_trace=False)
            buf = io.BytesIO()
            torch.jit.save(traced, buf)
            return ("script", buf.getvalue())
        except Exception:
            pass  # fall through to regular pickle path

    buf = io.BytesIO()
    torch.save(model, buf)
    return ("module", buf.getvalue())


def _recreate_stubs(specs: list[tuple[str, str, bool]]) -> None:
    """Recreate stub classes in sys.modules so torch.load can unpickle stub models."""
    import sys
    import types
    import torch.nn as nn

    for mod_name, cls_name, is_nn_mod in specs:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
        stub_mod = sys.modules[mod_name]
        if not hasattr(stub_mod, cls_name):
            if is_nn_mod:
                cls = type(cls_name, (nn.Module,), {
                    "__init__": lambda self: nn.Module.__init__(self),
                    "__module__": mod_name,
                })
            else:
                cls = type(cls_name, (), {"__module__": mod_name})
            setattr(stub_mod, cls_name, cls)


def _deserialize_model(model_payload: tuple[Any, ...]) -> nn.Module:
    import io
    from infermap.inspector import _patch_stub_forwards

    kind = model_payload[0]
    data = model_payload[1]
    buf = io.BytesIO(data)

    if kind == "script":
        return torch.jit.load(buf)

    if kind == "stub_module":
        _recreate_stubs(model_payload[2])
        model = torch.load(buf, weights_only=False)
        _patch_stub_forwards(model)
        return model

    return torch.load(buf, weights_only=False)


def _worker(
    queue: Any,
    candidate: DeploymentCandidate,
    model_payload: tuple[str, bytes],
    model_info: ModelInfo,
    input_shape: list[int],
    batch_size: int,
    warmup_iters: int,
    measure_iters: int,
    calibration_inputs: list[Any] | None,
) -> None:
    try:
        model = _deserialize_model(model_payload)
        result = _run_benchmark(
            candidate, model, model_info, input_shape, batch_size, warmup_iters, measure_iters, calibration_inputs
        )
    except Exception as exc:
        result = BenchmarkResult(
            candidate=candidate,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            throughput_rps=0.0,
            memory_mb=0.0,
            batch_size=batch_size,
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
    timeout_s: float | None = 180.0,
    calibration_inputs: list[Any] | None = None,
) -> BenchmarkResult:
    import multiprocessing as mp

    if timeout_s is None:
        return _worker_inline(
            candidate, model, model_info, input_shape, batch_size, warmup_iters, measure_iters, calibration_inputs
        )

    model_payload = _serialize_model(model, input_shape, candidate)

    ctx = mp.get_context("spawn")
    queue: mp.Queue[BenchmarkResult] = ctx.Queue()
    proc = ctx.Process(
        target=_worker,
        args=(
            queue, candidate, model_payload, model_info, input_shape,
            batch_size, warmup_iters, measure_iters, calibration_inputs,
        ),
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
            batch_size=batch_size,
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
        batch_size=batch_size,
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
    calibration_inputs: list[Any] | None = None,
) -> BenchmarkResult:
    try:
        return _run_benchmark(
            candidate, model, model_info, input_shape, batch_size, warmup_iters, measure_iters, calibration_inputs
        )
    except Exception as exc:
        return BenchmarkResult(
            candidate=candidate,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            throughput_rps=0.0,
            memory_mb=0.0,
            batch_size=batch_size,
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
    calibration_inputs: list[Any] | None = None,
) -> BenchmarkResult:
    device = torch.device(candidate.device)

    accuracy_drop: float | None = None
    if calibration_inputs and candidate.backend in _QUANT_BACKENDS:
        accuracy_drop = _measure_accuracy_drop(candidate, model, calibration_inputs)

    prepared_model, dummy_input, weight_mb = _prepare(candidate, model, input_shape, batch_size, device)

    timings_ms = _time_model(prepared_model, dummy_input, device, warmup_iters, measure_iters)
    memory_mb = _measure_memory(prepared_model, dummy_input, device, weight_mb)

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
        batch_size=batch_size,
        accuracy_drop=accuracy_drop,
    )


def _export_to_onnx_bytes(model: Any, dummy: torch.Tensor) -> bytes:
    """Export a model to ONNX and return the raw bytes."""
    import io
    import warnings

    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy.to("cpu"),
            buf,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,
        )
    return buf.getvalue()


def _prepare(
    candidate: DeploymentCandidate,
    model: nn.Module,
    input_shape: list[int],
    batch_size: int,
    device: torch.device,
) -> tuple[Any, torch.Tensor, float]:
    import copy
    import warnings

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(candidate.dtype, torch.float32)

    # TorchScript traced models have non-leaf parameter tensors whose .grad access
    # triggers a PyTorch UserWarning during deepcopy/to() — suppress across all
    # model-mutation calls so subprocess output stays clean.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The .grad attribute of a Tensor that is not a leaf")
        m = copy.deepcopy(model).to(device)
        m.eval()
        if candidate.dtype in ("fp16", "bf16"):
            m = m.to(torch_dtype)

    if candidate.backend == "torch_compile_fp32":
        if isinstance(m, torch.jit.ScriptModule):
            raise RuntimeError(
                "torch.compile is not compatible with TorchScript models. "
                "Save an eager nn.Module instead of a scripted one."
            )
        m = torch.compile(m)

    dummy = torch.randn(batch_size, *input_shape, dtype=torch_dtype, device=device)

    # Compute weight memory before quantization/ONNX conversion (quantized tensors
    # and ONNX sessions don't expose parameters in the same way).
    weight_mb = (
        sum(p.numel() * p.element_size() for p in m.parameters())
        + sum(b.numel() * b.element_size() for b in m.buffers())
    ) / 1e6

    if candidate.backend == "pytorch_int8_dynamic":
        if isinstance(m, torch.jit.ScriptModule):
            raise RuntimeError(
                "Dynamic quantization is not compatible with TorchScript models. "
                "Save an eager nn.Module instead of a scripted one."
            )
        _ensure_quantization_engine()
        m = _safe_quantize_dynamic(m)
        weight_mb = weight_mb / 4  # linear weights stored as INT8, ~4x smaller than FP32
    elif candidate.backend in ("onnx_cpu", "onnx_cuda", "onnx_coreml"):
        m, dummy = _prepare_onnx(candidate, m, dummy, device)
    elif candidate.backend == "onnx_int8_cpu":
        m, dummy = _prepare_onnx_int8(m, dummy)
        weight_mb = weight_mb / 4

    return m, dummy, weight_mb


def _prepare_onnx(
    candidate: DeploymentCandidate,
    model: Any,
    dummy: torch.Tensor,
    device: torch.device,
) -> tuple[Any, torch.Tensor]:
    import onnxruntime as ort

    onnx_bytes = _export_to_onnx_bytes(model, dummy)

    providers: list[str]
    if candidate.backend == "onnx_cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif candidate.backend == "onnx_coreml":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_bytes, providers=providers)
    return sess, dummy.to("cpu").float()


def _prepare_onnx_int8(model: Any, dummy: torch.Tensor) -> tuple[Any, torch.Tensor]:
    import logging
    import tempfile
    import warnings
    from pathlib import Path as _Path

    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_bytes = _export_to_onnx_bytes(model, dummy)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _Path(tmpdir) / "model.onnx"
        quant_path = _Path(tmpdir) / "model_int8.onnx"
        onnx_path.write_bytes(onnx_bytes)

        # Suppress onnxruntime.quantization's root-logger advisory and ORT session logs.
        root_logger = logging.getLogger()
        ort_logger = logging.getLogger("onnxruntime")
        prev_root, prev_ort = root_logger.level, ort_logger.level
        root_logger.setLevel(logging.ERROR)
        ort_logger.setLevel(logging.ERROR)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                quantize_dynamic(str(onnx_path), str(quant_path), weight_type=QuantType.QInt8)
        finally:
            root_logger.setLevel(prev_root)
            ort_logger.setLevel(prev_ort)

        # InferenceSession loads model into memory, so tmpdir can be deleted after.
        sess = ort.InferenceSession(str(quant_path), providers=["CPUExecutionProvider"])

    return sess, dummy.to("cpu").float()


def _measure_accuracy_drop(
    candidate: DeploymentCandidate,
    model: nn.Module,
    calibration_inputs: list[Any],
) -> float | None:
    """Return mean cosine-similarity drop between FP32 and quantized outputs.

    0.0 = outputs identical, 1.0 = outputs orthogonal. Returns None on any failure.
    """
    import copy
    import warnings

    import torch.nn.functional as F

    try:
        fp32_model = copy.deepcopy(model).cpu().eval()
        fp32_outs: list[torch.Tensor] = []
        with torch.no_grad():
            for inp in calibration_inputs:
                out = fp32_model(inp.float().cpu())
                fp32_outs.append(out.detach().flatten().float())

        quant_outs: list[torch.Tensor] = []

        if candidate.backend == "pytorch_int8_dynamic":
            _ensure_quantization_engine()
            quant_model = _safe_quantize_dynamic(copy.deepcopy(model).cpu().eval())
            with torch.no_grad():
                for inp in calibration_inputs:
                    out = quant_model(inp.float().cpu())
                    quant_outs.append(out.detach().flatten().float())

        elif candidate.backend == "onnx_int8_cpu":
            import logging
            import tempfile
            from pathlib import Path as _Path

            import onnxruntime as ort
            from onnxruntime.quantization import QuantType, quantize_dynamic

            dummy = calibration_inputs[0].float().cpu()
            onnx_bytes = _export_to_onnx_bytes(copy.deepcopy(model).cpu().eval(), dummy)

            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = _Path(tmpdir) / "model.onnx"
                quant_path = _Path(tmpdir) / "model_int8.onnx"
                onnx_path.write_bytes(onnx_bytes)

                root_logger = logging.getLogger()
                ort_logger = logging.getLogger("onnxruntime")
                prev_root, prev_ort = root_logger.level, ort_logger.level
                root_logger.setLevel(logging.ERROR)
                ort_logger.setLevel(logging.ERROR)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        quantize_dynamic(str(onnx_path), str(quant_path), weight_type=QuantType.QInt8)
                finally:
                    root_logger.setLevel(prev_root)
                    ort_logger.setLevel(prev_ort)

                sess = ort.InferenceSession(str(quant_path), providers=["CPUExecutionProvider"])

            for inp in calibration_inputs:
                out = sess.run(None, {"input": inp.float().cpu().numpy()})[0]
                quant_outs.append(torch.tensor(out).flatten().float())

        if not quant_outs:
            return None

        drops = []
        for fp32_out, quant_out in zip(fp32_outs, quant_outs):
            if fp32_out.numel() == 0:
                continue
            sim = F.cosine_similarity(fp32_out.unsqueeze(0), quant_out.unsqueeze(0)).item()
            drops.append(1.0 - max(-1.0, min(1.0, sim)))

        return sum(drops) / len(drops) if drops else None

    except Exception:
        return None


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


def _measure_memory(
    model: Any, dummy: torch.Tensor, device: torch.device, weight_mb: float
) -> float:
    """Return model memory in MB (weights + peak activations where measurable)."""
    import onnxruntime as ort

    if device.type == "cuda":
        # CUDA tracks peak allocation precisely — includes weights + activations.
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(dummy)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / 1e6

    if device.type == "mps":
        # MPS: weight bytes are exact; add allocation delta for activations.
        torch.mps.synchronize()  # type: ignore[attr-defined]
        before = torch.mps.current_allocated_memory()  # type: ignore[attr-defined]
        with torch.no_grad():
            model(dummy)
        torch.mps.synchronize()  # type: ignore[attr-defined]
        after = torch.mps.current_allocated_memory()  # type: ignore[attr-defined]
        activation_mb = max(0.0, (after - before) / 1e6)
        return weight_mb + activation_mb

    # CPU / ONNX: weight bytes are exact; activations are small for batch_size=1.
    if isinstance(model, ort.InferenceSession):
        model.run(None, {"input": dummy.numpy()})
    else:
        with torch.no_grad():
            model(dummy)
    return weight_mb
