"""Benchmark engine — measures latency, throughput, and memory per deployment candidate."""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from infermap.candidates import DeploymentCandidate
from infermap.inspector import ModelInfo

if TYPE_CHECKING:
    import torch
    from torch import nn

logger = logging.getLogger("infermap.benchmark")

_WARMUP_ITERS = 10
_MEASURE_ITERS = 100
_ACCURACY_SENSITIVE_BACKENDS = {
    "pytorch_fp16",
    "pytorch_bf16",
    "pytorch_int8_dynamic",
    "onnx_int8_cpu",
    "tensorrt_fp16",
    "tensorrt_int8",
    "openvino_int8",
    "pytorch_prune_unstructured_30",
    "pytorch_prune_unstructured_50",
    "pytorch_prune_unstructured_70",
    "pytorch_prune_2_4",
}
# Backends that require an eager nn.Module (not a ScriptModule): compile needs the
# graph to be traceable by dynamo, and quantize_dynamic rewrites Linear submodules.
_EAGER_BACKENDS = {"torch_compile_fp32", "pytorch_int8_dynamic"}
# torch.compile spends its first-run budget on Dynamo graph compilation, which can
# take several minutes for complex models. Give compile backends a longer timeout
# so they aren't killed during compilation before they ever get to measure latency.
_COMPILE_BACKENDS = {"torch_compile_fp32", "torch_compile_fp16", "torch_compile_bf16"}
_COMPILE_TIMEOUT_S = 600.0
# Backends where precision is controlled by the external runtime, not by pre-converting
# the nn.Module. We export FP32 ONNX and let TRT/OV handle precision internally.
_TRT_OV_BACKENDS = frozenset({
    "tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8",
    "openvino_fp32", "openvino_int8",
})

# Families where a single-pass cosine similarity proxy is not a meaningful
# accuracy signal. Listed here (rather than imported from evaluator) to avoid
# importing the larger evaluator module from the benchmark hot path.
_GENERATIVE_FAMILIES = frozenset({"llm", "transformer_decoder", "seq2seq"})


def _is_generative_family(info: ModelInfo | None) -> bool:
    if info is None:
        return False
    return (info.family or "").lower() in _GENERATIVE_FAMILIES


def _is_pruning_backend(backend: str) -> bool:
    """Local check to avoid importing infermap.pruning at module-load time."""
    return backend.startswith("pytorch_prune_")


def _is_dp_backend(backend: str) -> bool:
    """True for ``pytorch_dp{N}_{dtype}`` candidates."""
    return backend.startswith("pytorch_dp")


def _dp_world_size(backend: str) -> int:
    """Parse N from ``pytorch_dp{N}_{dtype}``. Raises ValueError on malformed names."""
    rest = backend[len("pytorch_dp"):]
    digits = rest.split("_", 1)[0]
    if not digits.isdigit():
        raise ValueError(f"malformed DP backend: {backend!r}")
    n = int(digits)
    if n < 2:
        raise ValueError(f"DP world size must be >= 2 in {backend!r}")
    return n


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyTorch is required for this operation. "
            "Install it with: pip install 'aphex-ml[torch]'"
        ) from None


def _ensure_quantization_engine() -> None:
    """Set the torch quantization engine if it hasn't been configured.

    macOS ARM defaults to 'none' (NoQEngine) which crashes quantize_dynamic.
    QNNPACK works on ARM; fbgemm works on x86.
    """
    import platform

    import torch

    if torch.backends.quantized.engine in ("none", ""):
        if platform.machine() in ("arm64", "aarch64"):
            torch.backends.quantized.engine = "qnnpack"
        else:
            torch.backends.quantized.engine = "fbgemm"


def _safe_quantize_dynamic(model: Any) -> Any:
    """Quantize all nn.Linear modules.

    On PyTorch versions where DynamicQuantizedLinear lacks a usable ``.weight``
    accessor, we substitute a subclass that exposes ``.weight`` and ``.bias`` as
    dequantized-tensor properties so models that introspect ``.weight`` directly
    (e.g. Swin-T) still work. On versions that already provide a working
    accessor, the patch is skipped.
    """
    import warnings

    import torch
    import torch.nn as nn

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qm = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    base_cls = torch.ao.nn.quantized.dynamic.Linear

    if _dyn_quant_linear_has_weight(base_cls):
        return qm

    class _CompatDynamicLinear(base_cls):  # type: ignore[misc, valid-type]
        @property
        def weight(self) -> torch.Tensor:
            return torch.dequantize(self._packed_params._weight_bias()[0])

        @property
        def bias(self) -> torch.Tensor | None:
            return self._packed_params._weight_bias()[1]

    for mod in qm.modules():
        if type(mod) is base_cls:
            mod.__class__ = _CompatDynamicLinear

    return qm


def _dyn_quant_linear_has_weight(cls: Any) -> bool:
    """Whether the given DynamicQuantizedLinear class already exposes .weight()."""
    try:
        instance_attr = cls.__dict__.get("weight")
        if isinstance(instance_attr, property):
            return True
        # Some PyTorch versions provide weight() as a method.
        return callable(getattr(cls, "weight", None))
    except Exception:
        return False


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


class _TensorRTRunner:
    """Wraps a TensorRT execution context as a callable (tensor → tensor)."""

    def __init__(self, engine_bytes: bytes) -> None:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        self._context = self._engine.create_execution_context()
        self._use_v10_api = int(trt.__version__.split(".")[0]) >= 10

    def __call__(self, x: Any) -> Any:
        x_cuda = x.to("cuda").float().contiguous()
        return self._run_v10(x_cuda) if self._use_v10_api else self._run_v8(x_cuda)

    def _run_v10(self, x: Any) -> Any:
        import torch
        self._context.set_input_shape("input", tuple(x.shape))
        out_shape = tuple(self._context.get_tensor_shape("output"))
        out = torch.empty(out_shape, device="cuda", dtype=torch.float32)
        self._context.set_tensor_address("input", x.data_ptr())
        self._context.set_tensor_address("output", out.data_ptr())
        self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return out

    def _run_v8(self, x: Any) -> Any:
        import torch
        engine = self._engine
        in_idx = engine.get_binding_index("input")
        out_idx = engine.get_binding_index("output")
        self._context.set_binding_shape(in_idx, tuple(x.shape))
        out_shape = tuple(self._context.get_binding_shape(out_idx))
        out = torch.empty(out_shape, device="cuda", dtype=torch.float32)
        bindings: list[int | None] = [None] * engine.num_bindings
        bindings[in_idx] = x.data_ptr()
        bindings[out_idx] = out.data_ptr()
        self._context.execute_async_v2(
            bindings=bindings, stream_handle=torch.cuda.current_stream().cuda_stream
        )
        return out


class _OpenVINORunner:
    """Wraps an OpenVINO compiled model as a callable (tensor → tensor)."""

    def __init__(self, compiled_model: Any) -> None:
        self._infer_req = compiled_model.create_infer_request()

    def __call__(self, x: Any) -> Any:
        import torch
        results = self._infer_req.infer({"input": x.cpu().float().numpy()})
        return torch.from_numpy(next(iter(results.values())))


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

    import torch
    import torch.nn as nn

    from infermap.inspector import _STUB_REGISTRY

    # PyTorch 2.x fuses TransformerEncoderLayer into aten::_transformer_encoder_layer_fwd
    # in eval mode when enable_nested_tensor=True (the default). That fused op is not
    # implemented on MPS and can't be exported to ONNX opset 17, so any transformer
    # model fails those backends. Disabling it here forces the standard unfused path
    # (standard matmul + softmax) that all runtimes support. Correctness is unchanged.
    for m in model.modules():
        if isinstance(m, nn.TransformerEncoder):
            m.enable_nested_tensor = False

    if isinstance(model, torch.jit.ScriptModule):
        buf = io.BytesIO()
        torch.jit.save(model, buf)
        return ("script", buf.getvalue())

    needs_eager = (
        candidate is not None
        and hasattr(candidate, "backend")
        and (candidate.backend in _EAGER_BACKENDS or _is_pruning_backend(candidate.backend))
    )

    # Backends that require an eager nn.Module (torch.compile, dynamic quantization,
    # pruning) must not be traced to TorchScript. Pickle the module directly.
    # Stub models also need the stub-class specs so the worker can recreate them.
    if needs_eager:
        buf = io.BytesIO()
        torch.save(model, buf)
        if _STUB_REGISTRY:
            return ("stub_module", buf.getvalue(), list(_STUB_REGISTRY))
        return ("module", buf.getvalue())

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
        except Exception as exc:
            logger.debug("torch.jit.trace failed (%s); falling back to torch.save", exc)

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


def _deserialize_model(model_payload: tuple[Any, ...]) -> Any:
    import io

    import torch

    from infermap.inspector import _patch_stub_forwards

    kind = model_payload[0]
    data = model_payload[1]
    buf = io.BytesIO(data)

    if kind == "script":
        return torch.jit.load(buf)

    # weights_only=False is required here because the parent process pickled
    # an nn.Module (not a state_dict) into this buffer. The buffer originates
    # from our own _serialize_model, not from disk, so there's no untrusted
    # input — it travels parent → subprocess via mp.Queue.
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
    # Compile backends spend their first forward pass doing Dynamo graph compilation,
    # which can take several minutes. Use a longer deadline so they aren't killed
    # before compilation finishes and actual inference has a chance to run.
    effective_timeout = (
        max(timeout_s, _COMPILE_TIMEOUT_S)
        if candidate.backend in _COMPILE_BACKENDS
        else timeout_s
    )

    try:
        proc.start()
        proc.join(timeout=effective_timeout)

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
                error=f"timed out after {effective_timeout:.0f}s",
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
    finally:
        # Release the Queue's pipe FD + feeder thread. Without this, long sweeps
        # accumulate open FDs and can hit "Too many open files" on macOS.
        try:
            queue.close()
            queue.join_thread()
        except Exception:
            pass


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
    model: Any,
    model_info: ModelInfo,
    input_shape: list[int],
    batch_size: int,
    warmup_iters: int,
    measure_iters: int,
    calibration_inputs: list[Any] | None = None,
) -> BenchmarkResult:
    import torch
    device = torch.device(candidate.device)

    accuracy_drop: float | None = None
    if (
        calibration_inputs
        and candidate.backend in _ACCURACY_SENSITIVE_BACKENDS
        and not _is_generative_family(model_info)
    ):
        accuracy_drop = _measure_accuracy_drop(candidate, model, calibration_inputs)
    elif (
        calibration_inputs
        and candidate.backend in _ACCURACY_SENSITIVE_BACKENDS
        and _is_generative_family(model_info)
    ):
        logger.warning(
            "skipping cosine-similarity accuracy proxy for family=%r: "
            "single-forward cosine doesn't approximate generation quality. "
            "Use --infer-fn with a perplexity or task-metric callable.",
            getattr(model_info, "family", None),
        )

    prepared_model, dummy_input, weight_mb = _prepare(candidate, model, input_shape, batch_size, device, calibration_inputs)

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


def _export_to_onnx_bytes(model: Any, dummy: Any) -> bytes:
    """Export a model to ONNX and return the raw bytes."""
    import io
    import warnings

    import torch

    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy.to("cpu"),
            buf,  # type: ignore[arg-type]
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,
        )
    return buf.getvalue()


def _prepare(
    candidate: DeploymentCandidate,
    model: Any,
    input_shape: list[int],
    batch_size: int,
    device: Any,
    calibration_inputs: list[Any] | None = None,
) -> tuple[Any, Any, float]:
    import copy
    import warnings

    import torch

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
        # TRT/OV handle precision internally; keep the model FP32 for ONNX export.
        if candidate.dtype in ("fp16", "bf16") and candidate.backend not in _TRT_OV_BACKENDS:
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
    elif _is_dp_backend(candidate.backend):
        if device.type != "cuda":
            raise RuntimeError(
                f"{candidate.backend!r} requires a CUDA device, got {device.type!r}."
            )
        world = _dp_world_size(candidate.backend)
        available = torch.cuda.device_count()
        if available < world:
            raise RuntimeError(
                f"{candidate.backend!r} needs {world} CUDA devices, found {available}."
            )
        if batch_size < world:
            raise RuntimeError(
                f"{candidate.backend!r} requires batch_size >= {world} so DP can "
                f"shard the batch dim; got batch_size={batch_size}."
            )
        m = torch.nn.DataParallel(m, device_ids=list(range(world)))
    elif _is_pruning_backend(candidate.backend):
        if isinstance(m, torch.jit.ScriptModule):
            raise RuntimeError(
                "Pruning requires an eager nn.Module so layer weights can be "
                "modified in place. Save a non-scripted model and rerun."
            )
        from infermap.pruning import prune_model, spec_for_backend
        m, report = prune_model(m, spec_for_backend(candidate.backend))
        if report.weight_params_total > 0:
            # Approximate the post-pruning storage: zeros still take space in
            # dense tensors. (1 - sparsity) is the conservative compressed-form
            # estimate that a sparse storage format could reach.
            weight_mb = weight_mb * max(0.05, 1.0 - report.sparsity)
    elif candidate.backend in ("onnx_cpu", "onnx_cuda", "onnx_coreml"):
        m, dummy = _prepare_onnx(candidate, m, dummy, device)
    elif candidate.backend == "onnx_int8_cpu":
        m, dummy = _prepare_onnx_int8(m, dummy)
        weight_mb = weight_mb / 4
    elif candidate.backend in ("tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8"):
        m, dummy, weight_mb = _prepare_tensorrt(candidate, m, dummy, calibration_inputs)
    elif candidate.backend in ("openvino_fp32", "openvino_int8"):
        m, dummy, weight_mb = _prepare_openvino(candidate, m, dummy, calibration_inputs)

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


def _make_trt_calibrator(calibration_inputs: list[Any]) -> Any:
    """Return a TensorRT IInt8MinMaxCalibrator backed by the given tensors."""
    import tensorrt as trt

    class _Calibrator(trt.IInt8MinMaxCalibrator):
        def __init__(self) -> None:
            super().__init__()
            self._batches = [x.float().cuda().contiguous() for x in calibration_inputs]
            self._idx = 0

        def get_batch_size(self) -> int:
            return self._batches[0].shape[0] if self._batches else 1

        def get_batch(self, names: list[str]) -> list[int] | None:
            if self._idx >= len(self._batches):
                return None
            data = self._batches[self._idx]
            self._idx += 1
            return [data.data_ptr()]

        def read_calibration_cache(self) -> bytes | None:
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            pass

    return _Calibrator()


def _prepare_tensorrt(
    candidate: DeploymentCandidate,
    model: Any,
    dummy: torch.Tensor,
    calibration_inputs: list[Any] | None,
) -> tuple[_TensorRTRunner, torch.Tensor, float]:
    """Build a TensorRT engine from an nn.Module; return (runner, dummy, engine_mb)."""
    import tensorrt as trt

    if candidate.backend == "tensorrt_int8" and not calibration_inputs:
        raise RuntimeError(
            "tensorrt_int8 requires calibration data. "
            "Re-run with --calibration-data to supply representative inputs."
        )

    # Always export FP32 ONNX; TRT precision flags control internal compute type.
    onnx_bytes = _export_to_onnx_bytes(model.cpu().eval(), dummy.float().cpu())

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        errors = [parser.get_error(i).desc() for i in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT ONNX parse failed: {'; '.join(errors)}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if candidate.backend == "tensorrt_fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif candidate.backend == "tensorrt_int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _make_trt_calibrator(calibration_inputs)  # type: ignore[arg-type]

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed — check GPU capabilities.")

    engine_bytes = bytes(plan)
    runner = _TensorRTRunner(engine_bytes)
    return runner, dummy.float().cuda().contiguous(), len(engine_bytes) / 1e6


def _prepare_openvino(
    candidate: DeploymentCandidate,
    model: Any,
    dummy: torch.Tensor,
    calibration_inputs: list[Any] | None,
) -> tuple[_OpenVINORunner, torch.Tensor, float]:
    """Convert an nn.Module to an OpenVINO compiled model; return (runner, dummy, model_mb)."""
    import io

    import openvino as ov

    if candidate.backend == "openvino_int8" and not calibration_inputs:
        raise RuntimeError(
            "openvino_int8 requires calibration data. "
            "Re-run with --calibration-data to supply representative inputs."
        )

    onnx_bytes = _export_to_onnx_bytes(model.cpu().eval(), dummy.float().cpu())
    ov_model = ov.convert_model(io.BytesIO(onnx_bytes))

    if candidate.backend == "openvino_int8":
        try:
            import nncf

            def _transform(item: Any) -> dict:
                return {"input": item.float().cpu().numpy()}

            dataset = nncf.Dataset(calibration_inputs, _transform)
            ov_model = nncf.quantize(ov_model, dataset)
        except ImportError as exc:
            raise RuntimeError(
                "openvino_int8 requires the nncf package: pip install nncf"
            ) from exc

    core = ov.Core()
    compiled = core.compile_model(ov_model, "CPU")
    runner = _OpenVINORunner(compiled)
    return runner, dummy.float().cpu(), len(onnx_bytes) / 1e6


def _measure_accuracy_drop(
    candidate: DeploymentCandidate,
    model: Any,
    calibration_inputs: list[Any],
) -> float | None:
    """Return mean cosine-similarity drop between FP32 and quantized outputs.

    0.0 = outputs identical, 1.0 = outputs orthogonal. Returns None on any failure.
    """
    import copy
    import warnings

    import torch
    import torch.nn.functional as F

    try:
        fp32_model = copy.deepcopy(model).cpu().eval()
        fp32_outs: list[Any] = []
        with torch.no_grad():
            for inp in calibration_inputs:
                out = fp32_model(inp.float().cpu())
                fp32_outs.append(out.detach().flatten().float())

        quant_outs: list[Any] = []

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

        elif candidate.backend in ("pytorch_fp16", "pytorch_bf16"):
            torch_dtype = torch.float16 if "fp16" in candidate.backend else torch.bfloat16
            try:
                half_model = copy.deepcopy(model).cpu().to(torch_dtype).eval()
                with torch.no_grad():
                    for inp in calibration_inputs:
                        out = half_model(inp.float().cpu().to(torch_dtype))
                        quant_outs.append(out.detach().flatten().float())
            except Exception as exc:
                logger.info("accuracy proxy (%s) failed: %s", candidate.backend, exc)
                return None

        elif candidate.backend == "tensorrt_fp16":
            # Use pytorch_fp16 as a proxy — same numerical precision reduction.
            try:
                half_model = copy.deepcopy(model).cpu().to(torch.float16).eval()
                with torch.no_grad():
                    for inp in calibration_inputs:
                        out = half_model(inp.float().cpu().to(torch.float16))
                        quant_outs.append(out.detach().flatten().float())
            except Exception as exc:
                logger.info("tensorrt_fp16 accuracy proxy failed: %s", exc)
                return None

        elif candidate.backend in ("tensorrt_int8", "openvino_int8"):
            # Use pytorch_int8_dynamic as a proxy for quantization accuracy impact.
            _ensure_quantization_engine()
            quant_model = _safe_quantize_dynamic(copy.deepcopy(model).cpu().eval())
            with torch.no_grad():
                for inp in calibration_inputs:
                    out = quant_model(inp.float().cpu())
                    quant_outs.append(out.detach().flatten().float())

        elif _is_pruning_backend(candidate.backend):
            from infermap.pruning import prune_model, spec_for_backend
            try:
                pruned = copy.deepcopy(model).cpu().eval()
                pruned, _report = prune_model(pruned, spec_for_backend(candidate.backend))
                with torch.no_grad():
                    for inp in calibration_inputs:
                        out = pruned(inp.float().cpu())
                        quant_outs.append(out.detach().flatten().float())
            except Exception as exc:
                logger.info("pruning accuracy proxy (%s) failed: %s", candidate.backend, exc)
                return None

        if not quant_outs:
            return None

        drops = []
        for fp32_out, quant_out in zip(fp32_outs, quant_outs, strict=False):
            if fp32_out.numel() == 0:
                continue
            sim = F.cosine_similarity(fp32_out.unsqueeze(0), quant_out.unsqueeze(0)).item()
            drops.append(1.0 - max(-1.0, min(1.0, sim)))

        return sum(drops) / len(drops) if drops else None

    except Exception as exc:
        logger.warning("accuracy-drop measurement failed for %s: %s", candidate.backend, exc)
        return None


def _time_model(
    model: Any,
    dummy: Any,
    device: Any,
    warmup_iters: int,
    measure_iters: int,
) -> list[float]:
    import onnxruntime as ort
    import torch

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
            torch.mps.synchronize()

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
    model: Any, dummy: Any, device: Any, weight_mb: float
) -> float:
    """Return model memory in MB (weights + peak activations where measurable)."""
    import onnxruntime as ort
    import torch

    # TRT and OV allocate device memory outside PyTorch's allocator.
    # Return the engine/model size set during _prepare as a proxy.
    if isinstance(model, (_TensorRTRunner, _OpenVINORunner)):
        return weight_mb

    if device.type == "cuda":
        # CUDA tracks peak allocation precisely — includes weights + activations.
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(dummy)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / 1e6

    if device.type == "mps":
        # MPS: weight bytes are exact; add allocation delta for activations.
        torch.mps.synchronize()
        before = torch.mps.current_allocated_memory()
        with torch.no_grad():
            model(dummy)
        torch.mps.synchronize()
        after = torch.mps.current_allocated_memory()
        activation_mb = max(0.0, (after - before) / 1e6)
        return weight_mb + activation_mb

    # CPU / ONNX: weight bytes are exact; activations are small for batch_size=1.
    if isinstance(model, ort.InferenceSession):
        model.run(None, {"input": dummy.numpy()})
    else:
        with torch.no_grad():
            model(dummy)
    return weight_mb
