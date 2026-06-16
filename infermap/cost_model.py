"""V4 Cost Model — predicts candidate performance before benchmarking via roofline analysis.

latency ≈ max(compute_time, memory_time) / efficiency + overhead
where:
  compute_time = flops / (fp16_tflops * 1e12)
  memory_time  = bytes_moved / (memory_bandwidth_gbps * 1e9)
"""

from __future__ import annotations

from dataclasses import dataclass

from infermap.candidates import DeploymentCandidate
from infermap.inspector import ModelInfo
from infermap.profiler import HardwareProfile


_DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "q_native": 4.0,  # conservative: treat as fp32 until actual bits are known
}

# Fraction of peak hardware throughput typically achieved per backend.
# Reflects real-world kernel efficiency, Python dispatch overhead, and JIT cost.
_BACKEND_EFFICIENCY: dict[str, float] = {
    "pytorch_fp32": 0.25,
    "pytorch_fp16": 0.35,
    "pytorch_bf16": 0.35,
    "pytorch_int8_dynamic": 0.30,
    "torch_compile_fp32": 0.40,
    "onnx_cpu": 0.35,
    "onnx_cuda": 0.35,
    "onnx_coreml": 0.40,
    "onnx_int8_cpu": 0.35,
    "tensorrt_fp32": 0.50,
    "tensorrt_fp16": 0.65,
    "tensorrt_int8": 0.70,
    "openvino_fp32": 0.35,
    "openvino_int8": 0.45,
}

# Fixed per-inference overhead (ms): kernel launch, Python dispatch, session amortization.
_BACKEND_OVERHEAD_MS: dict[str, float] = {
    "pytorch_fp32": 0.10,
    "pytorch_fp16": 0.10,
    "pytorch_bf16": 0.10,
    "pytorch_int8_dynamic": 0.15,
    "torch_compile_fp32": 0.05,
    "onnx_cpu": 0.20,
    "onnx_cuda": 0.15,
    "onnx_coreml": 0.20,
    "onnx_int8_cpu": 0.20,
    "tensorrt_fp32": 0.05,
    "tensorrt_fp16": 0.05,
    "tensorrt_int8": 0.05,
    "openvino_fp32": 0.15,
    "openvino_int8": 0.15,
}

_DEFAULT_CPU_BANDWIDTH_GBPS = 50.0
_DEFAULT_CPU_TFLOPS_PER_CORE = 0.05  # ~50 GFLOPS/core (conservative, SSE/AVX baseline)


@dataclass
class CostEstimate:
    candidate: DeploymentCandidate
    predicted_latency_ms: float
    predicted_memory_mb: float
    arithmetic_intensity: float  # ops/byte; 0.0 when unknown
    bottleneck: str              # "compute", "memory", or "unknown"
    confidence: str              # "high", "medium", or "low"


def estimate(
    candidate: DeploymentCandidate,
    info: ModelInfo,
    hardware: HardwareProfile,
    batch_size: int = 1,
) -> CostEstimate:
    """Predict latency and memory for a candidate without running it."""
    flops = _estimate_flops(info, batch_size)
    mem_bytes = _estimate_memory_bytes(info, candidate, batch_size)
    fp16_tflops, bw_gbps = _hardware_caps(candidate, hardware)
    confidence = _confidence(info, fp16_tflops, bw_gbps)

    if fp16_tflops is None or bw_gbps is None or flops == 0:
        pred_ms = max(
            mem_bytes / (_DEFAULT_CPU_BANDWIDTH_GBPS * 1e9) * 1000,
            _BACKEND_OVERHEAD_MS.get(candidate.backend, 0.15),
        )
        return CostEstimate(
            candidate=candidate,
            predicted_latency_ms=pred_ms,
            predicted_memory_mb=mem_bytes / 1e6,
            arithmetic_intensity=0.0,
            bottleneck="unknown",
            confidence=confidence,
        )

    compute_s = flops / (fp16_tflops * 1e12)
    memory_s = mem_bytes / (bw_gbps * 1e9)
    ai = flops / mem_bytes if mem_bytes > 0 else 0.0
    efficiency = _BACKEND_EFFICIENCY.get(candidate.backend, 0.30)
    overhead_ms = _BACKEND_OVERHEAD_MS.get(candidate.backend, 0.15)

    pred_ms = max(compute_s, memory_s) / efficiency * 1000 + overhead_ms
    bottleneck = "compute" if compute_s > memory_s else "memory"

    return CostEstimate(
        candidate=candidate,
        predicted_latency_ms=pred_ms,
        predicted_memory_mb=mem_bytes / 1e6,
        arithmetic_intensity=ai,
        bottleneck=bottleneck,
        confidence=confidence,
    )


def prune_candidates(
    candidates: list[DeploymentCandidate],
    info: ModelInfo,
    hardware: HardwareProfile,
    batch_size: int = 1,
    threshold: float = 10.0,
) -> tuple[list[DeploymentCandidate], list[CostEstimate]]:
    """Return (kept_candidates, all_estimates).

    Candidates predicted to be more than `threshold`× slower than the best
    prediction are dropped before benchmarking.  At least one candidate is
    always kept so the caller always has something to benchmark.
    """
    if not candidates:
        return [], []

    estimates = [estimate(c, info, hardware, batch_size) for c in candidates]
    best_ms = min(e.predicted_latency_ms for e in estimates)
    kept = [
        e.candidate for e in estimates
        if e.predicted_latency_ms <= best_ms * threshold
    ]
    if not kept:
        kept = [estimates[0].candidate]

    return kept, estimates


# ── Internal helpers ──────────────────────────────────────────────────────────

def _estimate_flops(info: ModelInfo, batch_size: int) -> float:
    """Estimate FLOPs for one forward pass (2 × MACs)."""
    P = info.parameters
    if P == 0:
        return 0.0

    if info.family in ("transformer", "llm") and info.hidden_size and info.num_layers:
        ff_len = info.hidden_size * 4  # standard FFN expansion ratio
        seq_len = info.max_sequence_length or 512
        # Q/K/V/O projections: 4 × hidden² each side → 8 × hidden²
        # FFN: 2 linear layers → 4 × hidden × ff_len
        # Attention scores: 2 × seq_len × hidden per layer
        per_layer = (
            8 * info.hidden_size ** 2
            + 4 * info.hidden_size * ff_len
            + 2 * seq_len * info.hidden_size
        )
        return 2.0 * info.num_layers * per_layer * batch_size

    # General fallback: 2 FLOPs per parameter (multiply-accumulate)
    return 2.0 * P * batch_size


def _estimate_memory_bytes(
    info: ModelInfo,
    candidate: DeploymentCandidate,
    batch_size: int,
) -> float:
    """Estimate bytes moved through memory during one forward pass."""
    bytes_per_param = _DTYPE_BYTES.get(candidate.dtype, 4.0)
    weight_bytes = info.parameters * bytes_per_param
    # Inference activations ≈ 50% of weight size (heuristic; no gradient buffers)
    activation_bytes = weight_bytes * 0.5 * batch_size
    return weight_bytes + activation_bytes


def _hardware_caps(
    candidate: DeploymentCandidate,
    hardware: HardwareProfile,
) -> tuple[float | None, float | None]:
    """Return (fp16_tflops, memory_bandwidth_gbps) for the candidate's target device."""
    if candidate.device in ("cuda", "mps"):
        return hardware.accelerator.fp16_tflops, hardware.accelerator.memory_bandwidth_gbps
    # CPU: use detected core count × conservative per-core TFLOPS
    tflops = hardware.cpu.physical_cores * _DEFAULT_CPU_TFLOPS_PER_CORE
    return tflops, _DEFAULT_CPU_BANDWIDTH_GBPS


def _confidence(
    info: ModelInfo,
    fp16_tflops: float | None,
    bw_gbps: float | None,
) -> str:
    """Rate confidence based on available fingerprint and hardware data."""
    has_hw = fp16_tflops is not None and bw_gbps is not None
    has_arch = info.hidden_size is not None or info.num_conv_layers is not None
    if has_hw and has_arch:
        return "high"
    if has_hw or has_arch:
        return "medium"
    return "low"
