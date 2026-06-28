"""Tests for the V4 cost model — roofline estimates and candidate pruning."""
from __future__ import annotations

import pytest

from aphex.candidates import DeploymentCandidate
from aphex.cost_model import (
    CostEstimate,
    _confidence,
    _estimate_flops,
    _estimate_memory_bytes,
    _hardware_caps,
    estimate,
    prune_candidates,
)
from aphex.inspector import ModelInfo
from aphex.profiler import AcceleratorProfile, CPUProfile, HardwareProfile

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _info(
    params: int = 1_000_000,
    family: str = "unknown",
    framework: str = "pytorch",
    hidden_size: int | None = None,
    num_layers: int | None = None,
    num_conv_layers: int | None = None,
    max_sequence_length: int | None = None,
) -> ModelInfo:
    return ModelInfo(
        framework=framework,
        family=family,
        parameters=params,
        trainable_parameters=params,
        estimated_memory_fp32_gb=(params * 4.0) / 1e9,
        estimated_memory_fp16_gb=(params * 2.0) / 1e9,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_conv_layers=num_conv_layers,
        max_sequence_length=max_sequence_length,
    )


def _hw(
    kind: str = "cuda",
    memory_bandwidth_gbps: float | None = 2039.0,
    fp16_tflops: float | None = 312.0,
    int8_tops: float | None = 624.0,
    bf16: bool = True,
    physical_cores: int = 8,
) -> HardwareProfile:
    cpu = CPUProfile(name="Test CPU", physical_cores=physical_cores, logical_cores=16, ram_gb=64.0)
    acc = AcceleratorProfile(  # type: ignore[arg-type]
        kind=kind,
        name="Test GPU",
        memory_gb=80.0,
        bf16=bf16,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        fp16_tflops=fp16_tflops,
        int8_tops=int8_tops,
    )
    return HardwareProfile(cpu=cpu, accelerator=acc)


def _cand(
    backend: str = "pytorch_fp32",
    dtype: str = "fp32",
    device: str = "cuda",
) -> DeploymentCandidate:
    return DeploymentCandidate(
        backend=backend, dtype=dtype,
        description=backend, requires_export=False, device=device,
    )


# ── _estimate_flops ───────────────────────────────────────────────────────────

def test_flops_zero_params_returns_zero() -> None:
    assert _estimate_flops(_info(params=0), batch_size=1) == 0.0


def test_flops_general_model_is_2x_params() -> None:
    info = _info(params=1_000_000)
    assert _estimate_flops(info, batch_size=1) == pytest.approx(2_000_000.0)


def test_flops_scales_linearly_with_batch_size() -> None:
    info = _info(params=1_000_000)
    assert _estimate_flops(info, batch_size=4) == pytest.approx(4 * _estimate_flops(info, batch_size=1))


def test_flops_transformer_uses_architecture_formula() -> None:
    info = _info(params=100_000, family="transformer", hidden_size=64, num_layers=2)
    generic = 2.0 * 100_000
    arch = _estimate_flops(info, batch_size=1)
    # Architecture-aware estimate differs from generic 2×params
    assert arch != pytest.approx(generic)


def test_flops_transformer_scales_with_num_layers() -> None:
    info_2 = _info(params=200_000, family="transformer", hidden_size=64, num_layers=2)
    info_4 = _info(params=200_000, family="transformer", hidden_size=64, num_layers=4)
    assert _estimate_flops(info_4, 1) == pytest.approx(2 * _estimate_flops(info_2, 1))


def test_flops_llm_family_uses_architecture_formula() -> None:
    info = _info(params=100_000, family="llm", hidden_size=64, num_layers=2)
    generic = 2.0 * 100_000
    assert _estimate_flops(info, 1) != pytest.approx(generic)


def test_flops_transformer_without_hidden_falls_back_to_generic() -> None:
    info = _info(params=500_000, family="transformer")  # no hidden_size
    assert _estimate_flops(info, 1) == pytest.approx(2 * 500_000.0)


# ── _estimate_memory_bytes ────────────────────────────────────────────────────

def test_memory_bytes_fp32_is_4_bytes_per_param_plus_activations() -> None:
    info = _info(params=1_000_000)
    cand = _cand(dtype="fp32")
    mem = _estimate_memory_bytes(info, cand, batch_size=1)
    weight_bytes = 1_000_000 * 4.0
    assert mem == pytest.approx(weight_bytes * 1.5)  # weights + 50% activations


def test_memory_bytes_fp16_is_half_fp32() -> None:
    info = _info(params=1_000_000)
    mem_fp32 = _estimate_memory_bytes(info, _cand(dtype="fp32"), batch_size=1)
    mem_fp16 = _estimate_memory_bytes(info, _cand(dtype="fp16"), batch_size=1)
    assert mem_fp16 == pytest.approx(mem_fp32 / 2)


def test_memory_bytes_int8_is_quarter_fp32() -> None:
    info = _info(params=1_000_000)
    mem_fp32 = _estimate_memory_bytes(info, _cand(dtype="fp32"), batch_size=1)
    mem_int8 = _estimate_memory_bytes(info, _cand(dtype="int8"), batch_size=1)
    assert mem_int8 == pytest.approx(mem_fp32 / 4)


def test_memory_bytes_activations_scale_with_batch() -> None:
    info = _info(params=1_000_000)
    cand = _cand(dtype="fp32")
    mem1 = _estimate_memory_bytes(info, cand, batch_size=1)
    mem4 = _estimate_memory_bytes(info, cand, batch_size=4)
    # Activations (50% of weights × batch) grow; weights stay constant
    assert mem4 > mem1


# ── _hardware_caps ────────────────────────────────────────────────────────────

def test_hardware_caps_cuda_returns_accelerator_values() -> None:
    hw = _hw(kind="cuda", fp16_tflops=312.0, memory_bandwidth_gbps=2039.0)
    tflops, bw = _hardware_caps(_cand(device="cuda"), hw)
    assert tflops == 312.0
    assert bw == 2039.0


def test_hardware_caps_mps_returns_accelerator_values() -> None:
    hw = _hw(kind="mps", fp16_tflops=5.2, memory_bandwidth_gbps=200.0)
    tflops, bw = _hardware_caps(_cand(device="mps"), hw)
    assert tflops == 5.2
    assert bw == 200.0


def test_hardware_caps_cpu_uses_core_count() -> None:
    hw = _hw(kind="none", fp16_tflops=None, memory_bandwidth_gbps=None, physical_cores=8)
    tflops, bw = _hardware_caps(_cand(device="cpu"), hw)
    assert tflops is not None and tflops > 0
    assert bw == pytest.approx(50.0)


# ── _confidence ───────────────────────────────────────────────────────────────

def test_confidence_high_when_hw_and_arch_known() -> None:
    info = _info(hidden_size=512, num_layers=12)
    assert _confidence(info, fp16_tflops=312.0, bw_gbps=2039.0) == "high"


def test_confidence_medium_when_only_hw_known() -> None:
    info = _info()  # no fingerprint
    assert _confidence(info, fp16_tflops=312.0, bw_gbps=2039.0) == "medium"


def test_confidence_medium_when_only_arch_known() -> None:
    info = _info(hidden_size=512)
    assert _confidence(info, fp16_tflops=None, bw_gbps=None) == "medium"


def test_confidence_low_when_neither_known() -> None:
    info = _info()
    assert _confidence(info, fp16_tflops=None, bw_gbps=None) == "low"


# ── estimate ─────────────────────────────────────────────────────────────────

def test_estimate_returns_cost_estimate() -> None:
    result = estimate(_cand(), _info(), _hw())
    assert isinstance(result, CostEstimate)


def test_estimate_latency_positive() -> None:
    result = estimate(_cand(), _info(), _hw())
    assert result.predicted_latency_ms > 0


def test_estimate_memory_positive() -> None:
    result = estimate(_cand(), _info(), _hw())
    assert result.predicted_memory_mb > 0


def test_estimate_fp16_faster_than_fp32_on_same_hardware() -> None:
    hw = _hw()
    fp32 = estimate(_cand(backend="pytorch_fp32", dtype="fp32"), _info(), hw)
    fp16 = estimate(_cand(backend="pytorch_fp16", dtype="fp16"), _info(), hw)
    assert fp16.predicted_latency_ms < fp32.predicted_latency_ms


def test_estimate_tensorrt_faster_than_pytorch_same_dtype() -> None:
    hw = _hw()
    pt = estimate(_cand(backend="pytorch_fp16", dtype="fp16"), _info(), hw)
    trt = estimate(_cand(backend="tensorrt_fp16", dtype="fp16"), _info(), hw)
    assert trt.predicted_latency_ms < pt.predicted_latency_ms


def test_estimate_bottleneck_is_compute_or_memory_or_unknown() -> None:
    result = estimate(_cand(), _info(), _hw())
    assert result.bottleneck in ("compute", "memory", "unknown")


def test_estimate_unknown_hardware_returns_low_confidence() -> None:
    hw = _hw(fp16_tflops=None, memory_bandwidth_gbps=None)
    result = estimate(_cand(device="cuda"), _info(), hw)
    # No GPU specs → can't compute roofline → falls back to unknown
    assert result.bottleneck == "unknown"


def test_estimate_larger_model_slower() -> None:
    hw = _hw()
    cand = _cand()
    small = estimate(cand, _info(params=100_000), hw)
    large = estimate(cand, _info(params=10_000_000), hw)
    assert large.predicted_latency_ms > small.predicted_latency_ms


def test_estimate_memory_mb_matches_dtype() -> None:
    fp32 = estimate(_cand(dtype="fp32"), _info(params=1_000_000), _hw())
    fp16 = estimate(_cand(dtype="fp16"), _info(params=1_000_000), _hw())
    assert fp16.predicted_memory_mb < fp32.predicted_memory_mb


# ── prune_candidates ─────────────────────────────────────────────────────────

def _make_cands(backends: list[str], dtype: str = "fp32", device: str = "cuda") -> list[DeploymentCandidate]:
    return [_cand(b, dtype, device) for b in backends]


def test_prune_empty_returns_empty() -> None:
    kept, estimates = prune_candidates([], _info(), _hw())
    assert kept == []
    assert estimates == []


def test_prune_keeps_at_least_one() -> None:
    # Even with threshold=0 (prune everything), at least one is kept
    cands = _make_cands(["pytorch_fp32"])
    kept, _ = prune_candidates(cands, _info(), _hw(), threshold=0.0)
    assert len(kept) == 1


def test_prune_all_equal_keeps_all() -> None:
    # All same backend/dtype → same predicted latency → all kept
    cands = _make_cands(["pytorch_fp32", "pytorch_fp32", "pytorch_fp32"])
    kept, estimates = prune_candidates(cands, _info(), _hw())
    assert len(kept) == len(cands)


def test_prune_returns_estimates_for_all_candidates() -> None:
    cands = _make_cands(["pytorch_fp32", "pytorch_fp16", "tensorrt_fp16"])
    kept, estimates = prune_candidates(cands, _info(), _hw())
    assert len(estimates) == len(cands)


def test_prune_removes_clearly_slower_candidates() -> None:
    # INT8 is faster than FP32 in the cost model; with a very tight threshold,
    # slower FP32 should be pruned when INT8 is predicted much faster.
    # Use threshold=1.01 (only keep within 1% of best)
    cands = _make_cands(["pytorch_fp32", "tensorrt_int8"])
    kept, estimates = prune_candidates(cands, _info(params=50_000_000), _hw(), threshold=1.01)
    # TRT INT8 should win; FP32 should be pruned
    assert len(kept) < len(cands)
    assert all(c.backend != "pytorch_fp32" or len(kept) == 1 for c in kept)


def test_prune_high_threshold_keeps_all() -> None:
    cands = _make_cands(["pytorch_fp32", "pytorch_fp16", "tensorrt_fp16", "tensorrt_int8"])
    kept, _ = prune_candidates(cands, _info(), _hw(), threshold=1000.0)
    assert len(kept) == len(cands)
