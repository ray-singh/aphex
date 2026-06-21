"""Tests for system-level serving configuration recommendations."""
from __future__ import annotations

import pytest

from infermap.benchmark import BenchmarkResult
from infermap.candidates import DeploymentCandidate
from infermap.inspector import ModelInfo
from infermap.profiler import AcceleratorProfile, CPUProfile, HardwareProfile
from infermap.recommender import Recommendation
from infermap.system_recommender import (
    SystemConfig,
    _has_fp8_support,
    _kv_cache_fraction,
    _max_safe_batch_size,
    _prefill_chunk_size,
    recommend_system_config,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _info(
    family: str = "unknown",
    params: int = 1_000_000,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    max_sequence_length: int | None = None,
) -> ModelInfo:
    return ModelInfo(
        framework="pytorch",
        family=family,
        parameters=params,
        trainable_parameters=params,
        estimated_memory_fp32_gb=(params * 4.0) / 1e9,
        estimated_memory_fp16_gb=(params * 2.0) / 1e9,
        hidden_size=hidden_size,
        num_layers=num_layers,
        max_sequence_length=max_sequence_length,
    )


def _hw(
    kind: str = "cuda",
    memory_gb: float = 80.0,
    ram_gb: float = 256.0,
    physical_cores: int = 16,
    accelerator_name: str = "NVIDIA A100-SXM4-80GB",
) -> HardwareProfile:
    cpu = CPUProfile(name="Intel Xeon", physical_cores=physical_cores, logical_cores=physical_cores * 2, ram_gb=ram_gb)
    acc = AcceleratorProfile(  # type: ignore[arg-type]
        kind=kind,
        name=accelerator_name,
        memory_gb=memory_gb,
        bf16=True,
    )
    return HardwareProfile(cpu=cpu, accelerator=acc)


def _cand(
    backend: str = "tensorrt_fp16",
    dtype: str = "fp16",
    device: str = "cuda",
) -> DeploymentCandidate:
    return DeploymentCandidate(
        backend=backend, dtype=dtype,
        description=backend, requires_export=False, device=device,
    )


def _result(
    memory_mb: float = 420.0,
    batch_size: int = 4,
    latency: float = 3.5,
    device: str = "cuda",
) -> BenchmarkResult:
    cand = _cand(device=device)
    return BenchmarkResult(
        candidate=cand,
        latency_p50_ms=latency,
        latency_p95_ms=latency * 1.1,
        latency_p99_ms=latency * 1.2,
        throughput_rps=1000.0 / latency,
        memory_mb=memory_mb,
        accuracy_drop=None,
        batch_size=batch_size,
    )


def _rec(result: BenchmarkResult | None = None) -> Recommendation:
    r = result or _result()
    return Recommendation(result=r, rationale="Best.", pareto_frontier=[r], all_results=[r])


# ── _max_safe_batch_size ──────────────────────────────────────────────────────

def test_max_safe_batch_size_positive() -> None:
    r = _result(memory_mb=420.0, batch_size=4)
    assert _max_safe_batch_size(r, _hw(memory_gb=80.0), "cuda") >= 1


def test_max_safe_batch_size_gpu_bounded() -> None:
    # 40 GB VRAM, 4 GB model at bs=4 → max safe < 512
    r = _result(memory_mb=4_000.0, batch_size=4)
    result = _max_safe_batch_size(r, _hw(memory_gb=40.0), "cuda")
    assert 1 <= result <= 512


def test_max_safe_batch_size_small_vram_limits_batch() -> None:
    # 8 GB VRAM, 7 GB model → only 1 batch fits
    r = _result(memory_mb=7_000.0, batch_size=1)
    small_hw = _hw(memory_gb=8.0)
    large_hw = _hw(memory_gb=80.0)
    assert _max_safe_batch_size(r, small_hw, "cuda") <= _max_safe_batch_size(r, large_hw, "cuda")


def test_max_safe_batch_size_cpu_uses_ram() -> None:
    r = _result(memory_mb=500.0, batch_size=1, device="cpu")
    # 128 GB RAM → large batch fits
    result = _max_safe_batch_size(r, _hw(kind="none", memory_gb=0.0, ram_gb=128.0), "cpu")
    assert result >= 4


def test_max_safe_batch_size_minimum_one_when_model_barely_fits() -> None:
    # Model almost fills VRAM entirely
    r = _result(memory_mb=75_000.0, batch_size=1)
    result = _max_safe_batch_size(r, _hw(memory_gb=80.0), "cuda")
    assert result >= 1


# ── _kv_cache_fraction ────────────────────────────────────────────────────────

def test_kv_cache_fraction_in_valid_range() -> None:
    frac = _kv_cache_fraction(10_000.0, _hw(memory_gb=80.0))
    assert 0.10 <= frac <= 0.90


def test_kv_cache_fraction_decreases_with_larger_model() -> None:
    hw = _hw(memory_gb=80.0)
    small = _kv_cache_fraction(5_000.0, hw)
    large = _kv_cache_fraction(60_000.0, hw)
    assert large < small


def test_kv_cache_fraction_minimum_when_model_overfills() -> None:
    # Model exceeds VRAM
    frac = _kv_cache_fraction(100_000.0, _hw(memory_gb=80.0))
    assert frac == pytest.approx(0.10)


# ── _prefill_chunk_size ───────────────────────────────────────────────────────

def test_prefill_chunk_none_for_short_context() -> None:
    assert _prefill_chunk_size(_info(max_sequence_length=512)) is None


def test_prefill_chunk_none_at_2048_boundary() -> None:
    assert _prefill_chunk_size(_info(max_sequence_length=2048)) is None


def test_prefill_chunk_1024_for_medium_context() -> None:
    assert _prefill_chunk_size(_info(max_sequence_length=4096)) == 1024


def test_prefill_chunk_512_for_long_context() -> None:
    assert _prefill_chunk_size(_info(max_sequence_length=8192)) == 512


def test_prefill_chunk_none_when_no_sequence_length() -> None:
    assert _prefill_chunk_size(_info()) is None


# ── _has_fp8_support ──────────────────────────────────────────────────────────

def test_fp8_on_h100() -> None:
    assert _has_fp8_support(_hw(accelerator_name="NVIDIA H100 SXM5 80GB"))


def test_fp8_on_h200() -> None:
    assert _has_fp8_support(_hw(accelerator_name="NVIDIA H200 141GB"))


def test_fp8_on_b200() -> None:
    assert _has_fp8_support(_hw(accelerator_name="NVIDIA B200"))


def test_fp8_on_rtx_4090() -> None:
    assert _has_fp8_support(_hw(accelerator_name="NVIDIA GeForce RTX 4090"))


def test_fp8_not_on_a100() -> None:
    assert not _has_fp8_support(_hw(accelerator_name="NVIDIA A100-SXM4-80GB"))


def test_fp8_not_on_v100() -> None:
    assert not _has_fp8_support(_hw(accelerator_name="NVIDIA V100-SXM2-32GB"))


def test_fp8_not_on_t4() -> None:
    assert not _has_fp8_support(_hw(accelerator_name="Tesla T4"))


# ── recommend_system_config ───────────────────────────────────────────────────

def test_returns_system_config() -> None:
    cfg = recommend_system_config(_rec(), _info(), _hw())
    assert isinstance(cfg, SystemConfig)


def test_recommended_batch_capped_by_max_safe() -> None:
    cfg = recommend_system_config(_rec(), _info(), _hw())
    assert cfg.recommended_batch_size <= cfg.max_safe_batch_size


def test_num_workers_from_cpu_cores() -> None:
    cfg = recommend_system_config(_rec(), _info(), _hw(physical_cores=16))
    assert cfg.num_workers == 8  # 16 // 2 = 8, capped at _MAX_WORKERS=8


def test_num_workers_minimum_one() -> None:
    cfg = recommend_system_config(_rec(), _info(), _hw(physical_cores=1))
    assert cfg.num_workers >= 1


def test_num_workers_capped_at_eight() -> None:
    cfg = recommend_system_config(_rec(), _info(), _hw(physical_cores=64))
    assert cfg.num_workers <= 8


def test_tensor_parallel_defaults_to_one() -> None:
    cfg = recommend_system_config(_rec(), _info(), _hw())
    assert cfg.tensor_parallel_size == 1


def test_dynamic_batching_when_throughput_objective() -> None:
    cfg = recommend_system_config(_rec(), _info(family="unknown"), _hw(), objective="throughput")
    assert cfg.dynamic_batching is True


def test_dynamic_batching_for_llm_family() -> None:
    cfg = recommend_system_config(_rec(), _info(family="llm"), _hw(), objective="latency")
    assert cfg.dynamic_batching is True


def test_dynamic_batching_for_transformer_family() -> None:
    cfg = recommend_system_config(_rec(), _info(family="transformer"), _hw(), objective="latency")
    assert cfg.dynamic_batching is True


def test_no_dynamic_batching_latency_non_llm() -> None:
    cfg = recommend_system_config(_rec(), _info(family="unknown"), _hw(), objective="latency")
    assert cfg.dynamic_batching is False


def test_llm_populates_kv_cache_fraction() -> None:
    cfg = recommend_system_config(_rec(), _info(family="llm"), _hw())
    assert cfg.kv_cache_fraction is not None
    assert 0.10 <= cfg.kv_cache_fraction <= 0.90


def test_llm_continuous_batching_true() -> None:
    cfg = recommend_system_config(_rec(), _info(family="llm"), _hw())
    assert cfg.continuous_batching is True


def test_non_llm_kv_cache_fraction_none() -> None:
    cfg = recommend_system_config(_rec(), _info(family="unknown"), _hw())
    assert cfg.kv_cache_fraction is None


def test_non_llm_continuous_batching_none() -> None:
    cfg = recommend_system_config(_rec(), _info(family="unknown"), _hw())
    assert cfg.continuous_batching is None


def test_llm_kv_dtype_fp16_on_a100() -> None:
    cfg = recommend_system_config(
        _rec(), _info(family="llm"),
        _hw(accelerator_name="NVIDIA A100-SXM4-80GB"),
    )
    assert cfg.kv_cache_dtype == "fp16"


def test_llm_kv_dtype_fp8_on_h100() -> None:
    cfg = recommend_system_config(
        _rec(), _info(family="llm"),
        _hw(accelerator_name="NVIDIA H100 SXM5 80GB"),
    )
    assert cfg.kv_cache_dtype == "fp8"


def test_cpu_only_llm_no_kv_cache() -> None:
    cpu_hw = _hw(kind="none", memory_gb=0.0, accelerator_name="CPU-only")
    r = _result(device="cpu")
    cfg = recommend_system_config(_rec(r), _info(family="llm"), cpu_hw)
    assert cfg.kv_cache_fraction is None
    assert cfg.continuous_batching is None


def test_classifier_enables_result_cache() -> None:
    cfg = recommend_system_config(_rec(), _info(family="sklearn"), _hw())
    assert cfg.enable_result_cache is True


def test_llm_disables_result_cache() -> None:
    cfg = recommend_system_config(_rec(), _info(family="llm"), _hw())
    assert cfg.enable_result_cache is False


def test_non_classifier_disables_result_cache() -> None:
    cfg = recommend_system_config(_rec(), _info(family="transformer"), _hw())
    assert cfg.enable_result_cache is False


def test_llm_prefill_chunk_for_long_context() -> None:
    cfg = recommend_system_config(
        _rec(), _info(family="llm", max_sequence_length=8192), _hw()
    )
    assert cfg.prefill_chunk_size == 512


def test_llm_no_prefill_chunk_for_short_context() -> None:
    cfg = recommend_system_config(
        _rec(), _info(family="llm", max_sequence_length=512), _hw()
    )
    assert cfg.prefill_chunk_size is None
