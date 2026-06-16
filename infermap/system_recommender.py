"""System-level serving configuration recommendations.

Derives batching, parallelism, KV-cache, and worker settings from the
benchmark result + hardware profile without running additional benchmarks.
"""

from __future__ import annotations
from dataclasses import dataclass
from infermap.benchmark import BenchmarkResult
from infermap.inspector import ModelInfo
from infermap.profiler import HardwareProfile
from infermap.recommender import Recommendation

_LLM_FAMILIES = frozenset({"llm", "transformer"})
_CLASSIFIER_FAMILIES = frozenset({"sklearn", "xgboost", "lightgbm", "catboost"})

# GPUs with native FP8 Tensor Core support (Ada Lovelace / Hopper / Blackwell)
_FP8_GPU_SUBSTR = ("h100", "h200", "b200", "rtx 4090", "rtx 4080", "rtx 4070", "rtx 4060", "4090", "4080")

_MAX_WORKERS = 8
_MEMORY_HEADROOM = 0.85  # leave 15 % of VRAM free for driver / framework overhead
_RAM_FRACTION = 0.80     # use at most 80 % of system RAM for CPU-device models


@dataclass
class SystemConfig:
    # Batching
    max_safe_batch_size: int
    recommended_batch_size: int
    dynamic_batching: bool
    # Workers
    num_workers: int
    # Parallelism
    tensor_parallel_size: int
    # LLM-specific (None for non-LLM models)
    kv_cache_fraction: float | None
    continuous_batching: bool | None
    prefill_chunk_size: int | None
    kv_cache_dtype: str | None
    # General
    enable_result_cache: bool


def recommend_system_config(
    rec: Recommendation,
    info: ModelInfo,
    hw: HardwareProfile,
    objective: str = "latency",
) -> SystemConfig:
    """Derive system-level serving settings from a benchmark recommendation."""
    result = rec.result
    cand = result.candidate

    is_llm = info.family in _LLM_FAMILIES
    is_classifier = info.family in _CLASSIFIER_FAMILIES

    max_safe = _max_safe_batch_size(result, hw, cand.device)
    rec_batch = min(result.batch_size, max_safe)
    dynamic = objective == "throughput" or is_llm
    num_workers = max(1, min(hw.cpu.physical_cores // 2, _MAX_WORKERS))

    kv_fraction: float | None = None
    continuous: bool | None = None
    prefill_chunk: int | None = None
    kv_dtype: str | None = None

    if is_llm and hw.accelerator.kind != "none":
        kv_fraction = _kv_cache_fraction(result.memory_mb, hw)
        continuous = True
        prefill_chunk = _prefill_chunk_size(info)
        kv_dtype = "fp8" if _has_fp8_support(hw) else "fp16"

    return SystemConfig(
        max_safe_batch_size=max_safe,
        recommended_batch_size=rec_batch,
        dynamic_batching=dynamic,
        num_workers=num_workers,
        tensor_parallel_size=1,
        kv_cache_fraction=kv_fraction,
        continuous_batching=continuous,
        prefill_chunk_size=prefill_chunk,
        kv_cache_dtype=kv_dtype,
        enable_result_cache=is_classifier,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _max_safe_batch_size(
    result: BenchmarkResult,
    hw: HardwareProfile,
    device: str,
) -> int:
    """Estimate the largest batch size that fits in available memory."""
    if device in ("cuda", "mps"):
        available_mb = hw.accelerator.memory_gb * 1024
    else:
        available_mb = hw.cpu.ram_gb * 1024 * _RAM_FRACTION

    # cost_model activation model: total_mb = weight_mb * (1 + 0.5 * batch_size)
    measured_bs = result.batch_size
    weight_mb = result.memory_mb / (1 + 0.5 * measured_bs)

    if weight_mb <= 0 or weight_mb >= available_mb:
        return 1

    # Solve: weight_mb * (1 + 0.5 * N) <= available_mb * headroom
    max_n = int((available_mb * _MEMORY_HEADROOM - weight_mb) / (weight_mb * 0.5))
    return max(1, min(max_n, 512))


def _kv_cache_fraction(model_memory_mb: float, hw: HardwareProfile) -> float:
    """Fraction of accelerator memory to reserve for the KV cache."""
    total_mb = hw.accelerator.memory_gb * 1024
    remaining_mb = total_mb - model_memory_mb
    if remaining_mb <= 0:
        return 0.10
    fraction = (remaining_mb / total_mb) * 0.95
    return round(min(0.90, max(0.10, fraction)), 2)


def _prefill_chunk_size(info: ModelInfo) -> int | None:
    """Return chunked-prefill size for long-context LLMs, else None."""
    seq_len = info.max_sequence_length or 0
    if seq_len > 4096:
        return 512
    if seq_len > 2048:
        return 1024
    return None


def _has_fp8_support(hw: HardwareProfile) -> bool:
    """True for Ada Lovelace / Hopper / Blackwell GPUs with native FP8 Tensor Cores."""
    name = hw.accelerator.name.lower()
    return any(s in name for s in _FP8_GPU_SUBSTR)
