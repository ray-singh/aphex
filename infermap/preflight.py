"""Pre-flight validator — fast feasibility check before any benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from infermap.inspector import BYTES_PER_PARAM, ModelInfo
from infermap.profiler import HardwareProfile

FeasibilityCategory = Literal["ok", "tight", "unlikely", "impossible"]

# Reference throughput ranges (tokens/sec or req/sec) for estimation.
# Keyed by (accelerator_kind, model_family, param_scale).
# param_scale: "small" <300M, "medium" 300M–3B, "large" >3B
_THROUGHPUT_TABLE: dict[tuple[str, str, str], tuple[float, float]] = {
    ("cuda", "transformer", "small"): (800.0, 2000.0),
    ("cuda", "transformer", "medium"): (200.0, 600.0),
    ("cuda", "transformer", "large"): (30.0, 120.0),
    ("cuda", "cnn", "small"): (500.0, 3000.0),
    ("cuda", "cnn", "medium"): (200.0, 800.0),
    ("mps", "transformer", "small"): (100.0, 400.0),
    ("mps", "transformer", "medium"): (20.0, 80.0),
    ("mps", "transformer", "large"): (2.0, 10.0),
    ("mps", "cnn", "small"): (150.0, 600.0),
    ("mps", "cnn", "medium"): (50.0, 200.0),
    ("none", "transformer", "small"): (30.0, 150.0),
    ("none", "transformer", "medium"): (5.0, 30.0),
    ("none", "cnn", "small"): (80.0, 300.0),
    ("none", "cnn", "medium"): (20.0, 100.0),
}


@dataclass
class PreflightResult:
    feasible: bool
    category: FeasibilityCategory
    message: str
    suggestions: list[str]
    estimated_memory_gb: float
    available_memory_gb: float
    estimated_throughput_range: tuple[float, float] | None = None
    constraint_throughput: float | None = None


def run_preflight(
    model_info: ModelInfo,
    hardware: HardwareProfile,
    dtype: str = "fp32",
    target_throughput: float | None = None,
) -> PreflightResult:
    """
    Validate hardware/constraint feasibility before benchmarking.

    Args:
        model_info: Output of inspect_model().
        hardware: Output of profile_hardware().
        dtype: Precision to use for memory estimation.
        target_throughput: User's target req/sec or tokens/sec (optional).
    """
    estimated_gb = model_info.estimated_memory_gb(dtype)
    available_gb = hardware.available_memory_gb

    throughput_range = _estimate_throughput(model_info, hardware)
    suggestions: list[str] = []

    # --- Memory feasibility ---
    if estimated_gb > available_gb:
        # Check if int4 would fit
        int4_gb = model_info.estimated_memory_gb("int4")
        suggestions.extend(
            _suggest_alternatives(model_info, hardware, estimated_gb, available_gb)
        )
        if int4_gb > available_gb:
            return PreflightResult(
                feasible=False,
                category="impossible",
                message=(
                    f"Model requires ~{estimated_gb:.1f} GB ({dtype.upper()}) "
                    f"but only ~{available_gb:.1f} GB is available. "
                    f"Even INT4 (~{int4_gb:.1f} GB) exceeds available memory."
                ),
                suggestions=suggestions,
                estimated_memory_gb=estimated_gb,
                available_memory_gb=available_gb,
                estimated_throughput_range=throughput_range,
                constraint_throughput=target_throughput,
            )
        else:
            suggestions.insert(
                0,
                f"INT4 quantization reduces memory to ~{int4_gb:.1f} GB — may fit with aggressive quantization",
            )
            return PreflightResult(
                feasible=True,
                category="tight",
                message=(
                    f"Model requires ~{estimated_gb:.1f} GB ({dtype.upper()}) "
                    f"but only ~{available_gb:.1f} GB is available. "
                    f"INT4 quantization (~{int4_gb:.1f} GB) may fit."
                ),
                suggestions=suggestions,
                estimated_memory_gb=estimated_gb,
                available_memory_gb=available_gb,
                estimated_throughput_range=throughput_range,
                constraint_throughput=target_throughput,
            )

    # --- Throughput feasibility (if a target was given) ---
    if target_throughput is not None and throughput_range is not None:
        low, high = throughput_range
        if target_throughput > high * 2:
            suggestions.append(
                f"Estimated achievable throughput on this hardware: {low:.0f}–{high:.0f} req/s"
            )
            suggestions.append("Consider a more powerful accelerator or reduce batch size")
            return PreflightResult(
                feasible=True,
                category="unlikely",
                message=(
                    f"Target throughput of {target_throughput:.0f} req/s is unlikely. "
                    f"Estimated range on this hardware: {low:.0f}–{high:.0f} req/s."
                ),
                suggestions=suggestions,
                estimated_memory_gb=estimated_gb,
                available_memory_gb=available_gb,
                estimated_throughput_range=throughput_range,
                constraint_throughput=target_throughput,
            )

    return PreflightResult(
        feasible=True,
        category="ok",
        message="Hardware and constraints look feasible. Proceeding with benchmarking.",
        suggestions=[],
        estimated_memory_gb=estimated_gb,
        available_memory_gb=available_gb,
        estimated_throughput_range=throughput_range,
        constraint_throughput=target_throughput,
    )


def _param_scale(params: int) -> str:
    if params < 300_000_000:
        return "small"
    if params < 3_000_000_000:
        return "medium"
    return "large"


def _estimate_throughput(
    model_info: ModelInfo, hardware: HardwareProfile
) -> tuple[float, float] | None:
    scale = _param_scale(model_info.parameters)
    key = (hardware.accelerator.kind, model_info.family, scale)
    return _THROUGHPUT_TABLE.get(key)


def _suggest_alternatives(
    model_info: ModelInfo,
    hardware: HardwareProfile,
    estimated_gb: float,
    available_gb: float,
) -> list[str]:
    suggestions = []
    gap = estimated_gb - available_gb

    for dtype, bpp in BYTES_PER_PARAM.items():
        if dtype == "fp32":
            continue
        alt_gb = (model_info.parameters * bpp * 1.2) / 1e9
        if alt_gb <= available_gb:
            suggestions.append(
                f"{dtype.upper()} quantization reduces memory to ~{alt_gb:.1f} GB — fits on this device"
            )

    suggestions.append(
        f"Target a machine with ≥ {estimated_gb + 4:.0f} GB RAM/VRAM"
    )
    suggestions.append("Consider an API-based deployment instead of local inference")
    return suggestions
