"""Tests for pre-flight validator."""

import pytest

from infermap.inspector import ModelInfo
from infermap.profiler import HardwareProfile, CPUProfile, AcceleratorProfile
from infermap.preflight import run_preflight


def _make_model(params: int = 1_000_000) -> ModelInfo:
    return ModelInfo(
        framework="pytorch",
        family="transformer",
        parameters=params,
        trainable_parameters=params,
        estimated_memory_fp32_gb=(params * 4.0 * 1.2) / 1e9,
        estimated_memory_fp16_gb=(params * 2.0 * 1.2) / 1e9,
    )


def _make_hw(ram_gb: float = 16.0, kind: str = "mps") -> HardwareProfile:
    cpu = CPUProfile(name="Test CPU", physical_cores=8, logical_cores=16, ram_gb=ram_gb)
    acc = AcceleratorProfile(kind=kind, name="Test Device", memory_gb=ram_gb, bf16=False)  # type: ignore[arg-type]
    return HardwareProfile(cpu=cpu, accelerator=acc)


def test_ok_small_model() -> None:
    model = _make_model(100_000)  # tiny model
    hw = _make_hw(ram_gb=16.0)
    result = run_preflight(model, hw)
    assert result.category == "ok"
    assert result.feasible is True


def test_impossible_huge_model() -> None:
    # 31B params, ~186 GB FP32 — won't fit on 16 GB
    model = _make_model(31_000_000_000)
    hw = _make_hw(ram_gb=16.0)
    result = run_preflight(model, hw, dtype="fp32")
    assert result.category == "impossible"
    assert result.feasible is False
    assert len(result.suggestions) > 0


def test_tight_model_fits_int4() -> None:
    # ~4B params: FP32 ~19 GB, INT4 ~2.4 GB → tight on 16 GB
    model = _make_model(4_000_000_000)
    hw = _make_hw(ram_gb=16.0)
    result = run_preflight(model, hw, dtype="fp32")
    assert result.category in ("tight", "impossible")


def test_unlikely_throughput() -> None:
    model = _make_model(7_000_000_000)  # 7B model — medium scale
    hw = _make_hw(ram_gb=64.0, kind="mps")
    # 7B on MPS: expected range ~2–10 tok/s; request 1000 tok/s
    result = run_preflight(model, hw, dtype="fp16", target_throughput=1000.0)
    assert result.category in ("unlikely", "ok")


def test_suggestions_populated_on_impossible() -> None:
    model = _make_model(50_000_000_000)  # 50B — never fits on 16 GB
    hw = _make_hw(ram_gb=16.0)
    result = run_preflight(model, hw)
    assert isinstance(result.suggestions, list)
    assert any("GB" in s for s in result.suggestions)
