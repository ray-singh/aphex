"""Tests for candidate generation."""

import pytest

from infermap.candidates import DeploymentCandidate, generate_candidates
from infermap.inspector import ModelInfo
from infermap.profiler import AcceleratorProfile, CPUProfile, HardwareProfile


def _make_model() -> ModelInfo:
    params = 1_000_000
    return ModelInfo(
        framework="pytorch",
        family="cnn",
        parameters=params,
        trainable_parameters=params,
        estimated_memory_fp32_gb=(params * 4.0 * 1.2) / 1e9,
        estimated_memory_fp16_gb=(params * 2.0 * 1.2) / 1e9,
    )


def _make_hw(kind: str = "none", bf16: bool = False, memory_gb: float = 16.0) -> HardwareProfile:
    cpu = CPUProfile(name="Test CPU", physical_cores=4, logical_cores=8, ram_gb=16.0)
    acc = AcceleratorProfile(kind=kind, name="Test Device", memory_gb=memory_gb, bf16=bf16)  # type: ignore[arg-type]
    return HardwareProfile(cpu=cpu, accelerator=acc)


def test_cpu_only_candidates_use_cpu_device() -> None:
    candidates = generate_candidates(_make_model(), _make_hw(kind="none"))
    assert len(candidates) > 0
    assert all(c.device == "cpu" for c in candidates)


def test_mps_candidates_include_mps_device() -> None:
    candidates = generate_candidates(_make_model(), _make_hw(kind="mps"))
    devices = {c.device for c in candidates}
    assert "mps" in devices


def test_mps_candidates_include_cpu_fallbacks() -> None:
    candidates = generate_candidates(_make_model(), _make_hw(kind="mps"))
    devices = {c.device for c in candidates}
    assert "cpu" in devices


def test_cuda_candidates_include_cuda_device() -> None:
    candidates = generate_candidates(_make_model(), _make_hw(kind="cuda", memory_gb=24.0))
    devices = {c.device for c in candidates}
    assert "cuda" in devices


def test_bf16_emitted_when_supported() -> None:
    candidates = generate_candidates(_make_model(), _make_hw(kind="cuda", bf16=True, memory_gb=80.0))
    backends = [c.backend for c in candidates]
    assert "pytorch_bf16" in backends


def test_bf16_omitted_when_not_supported() -> None:
    candidates = generate_candidates(_make_model(), _make_hw(kind="cuda", bf16=False, memory_gb=24.0))
    backends = [c.backend for c in candidates]
    assert "pytorch_bf16" not in backends


def test_all_candidates_have_non_empty_description() -> None:
    for kind in ("none", "cuda", "mps"):
        candidates = generate_candidates(_make_model(), _make_hw(kind=kind, memory_gb=24.0))
        for cand in candidates:
            assert cand.description.strip() != "", f"Empty description for backend {cand.backend}"


def test_all_candidates_have_valid_device() -> None:
    valid_devices = {"cpu", "cuda", "mps"}
    for kind in ("none", "cuda", "mps"):
        candidates = generate_candidates(_make_model(), _make_hw(kind=kind, memory_gb=24.0))
        for cand in candidates:
            assert cand.device in valid_devices, f"Invalid device {cand.device!r} for backend {cand.backend}"


def test_candidate_id_equals_backend() -> None:
    candidates = generate_candidates(_make_model(), _make_hw(kind="none"))
    for cand in candidates:
        assert cand.id == cand.backend
