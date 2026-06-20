"""Tests for multi-GPU data-parallel benchmarking support.

The plugin integration test is gated on having >=2 CUDA devices, which CI
usually lacks; the unit tests use a synthetic HardwareProfile so candidate
generation and helper logic are covered regardless.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from infermap.benchmark import (
    _dp_world_size,
    _is_dp_backend,
    _is_pruning_backend,
)
from infermap.candidates import (
    DeploymentCandidate,
    _cuda_candidates,
    _data_parallel_candidates,
    _mps_candidates,
    generate_candidates,
)
from infermap.inspector import ModelInfo
from infermap.plugins.pytorch import PytorchPlugin
from infermap.profiler import AcceleratorProfile, CPUProfile, HardwareProfile


def _hw(*, kind: str = "cuda", device_count: int = 1, bf16: bool = True) -> HardwareProfile:
    return HardwareProfile(
        cpu=CPUProfile(name="x", physical_cores=4, logical_cores=8, ram_gb=16.0),
        accelerator=AcceleratorProfile(
            kind=kind,  # type: ignore[arg-type]
            name="A100" if kind == "cuda" else "M1",
            memory_gb=40.0 if kind == "cuda" else 16.0,
            bf16=bf16,
            device_count=device_count,
        ),
    )


def _info() -> ModelInfo:
    return ModelInfo(
        framework="pytorch", family="mlp",
        parameters=1, trainable_parameters=1,
        estimated_memory_fp32_gb=0.0, estimated_memory_fp16_gb=0.0,
    )


# ── helpers ──────────────────────────────────────────────────────────────────


def test_is_dp_backend_positive() -> None:
    assert _is_dp_backend("pytorch_dp2_fp32")
    assert _is_dp_backend("pytorch_dp8_bf16")


def test_is_dp_backend_negative() -> None:
    assert not _is_dp_backend("pytorch_fp32")
    assert not _is_dp_backend("pytorch_prune_2_4")
    assert not _is_dp_backend("")


def test_dp_world_size_parses_digits() -> None:
    assert _dp_world_size("pytorch_dp2_fp32") == 2
    assert _dp_world_size("pytorch_dp4_fp16") == 4
    assert _dp_world_size("pytorch_dp8_bf16") == 8


def test_dp_world_size_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        _dp_world_size("pytorch_dp_fp32")
    with pytest.raises(ValueError):
        _dp_world_size("pytorch_dp1_fp32")  # world < 2


def test_dp_and_pruning_helpers_dont_overlap() -> None:
    assert not _is_pruning_backend("pytorch_dp2_fp32")
    assert not _is_dp_backend("pytorch_prune_unstructured_50")


# ── candidate generation ────────────────────────────────────────────────────


def test_single_gpu_emits_no_dp_candidates() -> None:
    cands = _data_parallel_candidates(_hw(device_count=1))
    assert cands == []


def test_cpu_only_emits_no_dp_candidates() -> None:
    cands = _data_parallel_candidates(_hw(kind="none", device_count=0))
    assert cands == []


def test_mps_emits_no_dp_candidates() -> None:
    # Even with multi-GPU shenanigans, MPS doesn't get DP variants.
    backends = {c.backend for c in _mps_candidates(_hw(kind="mps", device_count=1))}
    assert not any(b.startswith("pytorch_dp") for b in backends)


def test_two_gpu_emits_dp2_only() -> None:
    cands = _data_parallel_candidates(_hw(device_count=2, bf16=True))
    backends = {c.backend for c in cands}
    assert "pytorch_dp2_fp32" in backends
    assert "pytorch_dp2_fp16" in backends
    assert "pytorch_dp2_bf16" in backends
    assert "pytorch_dp4_fp32" not in backends
    assert "pytorch_dp8_fp32" not in backends


def test_four_gpu_emits_dp2_and_dp4() -> None:
    backends = {c.backend for c in _data_parallel_candidates(_hw(device_count=4))}
    assert {"pytorch_dp2_fp32", "pytorch_dp4_fp32"}.issubset(backends)
    assert "pytorch_dp8_fp32" not in backends


def test_eight_gpu_emits_all_world_sizes() -> None:
    backends = {c.backend for c in _data_parallel_candidates(_hw(device_count=8))}
    for world in (2, 4, 8):
        assert f"pytorch_dp{world}_fp32" in backends
        assert f"pytorch_dp{world}_fp16" in backends


def test_bf16_omitted_when_unsupported() -> None:
    backends = {c.backend for c in _data_parallel_candidates(_hw(device_count=2, bf16=False))}
    assert "pytorch_dp2_fp32" in backends
    assert "pytorch_dp2_fp16" in backends
    assert "pytorch_dp2_bf16" not in backends


def test_dp_candidate_descriptions_mention_batch_constraint() -> None:
    for cand in _data_parallel_candidates(_hw(device_count=2)):
        assert "batch" in cand.description.lower()


def test_cuda_candidates_include_dp_when_multi_gpu() -> None:
    backends = {c.backend for c in _cuda_candidates(_hw(device_count=4))}
    assert "pytorch_dp2_fp16" in backends
    assert "pytorch_dp4_fp16" in backends


def test_generate_candidates_routes_through_cuda() -> None:
    backends = {c.backend for c in generate_candidates(_info(), _hw(device_count=2))}
    assert "pytorch_dp2_fp32" in backends


# ── plugin behavior (no real GPUs needed for the rejection tests) ───────────


def test_dp_backend_rejects_cpu_device() -> None:
    """Asking for a DP backend with device='cpu' should error cleanly."""
    cand = DeploymentCandidate(
        backend="pytorch_dp2_fp32", dtype="fp32", description="dp2",
        requires_export=False, device="cpu",
    )
    r = PytorchPlugin().benchmark(
        cand, nn.Linear(8, 4), _info(), input_shape=[8],
        batch_size=4, warmup_iters=1, measure_iters=2,
        timeout_s=None, calibration_inputs=None,
    )
    assert not r.ok
    assert "CUDA" in (r.error or "")


def test_dp_backend_rejects_small_batch() -> None:
    """batch_size < world_size must be rejected with a useful error."""
    cand = DeploymentCandidate(
        backend="pytorch_dp4_fp32", dtype="fp32", description="dp4",
        requires_export=False, device="cuda",
    )
    r = PytorchPlugin().benchmark(
        cand, nn.Linear(8, 4), _info(), input_shape=[8],
        batch_size=2, warmup_iters=1, measure_iters=2,
        timeout_s=None, calibration_inputs=None,
    )
    # Either path is acceptable: no CUDA at all (CI), or batch-size rejection.
    assert not r.ok


def test_dp_backend_rejects_insufficient_devices() -> None:
    """Asking for dp8 on a host with <8 devices should error."""
    cand = DeploymentCandidate(
        backend="pytorch_dp8_fp32", dtype="fp32", description="dp8",
        requires_export=False, device="cuda",
    )
    r = PytorchPlugin().benchmark(
        cand, nn.Linear(8, 4), _info(), input_shape=[8],
        batch_size=8, warmup_iters=1, measure_iters=2,
        timeout_s=None, calibration_inputs=None,
    )
    assert not r.ok


# ── true integration: real DP on real multi-GPU box ─────────────────────────


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs >=2 CUDA devices",
)
def test_dp2_benchmark_end_to_end() -> None:
    cand = DeploymentCandidate(
        backend="pytorch_dp2_fp32", dtype="fp32", description="dp2",
        requires_export=False, device="cuda",
    )
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    r = PytorchPlugin().benchmark(
        cand, model, _info(), input_shape=[16],
        batch_size=8, warmup_iters=2, measure_iters=10,
        timeout_s=None, calibration_inputs=None,
    )
    assert r.ok, f"DP2 benchmark failed: {r.error}"
    assert r.latency_p50_ms > 0
    assert r.throughput_rps > 0


# ── profiler ────────────────────────────────────────────────────────────────


def test_accelerator_profile_default_device_count_is_one() -> None:
    p = AcceleratorProfile(kind="cuda", name="A100", memory_gb=40.0, bf16=True)
    assert p.device_count == 1
