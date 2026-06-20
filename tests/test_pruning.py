"""Tests for the pruning module and its integration with the benchmark/eval pipeline."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from infermap.benchmark import BenchmarkResult, _is_pruning_backend
from infermap.candidates import DeploymentCandidate
from infermap.evaluator import EvalDataset, _PYTORCH_EVAL_BACKENDS, fill_accuracy_drop
from infermap.inspector import ModelInfo
from infermap.plugins.pytorch import PytorchPlugin
from infermap.pruning import (
    PRUNING_BACKENDS,
    PruneSpec,
    is_pruning_backend,
    prune_model,
    spec_for_backend,
)


def _model_info() -> ModelInfo:
    return ModelInfo(
        framework="pytorch", family="mlp",
        parameters=1, trainable_parameters=1,
        estimated_memory_fp32_gb=0.0, estimated_memory_fp16_gb=0.0,
    )


def _two_layer_mlp() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))


# ── PruneSpec validation ─────────────────────────────────────────────────────


def test_prunespec_rejects_invalid_amount() -> None:
    with pytest.raises(ValueError):
        PruneSpec("unstructured_magnitude", 0.0)
    with pytest.raises(ValueError):
        PruneSpec("unstructured_magnitude", 1.0)
    with pytest.raises(ValueError):
        PruneSpec("unstructured_magnitude", -0.1)


def test_prunespec_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown pruning method"):
        PruneSpec("magic_method")


def test_prunespec_label() -> None:
    assert PruneSpec("unstructured_magnitude", 0.5).label == "prune_unstructured_50"
    assert PruneSpec("unstructured_magnitude", 0.3).label == "prune_unstructured_30"
    assert PruneSpec("structured_2_4").label == "prune_2_4"


# ── prune_model behavior ─────────────────────────────────────────────────────


def test_unstructured_magnitude_achieves_requested_sparsity() -> None:
    m = _two_layer_mlp()
    pruned, report = prune_model(m, PruneSpec("unstructured_magnitude", 0.5))
    # l1_unstructured zeros exactly `amount` fraction per layer.
    assert abs(report.sparsity - 0.5) < 0.01
    assert report.weight_params_total > 0


def test_unstructured_magnitude_70_pct() -> None:
    m = _two_layer_mlp()
    _, r = prune_model(m, PruneSpec("unstructured_magnitude", 0.7))
    assert abs(r.sparsity - 0.7) < 0.01


def test_structured_2_4_produces_50_pct_sparsity() -> None:
    m = _two_layer_mlp()
    _, r = prune_model(m, PruneSpec("structured_2_4"))
    # 2:4 zeros 2/4 of weights in groups divisible by 4.
    assert abs(r.sparsity - 0.5) < 0.05


def test_prune_model_makes_zeros_permanent() -> None:
    """After prune_model the weights should be real zeros, not pruning masks."""
    m = _two_layer_mlp()
    pruned, _ = prune_model(m, PruneSpec("unstructured_magnitude", 0.5))
    for mod in pruned.modules():
        if isinstance(mod, nn.Linear):
            # No pruning machinery left on the module.
            assert not hasattr(mod, "weight_orig")
            assert (mod.weight == 0).sum().item() > 0


def test_prune_model_handles_no_pruneable_layers(caplog) -> None:
    """A model with no Linear/Conv layers gets a warning and unchanged model."""
    m = nn.ReLU()  # no parameters
    pruned, r = prune_model(m, PruneSpec("unstructured_magnitude", 0.5))
    assert r.weight_params_total == 0
    assert r.sparsity == 0.0


def test_prune_model_does_not_break_forward() -> None:
    """A pruned model still forward-passes and produces finite outputs."""
    m = _two_layer_mlp()
    pruned, _ = prune_model(copy.deepcopy(m), PruneSpec("unstructured_magnitude", 0.5))
    out = pruned(torch.randn(4, 16))
    assert out.shape == (4, 4)
    assert torch.isfinite(out).all()


def test_prune_model_returns_modified_model_independent_of_input() -> None:
    """Even though prune_model is destructive, the caller's deepcopy stays intact."""
    m = _two_layer_mlp()
    untouched = copy.deepcopy(m)
    prune_model(m, PruneSpec("unstructured_magnitude", 0.5))
    for p1, p2 in zip(m.parameters(), untouched.parameters()):
        # The original (untouched) deepcopy should NOT be sparse.
        if p2.ndim >= 2:
            assert (p2 == 0).float().mean().item() < 0.05


# ── Catalog ──────────────────────────────────────────────────────────────────


def test_pruning_backend_catalog_is_consistent() -> None:
    expected = {
        "pytorch_prune_unstructured_30",
        "pytorch_prune_unstructured_50",
        "pytorch_prune_unstructured_70",
        "pytorch_prune_2_4",
    }
    assert set(PRUNING_BACKENDS) == expected
    for name in expected:
        assert is_pruning_backend(name)
        spec_for_backend(name)  # must not raise


def test_is_pruning_backend_negative() -> None:
    assert not is_pruning_backend("pytorch_fp16")
    assert not is_pruning_backend("onnx_int8_cpu")
    assert not is_pruning_backend("")


# ── benchmark.py wiring ──────────────────────────────────────────────────────


def test_benchmark_is_pruning_backend_helper() -> None:
    assert _is_pruning_backend("pytorch_prune_unstructured_50")
    assert _is_pruning_backend("pytorch_prune_2_4")
    assert not _is_pruning_backend("pytorch_fp16")


def test_pruning_backends_in_pytorch_eval_set() -> None:
    """The evaluator must know about pruning backends so accuracy_drop is filled."""
    assert "pytorch_prune_unstructured_50" in _PYTORCH_EVAL_BACKENDS
    assert "pytorch_prune_2_4" in _PYTORCH_EVAL_BACKENDS


@pytest.mark.parametrize("backend", [
    "pytorch_prune_unstructured_50",
    "pytorch_prune_2_4",
])
def test_plugin_benchmark_runs_pruning_backend(backend: str) -> None:
    """Pruning backends survive the full plugin.benchmark pipeline."""
    m = _two_layer_mlp()
    cand = DeploymentCandidate(
        backend=backend, dtype="fp32", description=backend,
        requires_export=False, device="cpu",
    )
    r = PytorchPlugin().benchmark(
        cand, m, _model_info(), input_shape=[16],
        batch_size=1, warmup_iters=1, measure_iters=3,
        timeout_s=None, calibration_inputs=None,
    )
    assert r.ok, f"benchmark failed: {r.error}"
    assert r.latency_p50_ms > 0


def test_fill_accuracy_drop_records_drop_for_pruning() -> None:
    """fill_accuracy_drop must populate accuracy_drop for pruning backends."""
    torch.manual_seed(0)
    m = nn.Linear(8, 4)
    inputs = [torch.randn(1, 8) for _ in range(20)]
    with torch.no_grad():
        labels = [int(m(x).argmax(-1).item()) for x in inputs]
    ds = EvalDataset(
        inputs=inputs, labels=labels,
        metric="accuracy", task="classification",
    )

    cand_70 = DeploymentCandidate(
        backend="pytorch_prune_unstructured_70", dtype="fp32",
        description="prune 70", requires_export=False, device="cpu",
    )
    cand_50 = DeploymentCandidate(
        backend="pytorch_prune_unstructured_50", dtype="fp32",
        description="prune 50", requires_export=False, device="cpu",
    )
    r70 = BenchmarkResult(
        candidate=cand_70, latency_p50_ms=1.0, latency_p95_ms=1.0,
        latency_p99_ms=1.0, throughput_rps=1.0, memory_mb=0.0,
    )
    r50 = BenchmarkResult(
        candidate=cand_50, latency_p50_ms=1.0, latency_p95_ms=1.0,
        latency_p99_ms=1.0, throughput_rps=1.0, memory_mb=0.0,
    )

    fill_accuracy_drop([r70, r50], m, _model_info(), ds)

    assert r70.accuracy_drop is not None
    assert r50.accuracy_drop is not None
    assert 0.0 <= r70.accuracy_drop <= 1.0
    assert 0.0 <= r50.accuracy_drop <= 1.0


def test_candidate_generator_emits_pruning_backends() -> None:
    """generate_candidates must include the pruning family on every device path."""
    from infermap.candidates import _cpu_candidates, _cuda_candidates, _mps_candidates
    from infermap.profiler import AcceleratorProfile, CPUProfile, HardwareProfile

    backends_cpu = {c.backend for c in _cpu_candidates()}
    assert "pytorch_prune_unstructured_50" in backends_cpu
    assert "pytorch_prune_2_4" in backends_cpu

    fake_hw = HardwareProfile(
        cpu=CPUProfile(name="x", physical_cores=4, logical_cores=8, ram_gb=8.0),
        accelerator=AcceleratorProfile(
            kind="cuda", name="A100", memory_gb=40.0, bf16=True,
        ),
    )
    backends_cuda = {c.backend for c in _cuda_candidates(fake_hw)}
    assert "pytorch_prune_unstructured_70" in backends_cuda

    fake_hw_mps = HardwareProfile(
        cpu=CPUProfile(name="x", physical_cores=4, logical_cores=8, ram_gb=8.0),
        accelerator=AcceleratorProfile(
            kind="mps", name="M1", memory_gb=8.0, bf16=False,
        ),
    )
    backends_mps = {c.backend for c in _mps_candidates(fake_hw_mps)}
    assert "pytorch_prune_unstructured_30" in backends_mps
