"""End-to-end integration test — full pipeline from saved model file to recommendation."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from infermap.benchmark import benchmark_candidate
from infermap.candidates import generate_candidates
from infermap.inspector import inspect_model
from infermap.preflight import run_preflight
from infermap.profiler import profile_hardware
from infermap.recommender import recommend


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.fixture(scope="module")
def saved_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("models") / "tiny_cnn.pt"
    torch.save(TinyCNN(), path)
    return path


def test_full_pipeline_cpu(saved_model_path: Path) -> None:
    # 1. Profile hardware
    hw = profile_hardware()

    # 2. Inspect model from disk
    input_shape = [1, 8, 8]
    info = inspect_model(saved_model_path, input_shape=input_shape)

    assert info.framework == "pytorch"
    assert info.family == "cnn"
    assert info.parameters > 0

    # 3. Pre-flight check
    pf = run_preflight(info, hw)
    assert pf.category != "impossible", f"Pre-flight blocked pipeline: {pf.message}"

    # 4. Generate candidates — restrict to CPU so the test is hardware-agnostic
    all_candidates = generate_candidates(info, hw)
    cpu_candidates = [
        c for c in all_candidates
        if c.device == "cpu" and c.backend != "torch_compile_fp32"
    ]
    assert len(cpu_candidates) > 0, "No CPU candidates generated"

    # 5. Benchmark each CPU candidate (inline, no timeout)
    model = torch.load(saved_model_path, map_location="cpu", weights_only=False)
    model.eval()

    results = [
        benchmark_candidate(
            cand, model, info, input_shape, batch_size=1, warmup_iters=2, measure_iters=5, timeout_s=None
        )
        for cand in cpu_candidates
    ]

    assert any(r.ok for r in results), "All candidates failed"

    # 6. Recommend
    rec = recommend(results, objective="latency")

    assert rec.result.ok
    assert rec.result.latency_p50_ms > 0
    assert rec.result.throughput_rps > 0
    assert len(rec.rationale) > 0
    assert len(rec.all_results) == len(results)


def test_pipeline_respects_latency_constraint(saved_model_path: Path) -> None:
    hw = profile_hardware()
    input_shape = [1, 8, 8]
    info = inspect_model(saved_model_path, input_shape=input_shape)

    all_candidates = generate_candidates(info, hw)
    cpu_candidates = [
        c for c in all_candidates
        if c.device == "cpu" and c.backend != "torch_compile_fp32"
    ]

    model = torch.load(saved_model_path, map_location="cpu", weights_only=False)
    model.eval()

    results = [
        benchmark_candidate(
            cand, model, info, input_shape, batch_size=1, warmup_iters=2, measure_iters=5, timeout_s=None
        )
        for cand in cpu_candidates
    ]

    # Use a very generous constraint so it's always satisfied
    best_latency = min(r.latency_p50_ms for r in results if r.ok)
    rec = recommend(results, objective="latency", max_latency_ms=best_latency * 10)

    assert rec.result.ok
    assert rec.result.latency_p50_ms <= best_latency * 10


def test_preflight_on_huge_model_returns_impossible() -> None:
    """Preflight should block models that can't possibly fit on the current hardware."""
    from infermap.inspector import ModelInfo

    hw = profile_hardware()
    # 500B params at FP32 = ~2.4 TB — impossible on any local device
    huge = ModelInfo(
        framework="pytorch",
        family="transformer",
        parameters=500_000_000_000,
        trainable_parameters=500_000_000_000,
        estimated_memory_fp32_gb=(500_000_000_000 * 4.0 * 1.2) / 1e9,
        estimated_memory_fp16_gb=(500_000_000_000 * 2.0 * 1.2) / 1e9,
    )
    pf = run_preflight(huge, hw)
    assert pf.category == "impossible"
    assert not pf.feasible
