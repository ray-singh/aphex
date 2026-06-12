"""Tests for the benchmark engine."""

import pytest
import torch
import torch.nn as nn

from infermap.benchmark import BenchmarkResult, _prepare, _time_model, benchmark_candidate
from infermap.candidates import DeploymentCandidate
from infermap.inspector import ModelInfo


class TinyMLP(nn.Module):
    """8→4 linear — fast on CPU, ONNX-exportable, no dynamic control flow."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _cpu_fp32_candidate() -> DeploymentCandidate:
    return DeploymentCandidate(
        backend="pytorch_fp32",
        dtype="fp32",
        description="PyTorch FP32 CPU",
        requires_export=False,
        device="cpu",
    )


def _model_info() -> ModelInfo:
    params = 8 * 4 + 4  # Linear(8, 4): 32 weights + 4 biases
    return ModelInfo(
        framework="pytorch",
        family="unknown",
        parameters=params,
        trainable_parameters=params,
        estimated_memory_fp32_gb=(params * 4.0 * 1.2) / 1e9,
        estimated_memory_fp16_gb=(params * 2.0 * 1.2) / 1e9,
    )


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


def test_ok_true_when_no_error() -> None:
    r = BenchmarkResult(
        candidate=_cpu_fp32_candidate(),
        latency_p50_ms=5.0,
        latency_p95_ms=6.0,
        latency_p99_ms=7.0,
        throughput_rps=200.0,
        memory_mb=10.0,
    )
    assert r.ok is True


def test_ok_false_when_error_set() -> None:
    r = BenchmarkResult(
        candidate=_cpu_fp32_candidate(),
        latency_p50_ms=0.0,
        latency_p95_ms=0.0,
        latency_p99_ms=0.0,
        throughput_rps=0.0,
        memory_mb=0.0,
        error="CUDA OOM",
    )
    assert r.ok is False


# ---------------------------------------------------------------------------
# _prepare
# ---------------------------------------------------------------------------


def test_prepare_returns_model_dummy_and_weight_mb() -> None:
    model = TinyMLP()
    cand = _cpu_fp32_candidate()
    prepared, dummy, weight_mb = _prepare(cand, model, [8], batch_size=1, device=torch.device("cpu"))

    assert isinstance(prepared, nn.Module)
    assert dummy.shape == (1, 8)
    assert weight_mb > 0.0


def test_prepare_weight_mb_matches_parameter_bytes() -> None:
    model = TinyMLP()
    cand = _cpu_fp32_candidate()
    _, _, weight_mb = _prepare(cand, model, [8], batch_size=1, device=torch.device("cpu"))

    expected_mb = (36 * 4) / 1e6  # 36 params * 4 bytes / 1e6
    assert abs(weight_mb - expected_mb) < 1e-4


def test_prepare_batch_size_reflected_in_dummy_shape() -> None:
    model = TinyMLP()
    cand = _cpu_fp32_candidate()
    _, dummy, _ = _prepare(cand, model, [8], batch_size=4, device=torch.device("cpu"))

    assert dummy.shape[0] == 4


def test_prepare_fp16_casts_model_and_dummy() -> None:
    model = TinyMLP()
    cand = DeploymentCandidate(
        backend="pytorch_fp16",
        dtype="fp16",
        description="FP16",
        requires_export=False,
        device="cpu",
    )
    prepared, dummy, _ = _prepare(cand, model, [8], batch_size=1, device=torch.device("cpu"))

    assert next(prepared.parameters()).dtype == torch.float16
    assert dummy.dtype == torch.float16


def test_prepare_does_not_mutate_original_model() -> None:
    model = TinyMLP()
    original_dtype = next(model.parameters()).dtype
    cand = DeploymentCandidate(
        backend="pytorch_fp16",
        dtype="fp16",
        description="FP16",
        requires_export=False,
        device="cpu",
    )
    _prepare(cand, model, [8], batch_size=1, device=torch.device("cpu"))

    assert next(model.parameters()).dtype == original_dtype


# ---------------------------------------------------------------------------
# _time_model
# ---------------------------------------------------------------------------


def test_time_model_returns_correct_count() -> None:
    model = TinyMLP()
    dummy = torch.randn(1, 8)
    timings = _time_model(model, dummy, torch.device("cpu"), warmup_iters=3, measure_iters=10)

    assert len(timings) == 10


def test_time_model_all_timings_positive() -> None:
    model = TinyMLP()
    dummy = torch.randn(1, 8)
    timings = _time_model(model, dummy, torch.device("cpu"), warmup_iters=2, measure_iters=5)

    assert all(t > 0 for t in timings)


# ---------------------------------------------------------------------------
# benchmark_candidate — inline path (timeout_s=None avoids subprocess)
# ---------------------------------------------------------------------------


def test_benchmark_cpu_inline_returns_valid_result() -> None:
    result = benchmark_candidate(
        _cpu_fp32_candidate(),
        TinyMLP(),
        _model_info(),
        [8],
        batch_size=1,
        warmup_iters=2,
        measure_iters=5,
        timeout_s=None,
    )

    assert result.ok, f"Unexpected error: {result.error}"
    assert result.latency_p50_ms > 0
    assert result.latency_p95_ms >= result.latency_p50_ms
    assert result.latency_p99_ms >= result.latency_p95_ms
    assert result.throughput_rps > 0
    assert result.memory_mb >= 0


def test_benchmark_onnx_cpu_inline() -> None:
    cand = DeploymentCandidate(
        backend="onnx_cpu",
        dtype="fp32",
        description="ONNX Runtime CPU",
        requires_export=True,
        device="cpu",
    )
    result = benchmark_candidate(
        cand,
        TinyMLP(),
        _model_info(),
        [8],
        batch_size=1,
        warmup_iters=2,
        measure_iters=5,
        timeout_s=None,
    )

    assert result.ok, f"ONNX benchmark failed: {result.error}"
    assert result.latency_p50_ms > 0


def test_benchmark_bad_input_shape_returns_error() -> None:
    # shape [999] doesn't match Linear(8, 4) — should error gracefully
    result = benchmark_candidate(
        _cpu_fp32_candidate(),
        TinyMLP(),
        _model_info(),
        [999],
        batch_size=1,
        warmup_iters=1,
        measure_iters=2,
        timeout_s=None,
    )

    assert not result.ok
    assert result.error is not None
