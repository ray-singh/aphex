"""Integration tests — full pipeline on real torchvision pretrained models.

Run with:  pytest -m integration
Excluded from the default test run (downloads weights, takes ~2–5 min).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from infermap.benchmark import benchmark_candidate
from infermap.candidates import generate_candidates
from infermap.inspector import inspect_model
from infermap.profiler import profile_hardware
from infermap.recommender import recommend

pytestmark = pytest.mark.integration

INPUT_SHAPE = [3, 224, 224]
WARMUP = 2
ITERS = 10
_CPU_BACKENDS = {"pytorch_fp32", "onnx_cpu", "pytorch_int8_dynamic", "onnx_int8_cpu"}


@pytest.fixture(scope="module")
def hw():
    return profile_hardware()


@pytest.fixture(
    scope="module",
    params=["resnet18", "mobilenet_v2", "efficientnet_b0", "swin_t"],
    ids=["resnet18", "mobilenet_v2", "efficientnet_b0", "swin_t"],
)
def model_and_info(request, tmp_path_factory):
    name = request.param
    if name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18
        m = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    elif name == "mobilenet_v2":
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
        m = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    elif name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).eval()
    else:
        from torchvision.models import Swin_T_Weights, swin_t
        m = swin_t(weights=Swin_T_Weights.DEFAULT).eval()

    path = tmp_path_factory.mktemp("models") / f"{name}.pt"
    torch.save(m, path)
    info = inspect_model(path, input_shape=INPUT_SHAPE)
    return m, info


@pytest.fixture(scope="module")
def cpu_candidates(model_and_info, hw):
    _, info = model_and_info
    all_cands = generate_candidates(info, hw)
    return [c for c in all_cands if c.device == "cpu" and c.backend in _CPU_BACKENDS]


def _run(candidate, model_and_info, batch_size=1, calibration_inputs=None):
    model, info = model_and_info
    return benchmark_candidate(
        candidate, model, info, INPUT_SHAPE,
        batch_size=batch_size, warmup_iters=WARMUP, measure_iters=ITERS,
        timeout_s=None, calibration_inputs=calibration_inputs,
    )


def test_pytorch_fp32_cpu(model_and_info, cpu_candidates):
    cand = next(c for c in cpu_candidates if c.backend == "pytorch_fp32")
    r = _run(cand, model_and_info)
    assert r.ok, r.error
    assert r.latency_p50_ms > 0
    assert r.throughput_rps > 0
    assert r.memory_mb > 0
    assert r.batch_size == 1


def test_onnx_cpu(model_and_info, cpu_candidates):
    cand = next((c for c in cpu_candidates if c.backend == "onnx_cpu"), None)
    if cand is None:
        pytest.skip("onnx_cpu not generated for this hardware profile")
    r = _run(cand, model_and_info)
    assert r.ok, r.error
    assert r.latency_p50_ms > 0


def test_pytorch_int8_dynamic(model_and_info, cpu_candidates):
    cand = next((c for c in cpu_candidates if c.backend == "pytorch_int8_dynamic"), None)
    if cand is None:
        pytest.skip("pytorch_int8_dynamic not generated")
    r = _run(cand, model_and_info)
    assert r.ok, r.error
    assert r.latency_p50_ms > 0


def test_onnx_int8_cpu(model_and_info, cpu_candidates):
    cand = next((c for c in cpu_candidates if c.backend == "onnx_int8_cpu"), None)
    if cand is None:
        pytest.skip("onnx_int8_cpu not generated")
    r = _run(cand, model_and_info)
    assert r.ok, r.error
    assert r.latency_p50_ms > 0


def test_batch_size_stamped_on_result(model_and_info, cpu_candidates):
    cand = next(c for c in cpu_candidates if c.backend == "pytorch_fp32")
    for bs in [1, 4, 8]:
        r = _run(cand, model_and_info, batch_size=bs)
        assert r.ok, r.error
        assert r.batch_size == bs


def test_throughput_increases_with_batch_size(model_and_info, cpu_candidates):
    cand = next(c for c in cpu_candidates if c.backend == "pytorch_fp32")
    r1 = _run(cand, model_and_info, batch_size=1)
    r8 = _run(cand, model_and_info, batch_size=8)
    assert r1.ok and r8.ok
    assert r8.throughput_rps > r1.throughput_rps


def test_recommend_picks_winner(model_and_info, cpu_candidates):
    model, info = model_and_info
    results = [
        benchmark_candidate(
            c, model, info, INPUT_SHAPE,
            batch_size=1, warmup_iters=WARMUP, measure_iters=ITERS, timeout_s=None,
        )
        for c in cpu_candidates
    ]
    assert any(r.ok for r in results), "All candidates failed"
    rec = recommend(results, objective="latency")
    assert rec.result.ok
    assert rec.result.latency_p50_ms > 0
    assert rec.result.throughput_rps > 0
    assert len(rec.rationale) > 0


def test_accuracy_drop_with_calibration(model_and_info, cpu_candidates):
    cand = next((c for c in cpu_candidates if c.backend == "pytorch_int8_dynamic"), None)
    if cand is None:
        pytest.skip("pytorch_int8_dynamic not generated")
    calib = [torch.randn(1, *INPUT_SHAPE) for _ in range(4)]
    r = _run(cand, model_and_info, calibration_inputs=calib)
    assert r.ok, r.error
    assert r.accuracy_drop is not None
    assert 0.0 <= r.accuracy_drop <= 1.0


# ---------------------------------------------------------------------------
# MLP — 1D input, pure Linear layers; tests a distinct architecture family
# and exercises INT8 quantization most aggressively.
# ---------------------------------------------------------------------------

INPUT_SHAPE_MLP = [512]


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@pytest.fixture(scope="module")
def mlp_and_info(tmp_path_factory):
    m = _MLP().eval()
    path = tmp_path_factory.mktemp("models") / "mlp.pt"
    torch.save(m, path)
    info = inspect_model(path, input_shape=INPUT_SHAPE_MLP)
    return m, info


@pytest.fixture(scope="module")
def mlp_cpu_candidates(mlp_and_info, hw):
    _, info = mlp_and_info
    all_cands = generate_candidates(info, hw)
    return [c for c in all_cands if c.device == "cpu" and c.backend in _CPU_BACKENDS]


def _run_mlp(candidate, mlp_and_info, batch_size=1, calibration_inputs=None):
    model, info = mlp_and_info
    return benchmark_candidate(
        candidate, model, info, INPUT_SHAPE_MLP,
        batch_size=batch_size, warmup_iters=WARMUP, measure_iters=ITERS,
        timeout_s=None, calibration_inputs=calibration_inputs,
    )


def test_mlp_fp32_cpu(mlp_and_info, mlp_cpu_candidates):
    cand = next(c for c in mlp_cpu_candidates if c.backend == "pytorch_fp32")
    r = _run_mlp(cand, mlp_and_info)
    assert r.ok, r.error
    assert r.latency_p50_ms > 0
    assert r.throughput_rps > 0


def test_mlp_int8_dynamic(mlp_and_info, mlp_cpu_candidates):
    cand = next((c for c in mlp_cpu_candidates if c.backend == "pytorch_int8_dynamic"), None)
    if cand is None:
        pytest.skip("pytorch_int8_dynamic not generated")
    r = _run_mlp(cand, mlp_and_info)
    assert r.ok, r.error
    assert r.latency_p50_ms > 0


def test_mlp_onnx_int8(mlp_and_info, mlp_cpu_candidates):
    cand = next((c for c in mlp_cpu_candidates if c.backend == "onnx_int8_cpu"), None)
    if cand is None:
        pytest.skip("onnx_int8_cpu not generated")
    r = _run_mlp(cand, mlp_and_info)
    assert r.ok, r.error
    assert r.latency_p50_ms > 0


def test_mlp_int8_accuracy_drop(mlp_and_info, mlp_cpu_candidates):
    cand = next((c for c in mlp_cpu_candidates if c.backend == "pytorch_int8_dynamic"), None)
    if cand is None:
        pytest.skip("pytorch_int8_dynamic not generated")
    calib = [torch.randn(1, *INPUT_SHAPE_MLP) for _ in range(8)]
    r = _run_mlp(cand, mlp_and_info, calibration_inputs=calib)
    assert r.ok, r.error
    assert r.accuracy_drop is not None
    assert 0.0 <= r.accuracy_drop <= 1.0


def test_mlp_recommend(mlp_and_info, mlp_cpu_candidates):
    model, info = mlp_and_info
    results = [
        benchmark_candidate(
            c, model, info, INPUT_SHAPE_MLP,
            batch_size=1, warmup_iters=WARMUP, measure_iters=ITERS, timeout_s=None,
        )
        for c in mlp_cpu_candidates
    ]
    assert any(r.ok for r in results), "All MLP candidates failed"
    rec = recommend(results, objective="latency")
    assert rec.result.ok
    assert rec.result.latency_p50_ms > 0
