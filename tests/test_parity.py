"""Tests for aphex.parity — artifact-vs-source numerical parity checks."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from aphex.converter import convert
from aphex.parity import ParityResult, verify_artifact

_HAS_ONNX = importlib.util.find_spec("onnxruntime") is not None
_INPUT_SHAPE = [4]
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _classifier() -> nn.Module:
    m = nn.Linear(4, 3, bias=False)
    nn.init.eye_(m.weight[:3, :3])
    return m.eval()


def _regressor() -> nn.Module:
    return nn.Linear(4, 1, bias=False).eval()


def _samples(n: int = 4, dim: int = 4) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(1, dim) for _ in range(n)]


# ── happy paths ─────────────────────────────────────────────────────────────


def test_pytorch_fp32_is_near_exact(tmp_path: Path) -> None:
    model = _classifier()
    written = convert(model, "pytorch_fp32", _INPUT_SHAPE, tmp_path / "m.pt")
    res = verify_artifact(model, "pytorch_fp32", written, _INPUT_SHAPE, _samples())
    assert res.passed
    assert not res.skipped
    assert res.cosine_sim > 0.99999
    assert res.top1_agreement == 1.0


@pytest.mark.skipif(not _HAS_ONNX, reason="onnxruntime not installed")
def test_onnx_cpu_passes(tmp_path: Path) -> None:
    model = _classifier()
    written = convert(model, "onnx_cpu", _INPUT_SHAPE, tmp_path / "m.onnx")
    res = verify_artifact(model, "onnx_cpu", written, _INPUT_SHAPE, _samples())
    assert res.passed
    assert res.cosine_sim > 0.999
    assert res.max_abs_diff < 1e-3


@pytest.mark.skipif(not _HAS_ONNX, reason="onnxruntime not installed")
def test_onnx_int8_tolerates_quantization_noise(tmp_path: Path) -> None:
    # A real (non-corrupt) int8 export drifts in absolute value but stays
    # directionally faithful — parity must accept it.
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3)).eval()
    written = convert(model, "onnx_int8_cpu", _INPUT_SHAPE, tmp_path / "m.onnx")
    res = verify_artifact(model, "onnx_int8_cpu", written, _INPUT_SHAPE, _samples())
    assert res.passed
    assert res.cosine_sim >= res.cosine_threshold


# ── corruption detection (the point of the feature) ──────────────────────────


@pytest.mark.skipif(not _HAS_ONNX, reason="onnxruntime not installed")
def test_detects_corrupt_artifact(tmp_path: Path) -> None:
    source = _classifier()
    # Artifact built from a sign-flipped model: cosine ≈ -1, top-1 disagrees.
    bad = nn.Linear(4, 3, bias=False)
    bad.weight.data = -source.weight.data.clone()
    written = convert(bad.eval(), "onnx_cpu", _INPUT_SHAPE, tmp_path / "bad.onnx")

    res = verify_artifact(source, "onnx_cpu", written, _INPUT_SHAPE, _samples())
    assert not res.passed
    assert not res.skipped
    assert res.cosine_sim < res.cosine_threshold


# ── graceful skips ───────────────────────────────────────────────────────────


def test_unavailable_backend_skips_without_failing(tmp_path: Path) -> None:
    # TensorRT isn't runnable without CUDA: parity must skip, not fail.
    res = verify_artifact(
        _classifier(), "tensorrt_fp16", [tmp_path / "x.engine"], _INPUT_SHAPE, _samples()
    )
    assert res.skipped
    assert res.passed  # skipped never blocks a conversion
    assert res.reason


def test_missing_artifact_skips(tmp_path: Path) -> None:
    res = verify_artifact(_classifier(), "onnx_cpu", [], _INPUT_SHAPE, _samples())
    assert res.skipped


# ── metric semantics ─────────────────────────────────────────────────────────


def test_regression_output_has_no_top1(tmp_path: Path) -> None:
    model = _regressor()
    written = convert(model, "pytorch_fp32", _INPUT_SHAPE, tmp_path / "r.pt")
    res = verify_artifact(model, "pytorch_fp32", written, _INPUT_SHAPE, _samples())
    assert res.top1_agreement is None
    assert res.passed


def test_summary_strings() -> None:
    ok = ParityResult(backend="onnx_cpu", passed=True, n_samples=4, cosine_sim=1.0)
    assert "ok" in ok.summary()
    skipped = ParityResult(
        backend="tensorrt_fp16", passed=True, n_samples=0, skipped=True, reason="no CUDA"
    )
    assert "skipped" in skipped.summary()
    failed = ParityResult(backend="onnx_cpu", passed=False, n_samples=4, cosine_sim=0.1)
    assert "FAILED" in failed.summary()


# ── real-model coverage (resnet18.pt in the repo root) ───────────────────────


@pytest.mark.skipif(
    not _HAS_ONNX or not (_REPO_ROOT / "resnet18.pt").exists(),
    reason="needs onnxruntime and resnet18.pt",
)
def test_resnet18_onnx_parity() -> None:
    model = torch.jit.load(str(_REPO_ROOT / "resnet18.pt"), map_location="cpu").eval()
    shape = [3, 224, 224]
    samples = [torch.randn(1, *shape) for _ in range(2)]
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        written = convert(model, "onnx_cpu", shape, Path(td) / "rn.onnx")
        res = verify_artifact(model, "onnx_cpu", written, shape, samples)
    assert res.passed
    assert res.top1_agreement == 1.0


# ── multi-input models ────────────────────────────────────────────────────────


class _TwoInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(6, 8)
        self.head = nn.Linear(8, 3)

    def forward(self, x, y):  # noqa: ANN001
        return self.head(torch.relu(self.a(x) + self.b(y)))


@pytest.mark.skipif(not _HAS_ONNX, reason="onnxruntime not installed")
def test_multi_input_onnx_parity_passes(tmp_path: Path) -> None:
    from aphex.inputspec import InputSpec

    model = _TwoInput().eval()
    spec = InputSpec.parse("x:4;y:6")
    written = convert(model, "onnx_cpu", spec.primary_shape, tmp_path / "m.onnx", input_spec=spec)
    res = verify_artifact(model, "onnx_cpu", written, spec.primary_shape, input_spec=spec)
    assert res.passed
    assert res.cosine_sim > 0.999


@pytest.mark.skipif(not _HAS_ONNX, reason="onnxruntime not installed")
def test_multi_input_onnx_detects_corruption(tmp_path: Path) -> None:
    from aphex.inputspec import InputSpec

    source = _TwoInput().eval()
    bad = _TwoInput().eval()
    for p in bad.parameters():
        p.data = -p.data
    spec = InputSpec.parse("x:4;y:6")
    written = convert(bad, "onnx_cpu", spec.primary_shape, tmp_path / "bad.onnx", input_spec=spec)
    res = verify_artifact(source, "onnx_cpu", written, spec.primary_shape, input_spec=spec)
    assert not res.passed


def test_multi_input_pytorch_parity_passes(tmp_path: Path) -> None:
    from aphex.inputspec import InputSpec

    model = _TwoInput().eval()
    spec = InputSpec.parse("x:4;y:6")
    written = convert(model, "pytorch_fp32", spec.primary_shape, tmp_path / "m.pt", input_spec=spec)
    res = verify_artifact(model, "pytorch_fp32", written, spec.primary_shape, input_spec=spec)
    assert res.passed
