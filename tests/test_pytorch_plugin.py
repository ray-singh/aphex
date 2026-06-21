"""Tests for the PyTorch plugin."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from infermap.benchmark import BenchmarkResult
from infermap.candidates import DeploymentCandidate
from infermap.plugins.pytorch import _PYTORCH_EXTENSIONS, PytorchPlugin
from infermap.profiler import AcceleratorProfile, CPUProfile, HardwareProfile


def _tiny_model() -> nn.Module:
    return nn.Linear(4, 2, bias=False).eval()


def _hw() -> HardwareProfile:
    cpu = CPUProfile(name="x", physical_cores=4, logical_cores=8, ram_gb=16.0)
    acc = AcceleratorProfile(kind="none", name="none", memory_gb=0.0, bf16=False)
    return HardwareProfile(cpu=cpu, accelerator=acc)


# ── PytorchPlugin.can_handle ──────────────────────────────────────────────────


class TestCanHandle:
    plugin = PytorchPlugin()

    def test_pt_file_path(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pt"
        torch.save(_tiny_model(), p)
        assert self.plugin.can_handle(p)

    def test_pth_file_path(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pth"
        torch.save(_tiny_model(), p)
        assert self.plugin.can_handle(p)

    def test_pkl_file_path(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pkl"
        p.write_bytes(b"fake")
        assert self.plugin.can_handle(p)

    def test_pt_as_string(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pt"
        torch.save(_tiny_model(), p)
        assert self.plugin.can_handle(str(p))

    def test_nn_module_instance(self) -> None:
        assert self.plugin.can_handle(_tiny_model())

    def test_gguf_file_returns_false(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        p.write_bytes(b"GGUF")
        assert not self.plugin.can_handle(p)

    def test_joblib_file_returns_false(self, tmp_path: Path) -> None:
        p = tmp_path / "model.joblib"
        p.write_bytes(b"fake")
        assert not self.plugin.can_handle(p)

    def test_integer_returns_false(self) -> None:
        assert not self.plugin.can_handle(42)

    def test_pytorch_extensions_set_is_complete(self) -> None:
        assert ".pt" in _PYTORCH_EXTENSIONS
        assert ".pth" in _PYTORCH_EXTENSIONS
        assert ".pkl" in _PYTORCH_EXTENSIONS


# ── PytorchPlugin.load ────────────────────────────────────────────────────────


class TestLoad:
    plugin = PytorchPlugin()

    def test_load_returns_nn_module(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pt"
        torch.save(_tiny_model(), p)
        model = self.plugin.load(p)
        assert isinstance(model, nn.Module)

    def test_loaded_model_is_in_eval_mode(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pt"
        torch.save(_tiny_model().train(), p)
        model = self.plugin.load(p)
        assert not model.training

    def test_load_pth_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pth"
        torch.save(_tiny_model(), p)
        assert isinstance(self.plugin.load(p), nn.Module)


# ── PytorchPlugin.inspect ─────────────────────────────────────────────────────


class TestInspect:
    plugin = PytorchPlugin()

    def test_framework_is_pytorch(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pt"
        torch.save(_tiny_model(), p)
        info = self.plugin.inspect(p, input_shape=[4])
        assert info.framework == "pytorch"

    def test_model_path_is_set(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pt"
        torch.save(_tiny_model(), p)
        info = self.plugin.inspect(p, input_shape=[4])
        assert info.model_path is not None

    def test_parameters_positive(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pt"
        torch.save(_tiny_model(), p)
        info = self.plugin.inspect(p, input_shape=[4])
        assert info.parameters > 0

    def test_inspect_nn_module_directly(self) -> None:
        info = self.plugin.inspect(_tiny_model(), input_shape=[4])
        assert info.framework == "pytorch"


# ── PytorchPlugin.generate_candidates ────────────────────────────────────────


class TestGenerateCandidates:
    plugin = PytorchPlugin()

    def test_returns_non_empty_list(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pt"
        torch.save(_tiny_model(), p)
        info = self.plugin.inspect(p, input_shape=[4])
        candidates = self.plugin.generate_candidates(info, _hw())
        assert len(candidates) > 0

    def test_all_candidates_have_description(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pt"
        torch.save(_tiny_model(), p)
        info = self.plugin.inspect(p, input_shape=[4])
        candidates = self.plugin.generate_candidates(info, _hw())
        assert all(c.description.strip() for c in candidates)

    def test_includes_pytorch_fp32(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pt"
        torch.save(_tiny_model(), p)
        info = self.plugin.inspect(p, input_shape=[4])
        backends = [c.backend for c in self.plugin.generate_candidates(info, _hw())]
        assert "pytorch_fp32" in backends


# ── PytorchPlugin.benchmark ───────────────────────────────────────────────────


def test_benchmark_pytorch_fp32_returns_result(tmp_path: Path) -> None:
    plugin = PytorchPlugin()
    p = tmp_path / "model.pt"
    torch.save(_tiny_model(), p)
    info = plugin.inspect(p, input_shape=[4])
    model = plugin.load(p)
    cand = DeploymentCandidate(
        backend="pytorch_fp32",
        dtype="fp32",
        description="PyTorch FP32 (CPU)",
        requires_export=False,
        device="cpu",
    )
    result = plugin.benchmark(cand, model, info, [4], 1, 2, 5, None, None)
    assert isinstance(result, BenchmarkResult)
    assert result.latency_p50_ms >= 0


def test_benchmark_result_ok_for_valid_candidate(tmp_path: Path) -> None:
    plugin = PytorchPlugin()
    p = tmp_path / "model.pt"
    torch.save(_tiny_model(), p)
    info = plugin.inspect(p, input_shape=[4])
    model = plugin.load(p)
    cand = DeploymentCandidate(
        backend="pytorch_fp32",
        dtype="fp32",
        description="PyTorch FP32 (CPU)",
        requires_export=False,
        device="cpu",
    )
    result = plugin.benchmark(cand, model, info, [4], 1, 2, 5, None, None)
    assert result.ok
