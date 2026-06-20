"""PyTorch plugin — handles .pt / .pth / .pkl files and nn.Module instances."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from infermap.benchmark import BenchmarkResult
from infermap.candidates import DeploymentCandidate
from infermap.inspector import ModelInfo
from infermap.plugin import ModelPlugin
from infermap.profiler import HardwareProfile

_PYTORCH_EXTENSIONS = {".pt", ".pth", ".pkl"}


class PytorchPlugin(ModelPlugin):
    name = "pytorch"

    def can_handle(self, path_or_model: Any) -> bool:
        try:
            import torch.nn as nn
            if isinstance(path_or_model, nn.Module):
                return True
        except ImportError:
            pass
        if isinstance(path_or_model, (str, Path)):
            return Path(path_or_model).suffix.lower() in _PYTORCH_EXTENSIONS
        return False

    def inspect(self, path_or_model: Any, input_shape: list[int] | None = None) -> ModelInfo:
        from infermap.inspector import inspect_model
        return inspect_model(path_or_model, input_shape)

    def load(self, path: Path) -> Any:
        from infermap.inspector import _load_model
        model = _load_model(Path(path))
        model.eval()
        return model

    def generate_candidates(
        self, info: ModelInfo, hardware: HardwareProfile
    ) -> list[DeploymentCandidate]:
        from infermap.candidates import generate_candidates
        return generate_candidates(info, hardware)

    def benchmark(
        self,
        candidate: DeploymentCandidate,
        model: Any,
        info: ModelInfo,
        input_shape: list[int],
        batch_size: int,
        warmup_iters: int,
        measure_iters: int,
        timeout_s: float | None,
        calibration_inputs: list[Any] | None,
    ) -> BenchmarkResult:
        from infermap.benchmark import benchmark_candidate
        return benchmark_candidate(
            candidate, model, info, input_shape, batch_size,
            warmup_iters, measure_iters, timeout_s, calibration_inputs,
        )
