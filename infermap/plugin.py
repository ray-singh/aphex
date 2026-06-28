"""Plugin protocol — base class every model-type adapter must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from infermap.benchmark import BenchmarkResult
from infermap.candidates import DeploymentCandidate
from infermap.inspector import ModelInfo
from infermap.profiler import HardwareProfile


class ModelPlugin(ABC):
    """Base class for all model-type adapters.

    Each plugin handles one framework (PyTorch, sklearn, XGBoost, …).  The core
    engine routes through these methods so CLI and recommender code never import
    framework-specific modules directly.
    """

    name: str  # unique identifier, e.g. "pytorch", "sklearn"

    @abstractmethod
    def can_handle(self, path_or_model: Any) -> bool:
        """Return True if this plugin can load and benchmark the given input."""

    @abstractmethod
    def inspect(self, path_or_model: Any, input_shape: list[int] | None = None) -> ModelInfo:
        """Inspect a model and return its profile."""

    @abstractmethod
    def load(self, path: Path) -> Any:
        """Load a model from disk and return the native model object."""

    @abstractmethod
    def generate_candidates(
        self, info: ModelInfo, hardware: HardwareProfile
    ) -> list[DeploymentCandidate]:
        """Return deployment strategies to benchmark for this model on this hardware."""

    @abstractmethod
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
        input_spec: Any = None,
    ) -> BenchmarkResult:
        """Benchmark a single deployment candidate and return timing + memory results."""
