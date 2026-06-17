"""Tests for the sklearn/tree-ensemble plugin."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("sklearn")

from infermap.benchmark import BenchmarkResult
from infermap.candidates import DeploymentCandidate
from infermap.plugins.sklearn import (
    SklearnPlugin,
    _detect_family,
    _detect_framework,
    _get_n_features,
    _is_traditional_ml,
)
from infermap.profiler import AcceleratorProfile, CPUProfile, HardwareProfile


def _hw() -> HardwareProfile:
    cpu = CPUProfile(name="x", physical_cores=4, logical_cores=8, ram_gb=16.0)
    acc = AcceleratorProfile(kind="none", name="none", memory_gb=0.0, bf16=False)
    return HardwareProfile(cpu=cpu, accelerator=acc)


def _logreg(n_features: int = 4) -> Any:
    from sklearn.linear_model import LogisticRegression

    X = np.random.randn(20, n_features)
    y = np.array([0, 1] * 10)
    return LogisticRegression(max_iter=200).fit(X, y)


def _forest(n_features: int = 4) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    X = np.random.randn(20, n_features)
    y = np.array([0, 1] * 10)
    return RandomForestClassifier(n_estimators=2, max_depth=2).fit(X, y)


# ── SklearnPlugin.can_handle ──────────────────────────────────────────────────


class TestCanHandle:
    plugin = SklearnPlugin()

    def test_fitted_sklearn_model(self) -> None:
        assert self.plugin.can_handle(_logreg())

    def test_joblib_path(self, tmp_path: Path) -> None:
        p = tmp_path / "model.joblib"
        p.write_bytes(b"fake")
        assert self.plugin.can_handle(p)

    def test_joblib_path_as_string(self, tmp_path: Path) -> None:
        p = tmp_path / "model.joblib"
        p.write_bytes(b"fake")
        assert self.plugin.can_handle(str(p))

    def test_pt_file_returns_false(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pt"
        p.write_bytes(b"fake")
        assert not self.plugin.can_handle(p)

    def test_gguf_file_returns_false(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        p.write_bytes(b"GGUF")
        assert not self.plugin.can_handle(p)

    def test_integer_returns_false(self) -> None:
        assert not self.plugin.can_handle(42)

    def test_forest_model_handled(self) -> None:
        assert self.plugin.can_handle(_forest())


# ── SklearnPlugin.inspect ─────────────────────────────────────────────────────


class TestInspect:
    plugin = SklearnPlugin()

    def test_logistic_regression_framework(self) -> None:
        info = self.plugin.inspect(_logreg())
        assert info.framework == "sklearn"

    def test_logistic_regression_family(self) -> None:
        info = self.plugin.inspect(_logreg())
        assert info.family == "linear"

    def test_forest_family(self) -> None:
        info = self.plugin.inspect(_forest())
        assert info.family == "tree_ensemble"

    def test_n_features_detected(self) -> None:
        info = self.plugin.inspect(_logreg(n_features=8))
        assert info.num_features == 8

    def test_input_shape_inferred_from_features(self) -> None:
        info = self.plugin.inspect(_logreg(n_features=6))
        assert info.input_shape == [6]

    def test_explicit_input_shape_overrides_detected(self) -> None:
        info = self.plugin.inspect(_logreg(n_features=4), input_shape=[100])
        assert info.input_shape == [100]

    def test_parameters_positive(self) -> None:
        info = self.plugin.inspect(_logreg())
        assert info.parameters > 0

    def test_memory_estimation_non_negative(self) -> None:
        info = self.plugin.inspect(_logreg())
        assert info.estimated_memory_fp32_gb >= 0.0


# ── SklearnPlugin.generate_candidates ────────────────────────────────────────


class TestGenerateCandidates:
    plugin = SklearnPlugin()

    def test_always_includes_native_predict(self) -> None:
        info = self.plugin.inspect(_logreg())
        backends = [c.backend for c in self.plugin.generate_candidates(info, _hw())]
        assert "sklearn_predict" in backends

    def test_native_predict_device_is_cpu(self) -> None:
        info = self.plugin.inspect(_logreg())
        cands = self.plugin.generate_candidates(info, _hw())
        native = next(c for c in cands if c.backend == "sklearn_predict")
        assert native.device == "cpu"

    def test_native_predict_not_requires_export(self) -> None:
        info = self.plugin.inspect(_logreg())
        cands = self.plugin.generate_candidates(info, _hw())
        native = next(c for c in cands if c.backend == "sklearn_predict")
        assert not native.requires_export

    def test_forest_generates_candidates(self) -> None:
        info = self.plugin.inspect(_forest())
        cands = self.plugin.generate_candidates(info, _hw())
        assert len(cands) > 0

    def test_all_candidates_have_non_empty_description(self) -> None:
        info = self.plugin.inspect(_logreg())
        for cand in self.plugin.generate_candidates(info, _hw()):
            assert cand.description.strip()


# ── SklearnPlugin.benchmark ───────────────────────────────────────────────────


def test_benchmark_native_predict_returns_result() -> None:
    plugin = SklearnPlugin()
    model = _logreg(n_features=4)
    info = plugin.inspect(model)
    cand = DeploymentCandidate(
        backend="sklearn_predict",
        dtype="float64",
        description="sklearn native predict",
        requires_export=False,
        device="cpu",
    )
    result = plugin.benchmark(cand, model, info, [4], 4, 1, 5, None, None)
    assert isinstance(result, BenchmarkResult)
    assert result.latency_p50_ms > 0


def test_benchmark_native_predict_throughput_positive() -> None:
    plugin = SklearnPlugin()
    model = _logreg(n_features=4)
    info = plugin.inspect(model)
    cand = DeploymentCandidate(
        backend="sklearn_predict",
        dtype="float64",
        description="sklearn native predict",
        requires_export=False,
        device="cpu",
    )
    result = plugin.benchmark(cand, model, info, [4], 4, 1, 5, None, None)
    assert result.throughput_rps > 0


def test_benchmark_unknown_backend_returns_error() -> None:
    plugin = SklearnPlugin()
    model = _logreg(n_features=4)
    info = plugin.inspect(model)
    cand = DeploymentCandidate(
        backend="nonexistent_backend",
        dtype="float32",
        description="bad backend",
        requires_export=False,
        device="cpu",
    )
    result = plugin.benchmark(cand, model, info, [4], 1, 1, 2, None, None)
    assert result.error is not None


def test_benchmark_uses_input_shape_for_feature_count() -> None:
    plugin = SklearnPlugin()
    model = _logreg(n_features=8)
    info = plugin.inspect(model)
    cand = DeploymentCandidate(
        backend="sklearn_predict",
        dtype="float64",
        description="sklearn native predict",
        requires_export=False,
        device="cpu",
    )
    result = plugin.benchmark(cand, model, info, [8], 1, 1, 3, None, None)
    assert result.ok


# ── Helper functions ──────────────────────────────────────────────────────────


def test_detect_framework_sklearn() -> None:
    assert _detect_framework(_logreg()) == "sklearn"


def test_detect_family_linear() -> None:
    assert _detect_family(_logreg()) == "linear"


def test_detect_family_tree_ensemble() -> None:
    assert _detect_family(_forest()) == "tree_ensemble"


def test_is_traditional_ml_true_for_sklearn() -> None:
    assert _is_traditional_ml(_logreg())


def test_is_traditional_ml_false_for_nn_module() -> None:
    import torch.nn as nn

    assert not _is_traditional_ml(nn.Linear(4, 2))


def test_get_n_features_fitted_model() -> None:
    assert _get_n_features(_logreg(n_features=5)) == 5


def test_get_n_features_returns_none_for_unfitted() -> None:
    from sklearn.linear_model import LogisticRegression

    assert _get_n_features(LogisticRegression()) is None


def test_detect_svm_family() -> None:
    from sklearn.svm import SVC

    X = np.random.randn(20, 4)
    y = np.array([0, 1] * 10)
    model = SVC().fit(X, y)
    assert _detect_family(model) == "svm"
