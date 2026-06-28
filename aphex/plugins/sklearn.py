"""Traditional ML plugin — sklearn, XGBoost, LightGBM, and CatBoost."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from aphex.benchmark import BenchmarkResult
from aphex.candidates import DeploymentCandidate
from aphex.inspector import ModelInfo
from aphex.plugin import ModelPlugin
from aphex.profiler import HardwareProfile

_SKLEARN_EXTENSIONS = frozenset({".joblib"})
_XGBOOST_EXTENSIONS = frozenset({".ubj"})   # .json claimed only after header check
_LIGHTGBM_EXTENSIONS = frozenset({".lgbm"}) # .txt claimed only after header check
_CATBOOST_EXTENSIONS = frozenset({".cbm"})

_UNAMBIGUOUS_EXTENSIONS = (
    _SKLEARN_EXTENSIONS | _XGBOOST_EXTENSIONS | _LIGHTGBM_EXTENSIONS | _CATBOOST_EXTENSIONS
)

_TREE_ENSEMBLE_HINTS = frozenset({
    "forest", "gradient", "adaboost", "extratrees", "bagging",
    "xgbclassifier", "xgbregressor", "xgbranker", "booster",
    "lgbmclassifier", "lgbmregressor", "lgbmranker",
    "catboostclassifier", "catboostregressor", "catboostranker", "catboost",
    "decisiontree",
})
_LINEAR_HINTS = frozenset({
    "linear", "logistic", "ridge", "lasso", "elasticnet", "sgd",
    "perceptron", "passiveaggressive", "huber", "quantile",
})
_SVM_HINTS = frozenset({"svc", "svr", "svm", "nusvc", "nusvr", "linearsvc", "linearsvr"})
_NAIVE_BAYES_HINTS = frozenset({
    "gaussiannb", "bernoullinb", "multinomialnb", "complementnb",
})


class SklearnPlugin(ModelPlugin):
    name = "sklearn"

    def can_handle(self, path_or_model: Any) -> bool:
        if isinstance(path_or_model, (str, Path)):
            path = Path(path_or_model)
            suffix = path.suffix.lower()
            if suffix in _UNAMBIGUOUS_EXTENSIONS:
                return True
            if suffix == ".json":
                return _is_xgboost_json(path)
            if suffix == ".txt":
                return _is_lightgbm_txt(path)
            return False
        return _is_traditional_ml(path_or_model)

    def inspect(self, path_or_model: Any, input_shape: list[int] | None = None) -> ModelInfo:
        if isinstance(path_or_model, (str, Path)):
            model = self.load(Path(path_or_model))
            path_str = str(path_or_model)
        else:
            model = path_or_model
            path_str = None

        framework = _detect_framework(model)
        family = _detect_family(model)
        params = _count_parameters(model, framework)
        mem_gb = _serialized_size_bytes(model, framework) / 1e9

        n_features = _get_n_features(model)
        if input_shape is None and n_features is not None:
            input_shape = [n_features]

        return ModelInfo(
            framework=framework,
            family=family,
            parameters=params,
            trainable_parameters=params,
            estimated_memory_fp32_gb=mem_gb,
            estimated_memory_fp16_gb=mem_gb / 2,
            input_shape=input_shape,
            model_path=path_str,
            num_features=n_features,
            num_trees=_get_num_trees(model, framework),
            max_depth=_get_max_depth(model, framework),
        )

    def load(self, path: Path) -> Any:
        suffix = Path(path).suffix.lower()
        if suffix in _SKLEARN_EXTENSIONS:
            import joblib
            return joblib.load(path)
        if suffix in _XGBOOST_EXTENSIONS or (suffix == ".json" and _is_xgboost_json(path)):
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(str(path))
            return booster
        if suffix in _LIGHTGBM_EXTENSIONS or (suffix == ".txt" and _is_lightgbm_txt(path)):
            import lightgbm as lgb
            return lgb.Booster(model_file=str(path))
        if suffix in _CATBOOST_EXTENSIONS:
            from catboost import CatBoost
            model = CatBoost()
            model.load_model(str(path))
            return model
        import joblib
        return joblib.load(path)

    def generate_candidates(
        self, info: ModelInfo, hardware: HardwareProfile
    ) -> list[DeploymentCandidate]:
        candidates: list[DeploymentCandidate] = []

        candidates.append(DeploymentCandidate(
            backend=f"{info.framework}_predict",
            dtype="float64",
            description=f"{info.framework.capitalize()} native predict",
            requires_export=False,
            device="cpu",
        ))

        if _onnx_converter_available(info.framework):
            candidates.append(DeploymentCandidate(
                backend="onnx_cpu",
                dtype="float32",
                description="ONNX Runtime CPU",
                requires_export=True,
                device="cpu",
            ))

        if info.family == "tree_ensemble" and _treelite_available():
            candidates.append(DeploymentCandidate(
                backend="treelite_cpu",
                dtype="float32",
                description="Treelite compiled (CPU)",
                requires_export=True,
                device="cpu",
            ))

        return candidates

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
        try:
            return _run_benchmark(
                candidate, model, info, input_shape, batch_size,
                warmup_iters, measure_iters,
            )
        except Exception as exc:
            return BenchmarkResult(
                candidate=candidate,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                latency_p99_ms=0.0,
                throughput_rps=0.0,
                memory_mb=0.0,
                batch_size=batch_size,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _run_benchmark(
    candidate: DeploymentCandidate,
    model: Any,
    info: ModelInfo,
    input_shape: list[int],
    batch_size: int,
    warmup_iters: int,
    measure_iters: int,
) -> BenchmarkResult:
    import numpy as np

    n_features = input_shape[0] if input_shape else (_get_n_features(model) or 1)
    X = np.random.randn(batch_size, n_features).astype(np.float32)

    backend = candidate.backend
    if backend.endswith("_predict"):
        run_fn = _prepare_native(model, info.framework, X)
    elif backend == "onnx_cpu":
        run_fn = _prepare_onnx(model, info.framework, X, n_features)
    elif backend == "treelite_cpu":
        run_fn = _prepare_treelite(model, info.framework, X)
    else:
        raise ValueError(f"Unknown backend for sklearn plugin: {backend!r}")

    for _ in range(warmup_iters):
        run_fn()

    timings_ms: list[float] = []
    for _ in range(measure_iters):
        t0 = time.perf_counter()
        run_fn()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    timings_ms.sort()
    n = len(timings_ms)
    p50 = timings_ms[int(n * 0.50)]
    p95 = timings_ms[int(n * 0.95)]
    p99 = timings_ms[int(n * 0.99)]

    return BenchmarkResult(
        candidate=candidate,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        throughput_rps=1000.0 / p50 * batch_size,
        memory_mb=info.estimated_memory_fp32_gb * 1000,
        batch_size=batch_size,
    )


def _prepare_native(model: Any, framework: str, X: Any):
    if framework == "xgboost" and type(model).__name__ == "Booster":
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        return lambda: model.predict(dmatrix)
    return lambda: model.predict(X)


def _prepare_onnx(model: Any, framework: str, X: Any, n_features: int):
    import onnxruntime as ort

    onnx_bytes = _export_to_onnx(model, framework, n_features)
    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    return lambda: sess.run(None, {input_name: X})


def _prepare_treelite(model: Any, framework: str, X: Any):
    import numpy as np
    import treelite

    if framework == "sklearn":
        tl_model = treelite.sklearn.import_model(model)
    elif framework == "xgboost":
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        tl_model = treelite.frontend.from_xgboost(booster)
    elif framework == "lightgbm":
        tl_model = treelite.frontend.from_lightgbm(model)
    else:
        raise ValueError(f"Treelite does not support framework: {framework!r}")

    X_f32 = X.astype(np.float32)

    # Prefer GTIL (Treelite >= 3.0, no compiler required)
    if hasattr(treelite, "gtil"):
        return lambda: treelite.gtil.predict(tl_model, X_f32)

    # Fall back: compile to shared library (requires gcc)
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        libpath = os.path.join(tmpdir, "model.so")
        tl_model.export_lib(toolchain="gcc", libpath=libpath, verbose=False)
        try:
            predictor = treelite.runtime.Predictor(libpath, verbose=False)
        except AttributeError:
            import treelite_runtime
            predictor = treelite_runtime.Predictor(libpath, verbose=False)

    try:
        batch = treelite.runtime.DMatrix(X_f32)
    except AttributeError:
        import treelite_runtime
        batch = treelite_runtime.DMatrix(X_f32)
    return lambda: predictor.predict(batch)


# ---------------------------------------------------------------------------
# ONNX export helpers
# ---------------------------------------------------------------------------

def _export_to_onnx(model: Any, framework: str, n_features: int) -> bytes:
    if framework == "sklearn":
        return _sklearn_to_onnx(model, n_features)
    if framework == "xgboost":
        return _xgboost_to_onnx(model)
    if framework == "lightgbm":
        return _lightgbm_to_onnx(model, n_features)
    if framework == "catboost":
        return _catboost_to_onnx(model)
    raise ValueError(f"No ONNX exporter for framework: {framework!r}")


def _sklearn_to_onnx(model: Any, n_features: int) -> bytes:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    onnx_model = convert_sklearn(
        model,
        initial_types=[("float_input", FloatTensorType([None, n_features]))],
    )
    return onnx_model.SerializeToString()


def _xgboost_to_onnx(model: Any) -> bytes:
    import os
    import tempfile

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.onnx")
        booster.save_model(path)
        with open(path, "rb") as f:
            return f.read()


def _lightgbm_to_onnx(model: Any, n_features: int) -> bytes:
    try:
        from onnxconverter_common.data_types import FloatTensorType
        from onnxmltools import convert_lightgbm

        onnx_model = convert_lightgbm(
            model,
            initial_types=[("input", FloatTensorType([None, n_features]))],
        )
        return onnx_model.SerializeToString()
    except ImportError:
        raise RuntimeError(
            "LightGBM ONNX export requires 'onnxmltools'. "
            "Install with: pip install onnxmltools"
        )


def _catboost_to_onnx(model: Any) -> bytes:
    import io

    buf = io.BytesIO()
    model.save_model(buf, format="onnx")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def _onnx_converter_available(framework: str) -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    checks: dict[str, str] = {
        "sklearn": "skl2onnx",
        "xgboost": "xgboost",   # native ONNX export (>= 1.7)
        "lightgbm": "onnxmltools",
        "catboost": "catboost",  # native ONNX export
    }
    pkg = checks.get(framework)
    if not pkg:
        return False
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def _treelite_available() -> bool:
    try:
        import treelite  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# File header detection
# ---------------------------------------------------------------------------

def _is_xgboost_json(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(32).decode("utf-8", errors="ignore")
        return '"learner"' in header or '"version"' in header
    except Exception:
        return False


def _is_lightgbm_txt(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(10).decode("ascii", errors="ignore")
        return header.startswith("tree")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Framework / family detection
# ---------------------------------------------------------------------------

def _is_traditional_ml(model: Any) -> bool:
    module = type(model).__module__
    return any(module.startswith(fw) for fw in ("sklearn", "xgboost", "lightgbm", "catboost"))


def _detect_framework(model: Any) -> str:
    module = type(model).__module__
    for fw in ("sklearn", "xgboost", "lightgbm", "catboost"):
        if module.startswith(fw):
            return fw
    return "sklearn"


def _detect_family(model: Any) -> str:
    cls = type(model).__name__.lower()
    module = type(model).__module__
    if any(h in cls for h in _TREE_ENSEMBLE_HINTS):
        return "tree_ensemble"
    if any(h in cls for h in _LINEAR_HINTS):
        return "linear"
    if any(h in cls for h in _SVM_HINTS):
        return "svm"
    if any(h in cls for h in _NAIVE_BAYES_HINTS):
        return "naive_bayes"
    if any(module.startswith(fw) for fw in ("xgboost", "lightgbm", "catboost")):
        return "tree_ensemble"
    return "unknown"


# ---------------------------------------------------------------------------
# Parameter counting and memory estimation
# ---------------------------------------------------------------------------

def _count_parameters(model: Any, framework: str) -> int:
    try:
        if framework == "sklearn":
            return _count_sklearn_params(model)
        if framework == "xgboost":
            booster = model.get_booster() if hasattr(model, "get_booster") else model
            return booster.num_trees() * 31  # ~31 nodes for depth=4 (2^5 - 1)
        if framework == "lightgbm":
            return model.num_trees() * model.num_leaves()
        if framework == "catboost":
            depth = model.get_all_params().get("depth", 6)
            return getattr(model, "tree_count_", 0) * (2 ** min(depth, 8))
    except Exception:  # noqa: S110 — best-effort introspection, safe to skip
        pass
    return 0


def _count_sklearn_params(model: Any) -> int:
    try:
        if hasattr(model, "estimators_"):
            estimators = model.estimators_
            # GradientBoosting: estimators_ is 2D array of shape (n_estimators, n_classes)
            if hasattr(estimators[0], "__iter__"):
                return sum(e[0].tree_.node_count for e in estimators)
            return sum(e.tree_.node_count for e in estimators)
        if hasattr(model, "tree_"):
            return int(model.tree_.node_count)
        if hasattr(model, "coef_"):
            n = int(model.coef_.size)
            if hasattr(model, "intercept_"):
                n += int(model.intercept_.size)
            return n
        if hasattr(model, "support_vectors_"):
            return int(model.support_vectors_.size)
    except Exception:  # noqa: S110 — best-effort introspection, safe to skip
        pass
    return 0


def _serialized_size_bytes(model: Any, framework: str) -> int:
    """Estimate model memory from serialized size."""
    import io
    import os
    import tempfile

    try:
        if framework == "xgboost":
            booster = model.get_booster() if hasattr(model, "get_booster") else model
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "m.ubj")
                booster.save_model(p)
                return os.path.getsize(p)
        if framework == "lightgbm":
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "m.txt")
                model.save_model(p)
                return os.path.getsize(p)
        if framework == "catboost":
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "m.cbm")
                model.save_model(p)
                return os.path.getsize(p)
        import joblib
        buf = io.BytesIO()
        joblib.dump(model, buf)
        return buf.tell()
    except Exception:
        return 0


def _get_num_trees(model: Any, framework: str) -> int | None:
    try:
        if framework == "sklearn":
            if hasattr(model, "n_estimators"):
                return int(model.n_estimators)
            if hasattr(model, "estimators_"):
                return len(model.estimators_)
        if framework == "xgboost":
            booster = model.get_booster() if hasattr(model, "get_booster") else model
            return int(booster.num_trees())
        if framework == "lightgbm":
            return int(model.num_trees())
        if framework == "catboost":
            return int(getattr(model, "tree_count_", 0)) or None
    except Exception:  # noqa: S110 — best-effort introspection, safe to skip
        pass
    return None


def _get_max_depth(model: Any, framework: str) -> int | None:
    try:
        if framework == "sklearn" and hasattr(model, "max_depth") and model.max_depth is not None:
            return int(model.max_depth)
        if framework == "xgboost":
            params = model.get_params() if hasattr(model, "get_params") else {}
            booster = model.get_booster() if hasattr(model, "get_booster") else model
            depth = params.get("max_depth") or booster.attributes().get("max_depth")
            if depth is not None:
                return int(depth)
        if framework == "lightgbm":
            return int(model.params.get("max_depth", -1)) or None
        if framework == "catboost":
            return int(model.get_all_params().get("depth", 6))
    except Exception:  # noqa: S110 — best-effort introspection, safe to skip
        pass
    return None


def _get_n_features(model: Any) -> int | None:
    """Return number of input features if detectable from the fitted model."""
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    if hasattr(model, "num_features"):
        try:
            return int(model.num_features())
        except Exception:  # noqa: S110
            pass
    if hasattr(model, "get_booster"):
        try:
            return int(model.get_booster().num_features())
        except Exception:  # noqa: S110
            pass
    if hasattr(model, "num_feature"):
        try:
            return int(model.num_feature())
        except Exception:  # noqa: S110
            pass
    if hasattr(model, "get_feature_names"):
        try:
            return len(model.get_feature_names())
        except Exception:  # noqa: S110
            pass
    return None
