"""Tests for real accuracy evaluation."""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from infermap.evaluator import (
    EvalDataset,
    _PYTORCH_EVAL_BACKENDS,
    auto_metric,
    compute_drop,
    compute_metric,
    evaluate_baseline,
    fill_accuracy_drop,
    load_eval_data,
    load_infer_fn,
)
from infermap.inspector import ModelInfo


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _info(family: str = "cnn", framework: str = "pytorch") -> ModelInfo:
    return ModelInfo(
        framework=framework,
        family=family,
        parameters=1_000_000,
        trainable_parameters=1_000_000,
        estimated_memory_fp32_gb=0.004,
        estimated_memory_fp16_gb=0.002,
    )


def _eval_dataset(n: int = 10, n_classes: int = 3, metric: str = "accuracy") -> EvalDataset:
    inputs = [torch.zeros(1, 4) for _ in range(n)]
    labels = (list(range(n_classes)) * (n // n_classes + 1))[:n]
    return EvalDataset(inputs=inputs, labels=labels, metric=metric, task="classification")


def _write_infer_fn(tmp_path: Path, body: str, fn_name: str = "predict") -> str:
    """Write a Python module with an inference function and return the spec string."""
    module = tmp_path / "inference.py"
    module.write_text(textwrap.dedent(body))
    return f"{module}:{fn_name}"


# ── auto_metric ───────────────────────────────────────────────────────────────

def test_auto_metric_cnn_returns_accuracy() -> None:
    task, metric = auto_metric(_info(family="cnn"))
    assert task == "classification"
    assert metric == "accuracy"


def test_auto_metric_sklearn_returns_accuracy() -> None:
    task, metric = auto_metric(_info(family="tree_ensemble", framework="sklearn"))
    assert task == "classification"
    assert metric == "accuracy"


def test_auto_metric_llm_returns_perplexity() -> None:
    task, metric = auto_metric(_info(family="llm"))
    assert task == "generation"
    assert metric == "perplexity"


def test_auto_metric_unknown_defaults_to_accuracy() -> None:
    _, metric = auto_metric(_info(family="unknown"))
    assert metric == "accuracy"


# ── compute_metric ────────────────────────────────────────────────────────────

def test_compute_metric_accuracy_perfect() -> None:
    outputs = [np.array([0.9, 0.05, 0.05])] * 5
    labels = [0] * 5
    assert compute_metric(outputs, labels, "accuracy") == pytest.approx(1.0)


def test_compute_metric_accuracy_zero() -> None:
    outputs = [np.array([0.05, 0.9, 0.05])] * 5
    labels = [0] * 5
    assert compute_metric(outputs, labels, "accuracy") == pytest.approx(0.0)


def test_compute_metric_accuracy_partial() -> None:
    outputs = [np.array([0.9, 0.1])] * 3 + [np.array([0.1, 0.9])] * 7
    labels = [0] * 3 + [0] * 7
    assert compute_metric(outputs, labels, "accuracy") == pytest.approx(0.3)


def test_compute_metric_rmse_perfect() -> None:
    outputs = [np.array([2.0])] * 4
    labels = [2.0] * 4
    assert compute_metric(outputs, labels, "rmse") == pytest.approx(0.0)


def test_compute_metric_rmse_nonzero() -> None:
    outputs = [np.array([3.0])] * 4
    labels = [1.0] * 4
    assert compute_metric(outputs, labels, "rmse") == pytest.approx(2.0)


def test_compute_metric_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        compute_metric([np.array([1.0])], [1], "nonsense")


# ── compute_drop ─────────────────────────────────────────────────────────────

def test_compute_drop_accuracy_no_drop() -> None:
    assert compute_drop(0.85, 0.85, "accuracy") == pytest.approx(0.0)


def test_compute_drop_accuracy_positive() -> None:
    assert compute_drop(0.85, 0.80, "accuracy") == pytest.approx(5 / 85, rel=1e-3)


def test_compute_drop_accuracy_improvement_clamped_at_zero() -> None:
    assert compute_drop(0.80, 0.85, "accuracy") == pytest.approx(0.0)


def test_compute_drop_rmse_positive() -> None:
    assert compute_drop(0.5, 0.75, "rmse") == pytest.approx(0.5)


def test_compute_drop_rmse_improvement_clamped_at_zero() -> None:
    assert compute_drop(0.75, 0.5, "rmse") == pytest.approx(0.0)


def test_compute_drop_zero_baseline_returns_zero() -> None:
    assert compute_drop(0.0, 0.5, "accuracy") == pytest.approx(0.0)


# ── load_eval_data ────────────────────────────────────────────────────────────

def test_load_eval_pt_tuple(tmp_path: Path) -> None:
    X = torch.randn(20, 4)
    y = torch.randint(0, 3, (20,))
    path = tmp_path / "eval.pt"
    torch.save((X, y), path)
    ds = load_eval_data(path, _info())
    assert len(ds.inputs) == 20
    assert len(ds.labels) == 20
    assert ds.metric == "accuracy"


def test_load_eval_pt_dict(tmp_path: Path) -> None:
    X = torch.randn(15, 4)
    y = torch.randint(0, 2, (15,))
    path = tmp_path / "eval.pt"
    torch.save({"inputs": X, "labels": y}, path)
    ds = load_eval_data(path, _info())
    assert len(ds.inputs) == 15


def test_load_eval_pt_respects_max_samples(tmp_path: Path) -> None:
    X = torch.randn(600, 4)
    y = torch.randint(0, 2, (600,))
    path = tmp_path / "eval.pt"
    torch.save((X, y), path)
    ds = load_eval_data(path, _info(), max_samples=100)
    assert len(ds.inputs) == 100


def test_load_eval_pt_bad_format_raises(tmp_path: Path) -> None:
    path = tmp_path / "eval.pt"
    torch.save({"wrong": "keys"}, path)
    with pytest.raises(ValueError):
        load_eval_data(path, _info())


def test_load_eval_csv(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "feat1": np.random.randn(30),
        "feat2": np.random.randn(30),
        "label": np.random.randint(0, 3, 30),
    })
    path = tmp_path / "eval.csv"
    df.to_csv(path, index=False)
    ds = load_eval_data(path, _info(framework="sklearn"), label_col="label")
    assert len(ds.inputs) == 30
    assert ds.task == "classification"


def test_load_eval_csv_auto_regression(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "feat1": np.random.randn(50),
        "target": np.random.randn(50),
    })
    path = tmp_path / "eval.csv"
    df.to_csv(path, index=False)
    ds = load_eval_data(path, _info(framework="sklearn"), label_col="target")
    assert ds.task == "regression"
    assert ds.metric == "rmse"


def test_load_eval_csv_missing_label_col_raises(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"feat": [1, 2, 3], "y": [0, 1, 0]})
    path = tmp_path / "eval.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Label column"):
        load_eval_data(path, _info(), label_col="label")


def test_load_eval_unsupported_format_raises(tmp_path: Path) -> None:
    path = tmp_path / "eval.npz"
    path.write_bytes(b"fake")
    with pytest.raises(ValueError, match="Unsupported eval format"):
        load_eval_data(path, _info())


# ── load_infer_fn ─────────────────────────────────────────────────────────────

def test_load_infer_fn_returns_callable(tmp_path: Path) -> None:
    spec = _write_infer_fn(tmp_path, """
        import numpy as np
        def predict(inputs):
            return np.zeros(len(inputs))
    """)
    fn = load_infer_fn(spec)
    assert callable(fn)


def test_load_infer_fn_missing_colon_raises() -> None:
    with pytest.raises(ValueError, match="--infer-fn must be"):
        load_infer_fn("path/to/module.py")


def test_load_infer_fn_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_infer_fn("/nonexistent/module.py:predict")


def test_load_infer_fn_missing_function_raises(tmp_path: Path) -> None:
    module = tmp_path / "inference.py"
    module.write_text("x = 1\n")
    with pytest.raises(AttributeError, match="not found"):
        load_infer_fn(f"{module}:predict")


def test_load_infer_fn_non_callable_raises(tmp_path: Path) -> None:
    spec = _write_infer_fn(tmp_path, "predict = 42\n")
    with pytest.raises(TypeError, match="not callable"):
        load_infer_fn(spec)


def test_load_infer_fn_executes_correctly(tmp_path: Path) -> None:
    spec = _write_infer_fn(tmp_path, """
        import numpy as np
        def predict(inputs):
            return np.array([1, 2, 3])
    """)
    fn = load_infer_fn(spec)
    result = fn([None, None, None])
    assert list(result) == [1, 2, 3]


def test_load_infer_fn_custom_function_name(tmp_path: Path) -> None:
    spec = _write_infer_fn(tmp_path, """
        import numpy as np
        def run_inference(inputs):
            return np.ones(len(inputs))
    """, fn_name="run_inference")
    fn = load_infer_fn(spec)
    assert callable(fn)


# ── evaluate_baseline ─────────────────────────────────────────────────────────

def test_evaluate_baseline_perfect_accuracy(tmp_path: Path) -> None:
    spec = _write_infer_fn(tmp_path, """
        import numpy as np
        def predict(inputs):
            # Always predicts class 0 — matches labels [0, 0, 0, ...]
            return np.zeros(len(inputs), dtype=int)
    """)
    fn = load_infer_fn(spec)
    ds = EvalDataset(
        inputs=[torch.zeros(1, 4)] * 5,
        labels=[0] * 5,
        metric="accuracy",
        task="classification",
    )
    score = evaluate_baseline(fn, ds)
    assert score == pytest.approx(1.0)


def test_evaluate_baseline_zero_accuracy(tmp_path: Path) -> None:
    spec = _write_infer_fn(tmp_path, """
        import numpy as np
        def predict(inputs):
            return np.ones(len(inputs), dtype=int)  # always wrong
    """)
    fn = load_infer_fn(spec)
    ds = EvalDataset(
        inputs=[torch.zeros(1, 4)] * 5,
        labels=[0] * 5,
        metric="accuracy",
        task="classification",
    )
    assert evaluate_baseline(fn, ds) == pytest.approx(0.0)


def test_evaluate_baseline_returns_float(tmp_path: Path) -> None:
    spec = _write_infer_fn(tmp_path, """
        import numpy as np
        def predict(inputs):
            return np.random.randint(0, 3, len(inputs))
    """)
    fn = load_infer_fn(spec)
    ds = _eval_dataset()
    result = evaluate_baseline(fn, ds)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_evaluate_baseline_with_logits(tmp_path: Path) -> None:
    """Function returning 2D logits should work via argmax."""
    spec = _write_infer_fn(tmp_path, """
        import numpy as np
        def predict(inputs):
            n = len(inputs)
            logits = np.zeros((n, 3))
            logits[:, 0] = 10.0  # always predicts class 0
            return logits
    """)
    fn = load_infer_fn(spec)
    ds = EvalDataset(
        inputs=[torch.zeros(1, 4)] * 4,
        labels=[0] * 4,
        metric="accuracy",
        task="classification",
    )
    assert evaluate_baseline(fn, ds) == pytest.approx(1.0)


def test_evaluate_baseline_rmse(tmp_path: Path) -> None:
    spec = _write_infer_fn(tmp_path, """
        import numpy as np
        def predict(inputs):
            return np.full(len(inputs), 3.0)
    """)
    fn = load_infer_fn(spec)
    ds = EvalDataset(
        inputs=[torch.zeros(1, 4)] * 4,
        labels=[1.0] * 4,
        metric="rmse",
        task="regression",
    )
    assert evaluate_baseline(fn, ds) == pytest.approx(2.0)


def test_evaluate_baseline_torch_tensor_output(tmp_path: Path) -> None:
    spec = _write_infer_fn(tmp_path, """
        import torch
        def predict(inputs):
            return torch.zeros(len(inputs), dtype=torch.long)
    """)
    fn = load_infer_fn(spec)
    ds = EvalDataset(
        inputs=[torch.zeros(1, 4)] * 3,
        labels=[0] * 3,
        metric="accuracy",
        task="classification",
    )
    assert evaluate_baseline(fn, ds) == pytest.approx(1.0)


# ── fill_accuracy_drop ────────────────────────────────────────────────────────

def _simple_model() -> torch.nn.Module:
    """Tiny linear model: 4 → 3."""
    m = torch.nn.Linear(4, 3)
    torch.nn.init.eye_(m.weight[:3, :3])
    torch.nn.init.zeros_(m.bias)
    m.eval()
    return m


def _fake_result(backend: str) -> object:
    from infermap.benchmark import BenchmarkResult
    from infermap.candidates import DeploymentCandidate
    cand = DeploymentCandidate(
        backend=backend, dtype="fp32", device="cpu", description=backend,
        requires_export=False,
    )
    return BenchmarkResult(
        candidate=cand,
        latency_p50_ms=1.0, latency_p95_ms=1.2, latency_p99_ms=1.5,
        throughput_rps=100.0, memory_mb=10.0, batch_size=1,
    )


def test_fill_accuracy_drop_pytorch_fp32_only_skips() -> None:
    """Backends not in _PYTORCH_EVAL_BACKENDS should have accuracy_drop=None."""
    r = _fake_result("pytorch_fp32")
    ds = EvalDataset(
        inputs=[torch.randn(1, 4)] * 4,
        labels=[0, 1, 2, 0],
        metric="accuracy",
        task="classification",
    )
    fill_accuracy_drop([r], _simple_model(), _info(), ds)
    assert r.accuracy_drop is None


def test_fill_accuracy_drop_pytorch_fp16_sets_drop() -> None:
    r = _fake_result("pytorch_fp16")
    ds = EvalDataset(
        inputs=[torch.randn(1, 4)] * 6,
        labels=[0, 1, 2, 0, 1, 2],
        metric="accuracy",
        task="classification",
    )
    fill_accuracy_drop([r], _simple_model(), _info(), ds)
    assert r.accuracy_drop is not None
    assert 0.0 <= r.accuracy_drop <= 1.0


def test_fill_accuracy_drop_pytorch_int8_sets_drop() -> None:
    r = _fake_result("pytorch_int8_dynamic")
    ds = EvalDataset(
        inputs=[torch.randn(1, 4)] * 6,
        labels=[0, 1, 2, 0, 1, 2],
        metric="accuracy",
        task="classification",
    )
    fill_accuracy_drop([r], _simple_model(), _info(), ds)
    assert r.accuracy_drop is not None


def test_fill_accuracy_drop_failed_result_skipped() -> None:
    """Results with error should not get accuracy_drop."""
    from infermap.benchmark import BenchmarkResult
    from infermap.candidates import DeploymentCandidate
    cand = DeploymentCandidate(
        backend="pytorch_fp16", dtype="fp16", device="cpu", description="fp16",
        requires_export=False,
    )
    r = BenchmarkResult(
        candidate=cand,
        latency_p50_ms=0.0, latency_p95_ms=0.0, latency_p99_ms=0.0,
        throughput_rps=0.0, memory_mb=0.0, batch_size=1,
        error="timed out",
    )
    ds = EvalDataset(
        inputs=[torch.randn(1, 4)] * 4,
        labels=[0, 1, 2, 0],
        metric="accuracy",
        task="classification",
    )
    fill_accuracy_drop([r], _simple_model(), _info(), ds)
    assert r.accuracy_drop is None


def test_fill_accuracy_drop_non_pytorch_framework_skips() -> None:
    """Non-PyTorch framework with pytorch_fp16 backend — no crash, no mutation."""
    r = _fake_result("pytorch_fp16")
    ds = _eval_dataset()
    fill_accuracy_drop([r], _simple_model(), _info(framework="sklearn"), ds)
    assert r.accuracy_drop is None


def test_fill_accuracy_drop_non_module_skips() -> None:
    """Non-nn.Module passed as model — no crash, no mutation."""
    r = _fake_result("pytorch_fp16")
    ds = _eval_dataset()
    fill_accuracy_drop([r], object(), _info(framework="pytorch"), ds)
    assert r.accuracy_drop is None


def test_fill_accuracy_drop_pytorch_eval_backends_set() -> None:
    expected = {
        "pytorch_fp16", "pytorch_bf16", "pytorch_int8_dynamic",
        "onnx_int8_cpu", "tensorrt_fp16", "tensorrt_int8", "openvino_int8",
        "pytorch_prune_unstructured_30",
        "pytorch_prune_unstructured_50",
        "pytorch_prune_unstructured_70",
        "pytorch_prune_2_4",
    }
    assert expected == _PYTORCH_EVAL_BACKENDS


def test_fill_accuracy_drop_multiple_backends() -> None:
    """Multiple backends in one call — each gets its own drop."""
    r_fp16 = _fake_result("pytorch_fp16")
    r_fp32 = _fake_result("pytorch_fp32")  # not in eval set
    r_int8 = _fake_result("pytorch_int8_dynamic")
    ds = EvalDataset(
        inputs=[torch.randn(1, 4)] * 6,
        labels=[0, 1, 2, 0, 1, 2],
        metric="accuracy",
        task="classification",
    )
    fill_accuracy_drop([r_fp16, r_fp32, r_int8], _simple_model(), _info(), ds)
    assert r_fp16.accuracy_drop is not None
    assert r_fp32.accuracy_drop is None
    assert r_int8.accuracy_drop is not None
