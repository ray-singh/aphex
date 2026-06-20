"""Real accuracy evaluation using a user-supplied inference callable.

The user provides the complete inference pipeline (preprocessing + model +
postprocessing) as a Python function. aphex calls it on labelled eval data
and computes a task-appropriate metric. No auto-detection of model internals.
"""

from __future__ import annotations

import contextlib
import copy as _copy
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np

from infermap.inspector import ModelInfo

logger = logging.getLogger("infermap.evaluator")

_MAX_EVAL_SAMPLES = 512
_CLOUD_SCHEMES = ("s3://", "gs://", "az://")

_HIGHER_IS_BETTER = frozenset({"accuracy", "f1_macro", "f1_weighted"})
_LOWER_IS_BETTER = frozenset({"rmse", "mae"})

def _trust_pickle() -> bool:
    """Allow weights_only=False torch.load only when the user opts in.

    Set APHEX_TRUST_PICKLE=1 to enable pickle-based loading (required for
    arbitrary nn.Module subclasses, but executes whatever pickled code is
    in the file). Default is False, which loads only tensors/state_dicts.
    """
    return os.environ.get("APHEX_TRUST_PICKLE", "").lower() in ("1", "true", "yes")


def _looks_like_pickled_module(exc: BaseException) -> bool:
    """Heuristic: does this weights_only=True failure look like a pickled object
    (nn.Module, custom class) vs a corrupt / unsupported file?

    torch raises ``_pickle.UnpicklingError`` / ``WeightsUnpickler`` errors when a
    .pt file references a non-tensor class under weights_only=True. Detect that
    family of error so we can guide the user toward state_dict / TRUST_PICKLE
    rather than print a confusing low-level trace.
    """
    msg = str(exc).lower()
    return (
        "weightsunpickler" in msg
        or "unsupported global" in msg
        or "weights_only" in msg
        or "globalrestricted" in msg
        or exc.__class__.__name__ == "UnpicklingError"
    )


def _torch_load(path: Any) -> Any:
    """Load a torch file with the safest acceptable mode.

    Strategy:
      1. Try ``weights_only=True`` — works for tensors, state_dicts, dicts of
         tensors, and typical eval/calibration payloads.
      2. If that fails *and* the failure looks like a pickled-class issue, fall
         back to ``weights_only=False`` only when APHEX_TRUST_PICKLE=1 is set,
         otherwise raise a friendly error with the actionable options.
      3. If the failure looks unrelated (file corruption, missing module, etc.)
         re-raise the original exception as-is — the user needs to see it.
    """
    import torch

    try:
        return torch.load(path, weights_only=True)
    except Exception as safe_exc:
        if not _looks_like_pickled_module(safe_exc):
            raise
        if _trust_pickle():
            logger.warning(
                "Loading %s with weights_only=False (APHEX_TRUST_PICKLE=1). "
                "This executes pickled code; only do this for trusted files.",
                path,
            )
            return torch.load(path, weights_only=False)
        raise RuntimeError(
            f"{path!r} contains pickled objects (likely a full nn.Module or a "
            "custom class) that cannot be loaded under PyTorch's safe "
            "weights_only=True mode.\n"
            "Options:\n"
            "  (a) Re-save as a state_dict / dict of tensors: "
            "torch.save(model.state_dict(), 'eval.pt')\n"
            "  (b) Trust the file and set APHEX_TRUST_PICKLE=1 — WARNING: "
            "pickle execution runs arbitrary code; only do this for files "
            "you produced or fully trust.\n"
            f"Underlying error: {safe_exc}"
        ) from safe_exc


METRIC_TASK: dict[str, str] = {
    "accuracy": "classification",
    "f1_macro": "classification",
    "f1_weighted": "classification",
    "rmse": "regression",
    "mae": "regression",
}


def _is_cloud_uri(path: str) -> bool:
    return isinstance(path, str) and any(path.startswith(s) for s in _CLOUD_SCHEMES)


def _download_cloud_uri(uri: str, tmp_dir: Path) -> Path:
    """Download a cloud eval URI (file or prefix) into tmp_dir; return the local path."""
    if uri.startswith("s3://"):
        return _download_s3_eval(uri, tmp_dir)
    if uri.startswith("gs://"):
        return _download_gcs_eval(uri, tmp_dir)
    raise ValueError(
        f"Unsupported cloud URI {uri!r}. Supported schemes: s3://, gs://"
    )


def _download_s3_eval(uri: str, tmp_dir: Path) -> Path:
    try:
        import boto3
    except ImportError:
        raise ImportError("S3 eval data requires boto3: pip install boto3")

    parsed = urlparse(uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    s3 = boto3.client("s3")

    # Single file: has an extension and doesn't end with "/"
    if not key.endswith("/") and "." in Path(key).name:
        dest = tmp_dir / Path(key).name
        s3.download_file(bucket, key, str(dest))
        return dest

    # Directory prefix: list and download all objects
    prefix = key.rstrip("/") + "/"
    local_root = tmp_dir / Path(key.rstrip("/")).name
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            relative = obj["Key"][len(prefix):]
            if not relative:
                continue
            dest = local_root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(dest))
    return local_root


def _download_gcs_eval(uri: str, tmp_dir: Path) -> Path:
    try:
        from google.cloud import storage as gcs
    except ImportError:
        raise ImportError(
            "GCS eval data requires google-cloud-storage: "
            "pip install google-cloud-storage"
        )

    parsed = urlparse(uri)
    bucket_name, obj_path = parsed.netloc, parsed.path.lstrip("/")
    client = gcs.Client()
    bucket = client.bucket(bucket_name)

    if not obj_path.endswith("/") and "." in Path(obj_path).name:
        dest = tmp_dir / Path(obj_path).name
        bucket.blob(obj_path).download_to_filename(str(dest))
        return dest

    prefix = obj_path.rstrip("/") + "/"
    local_root = tmp_dir / Path(obj_path.rstrip("/")).name
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        relative = blob.name[len(prefix):]
        if not relative:
            continue
        dest = local_root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
    return local_root


def infer_task_from_labels(labels: list) -> str:
    """Infer classification vs regression from label distribution."""
    if not labels:
        return "classification"
    arr = np.array(labels)
    is_int_like = np.issubdtype(arr.dtype, np.integer) or (
        np.issubdtype(arr.dtype, np.floating) and np.all(arr == arr.astype(int))
    )
    return "classification" if is_int_like and len(np.unique(arr)) <= 50 else "regression"


def validate_metric_for_task(metric: str, task: str) -> None:
    """Raise ValueError if metric doesn't match the detected task."""
    expected = METRIC_TASK.get(metric)
    if expected is not None and expected != task:
        label_type = "continuous" if task == "regression" else "discrete"
        raise ValueError(
            f"--max-{metric}-loss is for {expected} tasks, but the eval dataset "
            f"has {label_type} labels (detected task: {task!r}). "
            f"Check that you're using the right metric for your model."
        )


@dataclass
class EvalDataset:
    inputs: list        # one item per sample; type matches what the eval file contains
    labels: list        # one label per sample
    metric: str         # "accuracy", "f1_macro", "f1_weighted", "rmse", "mae"
    task: str           # "classification", "regression", "generation"


# ── Loading ───────────────────────────────────────────────────────────────────

def load_eval_data(
    path: Path | str,
    info: ModelInfo,
    label_col: str = "label",
    max_samples: int = _MAX_EVAL_SAMPLES,
    metric_override: str | None = None,
) -> EvalDataset:
    """Load labelled eval data.

    Supported formats:
      .pt   — (inputs, labels) tuple or {"inputs": ..., "labels": ...} dict
      .csv / .parquet — tabular; label_col is the target column
      directory — ImageNet-style class subdirs (cat/, dog/, …)
      s3://bucket/key or gs://bucket/key — downloaded to a temp dir, then loaded

    If *metric_override* is given (e.g. "mae", "f1_macro"), it is validated
    against the inferred task from the label distribution and used instead of
    the auto-detected metric.  Raises ValueError on mismatch.
    """
    task, metric = auto_metric(info)

    with contextlib.ExitStack() as stack:
        if _is_cloud_uri(str(path)):
            tmp_dir = Path(stack.enter_context(
                tempfile.TemporaryDirectory(prefix="aphex-eval-")
            ))
            path = _download_cloud_uri(str(path), tmp_dir)
        else:
            path = Path(path)

        if path.is_dir():
            dataset = _load_image_dir(path, info, max_samples, task, metric)
        elif path.suffix.lower() == ".pt":
            dataset = _load_pt(path, max_samples, task, metric)
        elif path.suffix.lower() in (".parquet", ".csv"):
            dataset = _load_tabular(path, label_col, max_samples, task, metric)
        else:
            raise ValueError(
                f"Unsupported eval format {path.suffix.lower()!r}. "
                f"Supported: .pt, .parquet, .csv, or an image directory with class subdirs."
            )

    inferred_task = infer_task_from_labels(dataset.labels)
    if metric_override is not None:
        validate_metric_for_task(metric_override, inferred_task)
        dataset.task = inferred_task
        dataset.metric = metric_override
    elif inferred_task != dataset.task:
        dataset.task = inferred_task
        dataset.metric = "rmse" if inferred_task == "regression" else "accuracy"

    return dataset


def _load_pt(path: Path, max_samples: int, task: str, metric: str) -> EvalDataset:
    import torch

    raw = _torch_load(path)

    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        X, y = raw
    elif isinstance(raw, dict):
        X = next((raw[k] for k in ("inputs", "X", "x") if k in raw), None)
        y = next((raw[k] for k in ("labels", "y", "targets") if k in raw), None)
        if X is None or y is None:
            raise ValueError(
                f"Expected keys 'inputs'/'X' and 'labels'/'y', got: {list(raw.keys())}"
            )
    else:
        raise ValueError(
            "Eval .pt file must be a (inputs, labels) tuple or a dict with 'inputs'/'labels' keys."
        )

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    n = min(len(X), max_samples)
    inputs = [X[i : i + 1] for i in range(n)]
    labels = [y[i].item() if hasattr(y[i], "item") and y[i].numel() == 1 else y[i] for i in range(n)]
    return EvalDataset(inputs=inputs, labels=labels, metric=metric, task=task)


def _load_tabular(
    path: Path, label_col: str, max_samples: int, task: str, metric: str
) -> EvalDataset:
    import torch

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore[import]
            df = pd.read_parquet(path)
        except ImportError:
            raise ImportError("pandas and pyarrow are required to read .parquet files.")
    else:
        try:
            import pandas as pd  # type: ignore[import]
            df = pd.read_csv(path)
        except ImportError:
            raise ImportError("pandas is required to read .csv files.")

    if label_col not in df.columns:
        raise ValueError(
            f"Label column {label_col!r} not found. "
            f"Available columns: {list(df.columns)}. "
            f"Use --eval-label-col to specify the correct column."
        )

    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values

    # Auto-detect regression from continuous float labels
    if np.issubdtype(y.dtype, np.floating) and len(np.unique(y)) > 20:
        task, metric = "regression", "rmse"

    n = min(len(X), max_samples)
    inputs = [torch.from_numpy(X[i : i + 1]) for i in range(n)]
    labels = [float(y[i]) if task == "regression" else int(y[i]) for i in range(n)]
    return EvalDataset(inputs=inputs, labels=labels, metric=metric, task=task)


def _load_image_dir(
    path: Path, info: ModelInfo, max_samples: int, task: str, metric: str
) -> EvalDataset:
    """Load images from class subdirectories (ImageNet-style)."""
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image  # type: ignore[import]

    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    class_dirs = sorted(d for d in path.iterdir() if d.is_dir())
    if not class_dirs:
        raise ValueError(
            f"{path} has no subdirectories. "
            f"Image eval expects class subdirs: {path}/cat/*.jpg, {path}/dog/*.jpg"
        )

    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    input_shape = info.input_shape or [3, 224, 224]
    c = input_shape[0] if len(input_shape) >= 1 else 3
    h = input_shape[1] if len(input_shape) >= 2 else 224
    w = input_shape[2] if len(input_shape) >= 3 else 224

    inputs, labels = [], []
    for class_dir in class_dirs:
        cls_idx = class_to_idx[class_dir.name]
        files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS)
        for f in files:
            if len(inputs) >= max_samples:
                break
            try:
                img = Image.open(f).convert("RGB" if c == 3 else "L")
                t = TF.to_tensor(img)
                t = TF.resize(t, [h, w], antialias=True)
                if c == 3:
                    t = TF.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD)
                inputs.append(t.unsqueeze(0))
                labels.append(cls_idx)
            except Exception as exc:
                logger.debug("Skipping unreadable image %s: %s", f, exc)
                continue

    if not inputs:
        raise ValueError(f"No valid images found under {path}")

    return EvalDataset(inputs=inputs, labels=labels, metric=metric, task=task)


# ── Inference function loading ─────────────────────────────────────────────────

def load_infer_fn(spec: str) -> Callable[[list], Any]:
    """Load a callable from a 'path/to/module.py:function_name' spec.

    The function must accept a list of inputs (one item per sample, matching
    the format of EvalDataset.inputs) and return predictions as a numpy array,
    torch tensor, or list.

    Example::

        # inference.py
        def predict(inputs):
            tokens = tokenizer(inputs, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(**tokens).logits
            return logits.argmax(-1).numpy()
    """
    if ":" not in spec:
        raise ValueError(
            f"--infer-fn must be 'path/to/module.py:function_name', got: {spec!r}"
        )

    module_path, fn_name = spec.rsplit(":", 1)
    import importlib.util
    import sys

    path = Path(module_path)
    if not path.exists():
        raise FileNotFoundError(f"Inference module not found: {module_path}")

    spec_obj = importlib.util.spec_from_file_location("_aphex_infer_module", path)
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

    # Add the module's directory to sys.path so relative imports inside it work
    module_dir = str(path.parent.resolve())
    inserted = module_dir not in sys.path
    if inserted:
        sys.path.insert(0, module_dir)

    try:
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        if inserted:
            sys.path.remove(module_dir)

    fn = getattr(module, fn_name, None)
    if fn is None:
        available = [k for k in dir(module) if not k.startswith("_") and callable(getattr(module, k))]
        raise AttributeError(
            f"Function {fn_name!r} not found in {module_path}. "
            f"Available callables: {available}"
        )
    if not callable(fn):
        raise TypeError(f"{fn_name!r} in {module_path} is not callable")
    return fn


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_baseline(infer_fn: Callable, eval_dataset: EvalDataset) -> float:
    """Call the user's inference function on eval inputs and return the metric score.

    The function receives eval_dataset.inputs (a list, one item per sample) and
    must return predictions as a numpy array, torch tensor, or list.
    """
    raw = infer_fn(eval_dataset.inputs)
    outputs = _normalise_outputs(raw)
    return compute_metric(outputs, eval_dataset.labels, eval_dataset.metric)


# ── Metrics ───────────────────────────────────────────────────────────────────

def auto_metric(info: ModelInfo) -> tuple[str, str]:
    """Return (task, metric) based on model family."""
    family = (info.family or "").lower()
    framework = (info.framework or "").lower()

    if family == "llm":
        return "generation", "perplexity"
    if framework in ("sklearn", "xgboost", "lightgbm", "catboost"):
        return "classification", "accuracy"
    if family in ("cnn", "vit", "resnet", "image_classifier"):
        return "classification", "accuracy"
    return "classification", "accuracy"


def compute_metric(outputs: list[Any], labels: list[Any], metric: str) -> float:
    """Compute a scalar metric from model outputs and ground-truth labels."""
    preds = _stack_outputs(outputs)
    gts = np.array(labels)

    if metric == "accuracy":
        if preds.ndim > 1:
            preds = preds.argmax(axis=-1)
        return float((preds == gts).mean())

    if metric in ("f1_macro", "f1_weighted"):
        from sklearn.metrics import f1_score  # type: ignore[import]
        avg = "macro" if metric == "f1_macro" else "weighted"
        if preds.ndim > 1:
            preds = preds.argmax(axis=-1)
        return float(f1_score(gts, preds, average=avg, zero_division=0))

    if metric == "rmse":
        return float(np.sqrt(np.mean((preds.flatten() - gts.flatten()) ** 2)))

    if metric == "mae":
        return float(np.mean(np.abs(preds.flatten() - gts.flatten())))

    raise ValueError(
        f"Unknown metric {metric!r}. "
        f"Supported: accuracy, f1_macro, f1_weighted, rmse, mae"
    )


def compute_drop(baseline: float, candidate: float, metric: str) -> float:
    """Relative degradation in [0, ∞). 0 = no degradation."""
    if baseline == 0.0:
        return 0.0
    if metric in _HIGHER_IS_BETTER:
        return max(0.0, (baseline - candidate) / abs(baseline))
    if metric in _LOWER_IS_BETTER:
        return max(0.0, (candidate - baseline) / abs(baseline))
    return max(0.0, (baseline - candidate) / abs(baseline))


# ── Auto-optimization accuracy fill ───────────────────────────────────────────

_PYTORCH_EVAL_BACKENDS = frozenset({
    "pytorch_fp16", "pytorch_bf16", "pytorch_int8_dynamic", "onnx_int8_cpu",
    "tensorrt_fp16",   # proxy: evaluate as fp16
    "tensorrt_int8",   # proxy: evaluate as int8_dynamic
    "openvino_int8",   # proxy: evaluate as int8_dynamic
    "pytorch_prune_unstructured_30",
    "pytorch_prune_unstructured_50",
    "pytorch_prune_unstructured_70",
    "pytorch_prune_2_4",
})


_GENERATIVE_FAMILIES = frozenset({"llm", "transformer_decoder", "seq2seq"})


def fill_accuracy_drop(
    results: list,
    model: Any,
    info: ModelInfo,
    eval_dataset: EvalDataset,
) -> None:
    """Mutate BenchmarkResult.accuracy_drop in-place for accuracy-sensitive backends.

    Runs each optimized variant on labelled eval data, computes the task metric
    vs the original model baseline, and writes the relative drop back to the result.

    Skips models whose family is generative (LLM / decoder / seq2seq): a single-pass
    cosine similarity on logits doesn't approximate perplexity or generation quality,
    so reporting a number there would actively mislead. Use ``--infer-fn`` plus a
    user-defined metric in that case.
    """
    family = (info.family or "").lower()
    if family in _GENERATIVE_FAMILIES or eval_dataset.task == "generation":
        logger.warning(
            "accuracy fill: skipping family=%r because cosine-similarity on a "
            "single forward pass does not approximate generation quality. "
            "Pass --infer-fn pointing at a generation+scoring callable to score "
            "this model.", info.family,
        )
        return

    framework = (info.framework or "").lower()
    if framework == "pytorch":
        _fill_pytorch(results, model, eval_dataset)
    elif framework in ("sklearn", "xgboost", "lightgbm", "catboost"):
        _fill_sklearn(results, model, eval_dataset)


def _fill_pytorch(results: list, model: Any, eval_dataset: EvalDataset) -> None:
    """Measure accuracy drop per candidate, sharing a single FP32 deepcopy.

    The model is deep-copied exactly once. Dtype-only variants (fp16/bf16)
    reuse that copy and restore its FP32 state via the snapshotted state_dict
    after each measurement; backends that require structural rewrites (int8
    dynamic, ONNX export+quantize) get a fresh copy because they mutate
    module classes irreversibly.
    """
    import torch
    import torch.nn as nn

    if not isinstance(model, nn.Module):
        return

    inputs = eval_dataset.inputs

    try:
        base_model = _copy.deepcopy(model).cpu().eval()
    except Exception as exc:
        logger.warning("accuracy fill: deepcopy(model) failed: %s", exc)
        return

    fp32_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

    baseline_outputs = _run_in_dtype(base_model, inputs, torch.float32)
    if baseline_outputs is None:
        return
    base_model.load_state_dict(fp32_state)  # restore in case the run mutated buffers

    baseline_score = compute_metric(baseline_outputs, eval_dataset.labels, eval_dataset.metric)

    for r in results:
        if not r.ok or r.candidate.backend not in _PYTORCH_EVAL_BACKENDS:
            continue
        try:
            candidate_outputs = _run_pytorch_candidate(
                r.candidate, base_model, fp32_state, model, inputs
            )
        except Exception as exc:
            logger.warning(
                "accuracy fill: backend %s raised %s",
                r.candidate.backend, exc,
            )
            continue
        if candidate_outputs is None:
            continue
        candidate_score = compute_metric(candidate_outputs, eval_dataset.labels, eval_dataset.metric)
        r.accuracy_drop = compute_drop(baseline_score, candidate_score, eval_dataset.metric)


def _run_in_dtype(model: Any, inputs: list, dtype: Any) -> list | None:
    """Run model in *dtype* and return per-sample outputs as FP32 numpy arrays.

    Mutates the model in-place via .to(dtype); caller is responsible for
    restoring state_dict afterwards if FP32 weights are still needed.
    """
    import torch

    try:
        m = model.to(dtype).eval() if dtype != torch.float32 else model.eval()
        outputs = []
        with torch.no_grad():
            for inp in inputs:
                x = inp.float().cpu().to(dtype) if isinstance(inp, torch.Tensor) else inp
                out = m(x)
                outputs.append(
                    out.detach().float().cpu().numpy()
                    if isinstance(out, torch.Tensor) else out
                )
        return outputs
    except Exception as exc:
        logger.warning("accuracy run failed at dtype=%s: %s", dtype, exc)
        return None


def _run_pytorch_candidate(
    candidate: Any,
    base_model: Any,
    fp32_state: dict,
    original_model: Any,
    inputs: list,
) -> list | None:
    """Compute outputs for a candidate.

    For dtype-only backends we mutate base_model and restore fp32_state after.
    For structural backends (int8 dynamic, ONNX) we deepcopy `original_model`
    fresh, since those rewrites can't be reversed by load_state_dict.
    """
    import torch

    from infermap.benchmark import _ensure_quantization_engine, _export_to_onnx_bytes, _safe_quantize_dynamic

    backend = candidate.backend

    if backend in ("pytorch_fp16", "tensorrt_fp16"):
        try:
            return _run_in_dtype(base_model, inputs, torch.float16)
        finally:
            base_model.to(torch.float32)
            base_model.load_state_dict(fp32_state)

    if backend == "pytorch_bf16":
        try:
            return _run_in_dtype(base_model, inputs, torch.bfloat16)
        finally:
            base_model.to(torch.float32)
            base_model.load_state_dict(fp32_state)

    if backend in ("pytorch_int8_dynamic", "tensorrt_int8", "openvino_int8"):
        _ensure_quantization_engine()
        try:
            m = _safe_quantize_dynamic(_copy.deepcopy(original_model).cpu().eval())
        except Exception as exc:
            logger.warning("quantize_dynamic failed: %s", exc)
            return None
        outputs = []
        with torch.no_grad():
            for inp in inputs:
                x = inp.float().cpu() if isinstance(inp, torch.Tensor) else inp
                out = m(x)
                outputs.append(out.detach().cpu().numpy())
        return outputs

    if backend.startswith("pytorch_prune_"):
        from infermap.pruning import prune_model, spec_for_backend
        try:
            m = _copy.deepcopy(original_model).cpu().eval()
            m, _report = prune_model(m, spec_for_backend(backend))
        except Exception as exc:
            logger.warning("pruning (%s) failed: %s", backend, exc)
            return None
        outputs = []
        with torch.no_grad():
            for inp in inputs:
                x = inp.float().cpu() if isinstance(inp, torch.Tensor) else inp
                out = m(x)
                outputs.append(out.detach().float().cpu().numpy())
        return outputs

    if backend == "onnx_int8_cpu":
        import logging as _logging
        import tempfile as _tempfile
        import warnings
        from pathlib import Path as _Path

        import onnxruntime as ort
        from onnxruntime.quantization import QuantType
        from onnxruntime.quantization import quantize_dynamic as ort_quantize_dynamic

        dummy = inputs[0].float().cpu() if isinstance(inputs[0], torch.Tensor) else inputs[0]
        try:
            onnx_bytes = _export_to_onnx_bytes(_copy.deepcopy(original_model).cpu().eval(), dummy)
        except Exception as exc:
            logger.warning("ONNX export for accuracy eval failed: %s", exc)
            return None

        with _tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = _Path(tmpdir) / "model.onnx"
            quant_path = _Path(tmpdir) / "model_int8.onnx"
            onnx_path.write_bytes(onnx_bytes)

            root_logger = _logging.getLogger()
            ort_logger = _logging.getLogger("onnxruntime")
            prev_root, prev_ort = root_logger.level, ort_logger.level
            root_logger.setLevel(_logging.ERROR)
            ort_logger.setLevel(_logging.ERROR)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ort_quantize_dynamic(str(onnx_path), str(quant_path), weight_type=QuantType.QInt8)
            finally:
                root_logger.setLevel(prev_root)
                ort_logger.setLevel(prev_ort)

            sess = ort.InferenceSession(str(quant_path), providers=["CPUExecutionProvider"])

        outputs = []
        for inp in inputs:
            x = inp.float().cpu().numpy() if isinstance(inp, torch.Tensor) else inp
            out = sess.run(None, {"input": x})[0]
            outputs.append(out)
        return outputs

    return None


def _fill_sklearn(results: list, model: Any, eval_dataset: EvalDataset) -> None:
    predict_fn = _sklearn_predict(model)
    if predict_fn is None:
        return

    X = np.vstack([
        inp.numpy() if hasattr(inp, "numpy") else np.array(inp)
        for inp in eval_dataset.inputs
    ])

    try:
        baseline_raw = predict_fn(X)
    except Exception as exc:
        logger.warning("sklearn baseline predict failed: %s", exc)
        return

    baseline_outputs = _sklearn_raw_to_list(baseline_raw)
    baseline_score = compute_metric(baseline_outputs, eval_dataset.labels, eval_dataset.metric)

    for r in results:
        if not r.ok:
            continue
        candidate_raw = _run_sklearn_candidate(r.candidate, model, X)
        if candidate_raw is None:
            continue
        candidate_outputs = _sklearn_raw_to_list(candidate_raw)
        candidate_score = compute_metric(candidate_outputs, eval_dataset.labels, eval_dataset.metric)
        r.accuracy_drop = compute_drop(baseline_score, candidate_score, eval_dataset.metric)


def _sklearn_raw_to_list(raw: Any) -> list:
    arr = np.array(raw)
    if arr.ndim == 2:
        return [arr[i] for i in range(len(arr))]
    return [np.array([x]) for x in arr]


def _sklearn_predict(model: Any) -> Any:
    if hasattr(model, "predict_proba"):
        return model.predict_proba
    if hasattr(model, "predict"):
        return model.predict
    return None


def _run_sklearn_candidate(candidate: Any, model: Any, X: Any) -> Any:
    backend = candidate.backend
    try:
        if backend in ("sklearn_native", "xgboost_native", "lightgbm_native", "catboost_native"):
            predict_fn = _sklearn_predict(model)
            return predict_fn(X) if predict_fn else None
        if "onnx" in backend:
            return _sklearn_onnx_predict(model, X)
    except Exception as exc:
        logger.warning("sklearn candidate %s failed: %s", backend, exc)
        return None
    return None


def _sklearn_onnx_predict(model: Any, X: Any) -> Any:
    try:
        import onnxruntime as ort
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        n_features = X.shape[1] if X.ndim == 2 else 1
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: X.astype(np.float32)})[0]
        return out
    except Exception as exc:
        logger.warning("sklearn ONNX predict failed: %s", exc)
        return None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalise_outputs(raw: Any) -> list:
    """Convert model output (array/tensor/list) to a list of per-sample items."""
    import torch

    if isinstance(raw, torch.Tensor):
        arr = raw.detach().float().cpu().numpy()
        return [arr[i] for i in range(len(arr))]
    if isinstance(raw, np.ndarray):
        return [raw[i] for i in range(len(raw))]
    return list(raw)


def _stack_outputs(outputs: list[Any]) -> np.ndarray:
    import torch

    if not outputs:
        return np.array([])
    if isinstance(outputs[0], np.ndarray):
        return np.vstack(outputs) if outputs[0].ndim > 0 else np.array(outputs)
    if isinstance(outputs[0], torch.Tensor):
        return torch.stack([o.float() for o in outputs]).numpy()
    return np.array(outputs)
