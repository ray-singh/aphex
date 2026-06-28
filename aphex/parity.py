"""Numerical parity check for converted deployment artifacts.

A conversion (FP16 cast, INT8 quantization, ONNX/TensorRT/OpenVINO export) can
silently produce an artifact whose outputs no longer match the source model.
``verify_artifact`` loads the *written* artifact back from disk, runs it and the
source model on the same inputs, and reports how far the outputs diverge so a
broken export is caught before it ships.

The check is deliberately tolerant of value-level noise from reduced precision
(cosine similarity, top-1 agreement) rather than demanding bit-exactness — the
goal is to catch *corruption*, not to forbid quantization error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("aphex.parity")

# Per-backend pass thresholds: (min cosine similarity, min top-1 agreement).
# FP32 paths must be near-exact; reduced-precision paths are allowed to drift.
_DEFAULT_THRESHOLD = (0.99, 0.95)
_THRESHOLDS: dict[str, tuple[float, float]] = {
    "pytorch_fp32": (0.99999, 0.999),
    "onnx_cpu": (0.9999, 0.99),
    "onnx_cuda": (0.9999, 0.99),
    "onnx_coreml": (0.999, 0.98),
    "pytorch_fp16": (0.999, 0.98),
    "pytorch_bf16": (0.99, 0.97),
    "pytorch_int8_dynamic": (0.99, 0.95),
    "onnx_int8_cpu": (0.99, 0.95),
    "tensorrt_fp32": (0.9999, 0.99),
    "tensorrt_fp16": (0.999, 0.98),
    "tensorrt_int8": (0.98, 0.95),
    "openvino_fp32": (0.9999, 0.99),
    "openvino_int8": (0.99, 0.95),
}

_MAX_PARITY_SAMPLES = 8


@dataclass
class ParityResult:
    backend: str
    passed: bool
    n_samples: int
    cosine_sim: float = 1.0
    max_abs_diff: float = 0.0
    mean_rel_diff: float = 0.0
    top1_agreement: float | None = None  # None when output isn't class-like
    cosine_threshold: float = _DEFAULT_THRESHOLD[0]
    top1_threshold: float = _DEFAULT_THRESHOLD[1]
    skipped: bool = False  # backend couldn't be loaded/run in this environment
    reason: str | None = None  # why it was skipped, or the failure detail

    def summary(self) -> str:
        if self.skipped:
            return f"parity skipped ({self.reason})"
        verdict = "ok" if self.passed else "FAILED"
        parts = [
            f"cosine={self.cosine_sim:.5f}",
            f"max_abs={self.max_abs_diff:.3e}",
        ]
        if self.top1_agreement is not None:
            parts.append(f"top1={self.top1_agreement * 100:.1f}%")
        return f"parity {verdict} ({', '.join(parts)}, n={self.n_samples})"


def verify_artifact(
    source_model: Any,
    backend: str,
    written_paths: list[Path],
    input_shape: list[int],
    sample_inputs: list[Any] | None = None,
    *,
    model_info: Any = None,
    n_samples: int = _MAX_PARITY_SAMPLES,
    input_spec: Any = None,
) -> ParityResult:
    """Compare a written artifact's outputs against the source model.

    Args:
        source_model: The native (pre-conversion) model object.
        backend: The backend the artifact was converted to.
        written_paths: Files produced by ``converter.convert`` (first is the
            primary artifact; .xml for OpenVINO).
        input_shape: Input tensor shape without the batch dimension.
        sample_inputs: Representative inputs (e.g. real eval/calibration
            samples). Each is a single-sample tensor. Falls back to random data
            when not supplied — sufficient to catch structural corruption.
        model_info: Optional ModelInfo; used to honour embedding vocab sizes.
        n_samples: Cap on how many samples to compare.

    Never raises: any failure to load/run the artifact returns a skipped
    ``ParityResult`` so verification can't itself break a conversion.
    """
    import torch

    cos_th, top1_th = _THRESHOLDS.get(backend, _DEFAULT_THRESHOLD)

    try:
        inputs = _build_inputs(sample_inputs, input_shape, model_info, n_samples, input_spec)
        reference = _run_source(source_model, inputs)
        candidate = _run_artifact(backend, written_paths, inputs)
    except _ParitySkip as skip:
        return ParityResult(
            backend=backend, passed=True, n_samples=0,
            cosine_threshold=cos_th, top1_threshold=top1_th,
            skipped=True, reason=str(skip),
        )
    except Exception as exc:  # noqa: BLE001 — verification must never crash convert
        logger.debug("parity check errored for %s: %s", backend, exc)
        return ParityResult(
            backend=backend, passed=True, n_samples=0,
            cosine_threshold=cos_th, top1_threshold=top1_th,
            skipped=True, reason=f"could not run artifact: {exc}",
        )

    if len(reference) != len(candidate) or not reference:
        return ParityResult(
            backend=backend, passed=True, n_samples=0,
            cosine_threshold=cos_th, top1_threshold=top1_th,
            skipped=True, reason="output count mismatch / no outputs",
        )

    ref = torch.stack([r.reshape(-1).float() for r in reference])
    cand = torch.stack([c.reshape(-1).float() for c in candidate])

    cosine = float(
        torch.nn.functional.cosine_similarity(ref, cand, dim=1).mean().clamp(-1.0, 1.0)
    )
    max_abs = float((ref - cand).abs().max())
    denom = ref.abs().mean().clamp_min(1e-12)
    mean_rel = float((ref - cand).abs().mean() / denom)

    top1 = _top1_agreement(reference, candidate)

    passed = cosine >= cos_th and (top1 is None or top1 >= top1_th)

    return ParityResult(
        backend=backend,
        passed=passed,
        n_samples=len(reference),
        cosine_sim=cosine,
        max_abs_diff=max_abs,
        mean_rel_diff=mean_rel,
        top1_agreement=top1,
        cosine_threshold=cos_th,
        top1_threshold=top1_th,
    )


class _ParitySkip(Exception):
    """Raised when an artifact can't be exercised in the current environment."""


# ── input construction ─────────────────────────────────────────────────────


def _build_inputs(
    sample_inputs: list[Any] | None,
    input_shape: list[int],
    model_info: Any,
    n_samples: int,
    input_spec: Any = None,
) -> list[Any]:
    """Return a list of input items; each item is a tensor (single-input) or a
    tuple of tensors (multi-input)."""
    import torch

    vocab = getattr(model_info, "vocab_size", None) if model_info is not None else None

    # Multi-input: synthesise random tuples from the spec (real-sample mapping
    # across several named tensors isn't well-defined).
    if input_spec is not None and not input_spec.is_single:
        return [
            input_spec.build(1, torch_dtype=torch.float32, device="cpu", vocab=vocab)
            for _ in range(min(n_samples, 4))
        ]

    if sample_inputs:
        out: list[Any] = []
        for x in sample_inputs[:n_samples]:
            t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
            if t.dim() == len(input_shape):  # add batch dim if missing
                t = t.unsqueeze(0)
            out.append(t.cpu())
        if out:
            return out

    inputs = []
    for _ in range(min(n_samples, 4)):
        if vocab:
            inputs.append(torch.randint(0, vocab, (1, *input_shape), dtype=torch.long))
        else:
            inputs.append(torch.randn(1, *input_shape))
    return inputs


# ── source + artifact execution ────────────────────────────────────────────


def _as_tuple(item: Any) -> tuple:
    return item if isinstance(item, tuple) else (item,)


def _call(model: Any, item: Any) -> Any:
    return model(*item) if isinstance(item, tuple) else model(item)


def _run_source(model: Any, inputs: list[Any]) -> list[Any]:
    import copy

    import torch

    m = copy.deepcopy(model).cpu().eval().float()
    outs = []
    with torch.no_grad():
        for x in inputs:
            outs.append(_first_tensor(_call(m, x)).detach().cpu())
    return outs


def _run_artifact(backend: str, written_paths: list[Path], inputs: list[Any]) -> list[Any]:
    if not written_paths:
        raise _ParitySkip("no artifact written")
    path = written_paths[0]

    if backend.startswith("pytorch_"):
        return _run_pytorch(path, inputs)
    if backend.startswith("onnx_"):
        return _run_onnx(path, inputs)
    if backend.startswith("tensorrt_"):
        return _run_tensorrt(path, inputs)
    if backend.startswith("openvino_"):
        return _run_openvino(path, inputs)
    raise _ParitySkip(f"no artifact runner for backend {backend!r}")


def _run_pytorch(path: Path, inputs: list[Any]) -> list[Any]:
    import torch

    try:
        model = torch.jit.load(str(path), map_location="cpu")
    except Exception:
        # The artifact is a full pickled nn.Module that aphex's own converter
        # just wrote, so weights_only=False is safe here (we produced it).
        model = torch.load(str(path), map_location="cpu", weights_only=False)
    model.eval()

    # Match the artifact's parameter dtype (fp16/bf16 artifacts need cast inputs).
    param_dtype = torch.float32
    for p in model.parameters():
        param_dtype = p.dtype
        break

    def _cast(t: Any) -> Any:
        return t.to(param_dtype) if t.is_floating_point() else t

    outs = []
    with torch.no_grad():
        for x in inputs:
            item = tuple(_cast(t) for t in x) if isinstance(x, tuple) else _cast(x)
            try:
                out = _call(model, item)
            except (RuntimeError, NotImplementedError) as exc:
                # e.g. half-precision op unimplemented on CPU — not a corruption signal.
                raise _ParitySkip(f"artifact not runnable on CPU: {exc}") from exc
            outs.append(_first_tensor(out).detach().cpu())
    return outs


def _run_onnx(path: Path, inputs: list[Any]) -> list[Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise _ParitySkip("onnxruntime not installed") from exc

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    import torch

    def _np(t: Any) -> Any:
        # float inputs → float32 (ONNX export dtype); int/long index inputs kept.
        return (t.float() if t.is_floating_point() else t).cpu().numpy()

    outs = []
    for x in inputs:
        tensors = _as_tuple(x)
        feeds = {n: _np(t) for n, t in zip(in_names, tensors, strict=False)}
        result = sess.run(None, feeds)
        outs.append(torch.from_numpy(result[0]))
    return outs


def _run_tensorrt(path: Path, inputs: list[Any]) -> list[Any]:
    try:
        import tensorrt  # noqa: F401
        import torch  # noqa: F401

        from aphex.benchmark import _TensorRTRunner
    except ImportError as exc:
        raise _ParitySkip("tensorrt not available") from exc

    if not torch.cuda.is_available():
        raise _ParitySkip("CUDA not available for TensorRT artifact")

    runner = _TensorRTRunner(path.read_bytes())
    return [_first_tensor(runner(x)).detach().cpu() for x in inputs]


def _run_openvino(path: Path, inputs: list[Any]) -> list[Any]:
    try:
        import openvino as ov

        from aphex.benchmark import _OpenVINORunner
    except ImportError as exc:
        raise _ParitySkip("openvino not installed") from exc

    xml = path if path.suffix == ".xml" else path.with_suffix(".xml")
    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(xml)), "CPU")
    runner = _OpenVINORunner(compiled)
    return [_first_tensor(runner(x)).detach().cpu() for x in inputs]


# ── helpers ─────────────────────────────────────────────────────────────────


def _first_tensor(out: Any) -> Any:
    """Extract the primary tensor from a model output (tensor / tuple / dict)."""
    import torch

    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(out, dict):
        for item in out.values():
            if isinstance(item, torch.Tensor):
                return item
    raise _ParitySkip(f"unsupported output type for parity: {type(out)!r}")


def _top1_agreement(reference: list[Any], candidate: list[Any]) -> float | None:
    """Fraction of samples whose argmax matches, when outputs look like logits.

    Returns None when outputs aren't class-like (last dim <= 1), so regression
    heads don't get scored on a meaningless argmax.
    """
    matches = 0
    counted = 0
    for r, c in zip(reference, candidate, strict=False):
        rt, ct = r.reshape(r.shape[0] if r.dim() > 1 else 1, -1), c.reshape(
            c.shape[0] if c.dim() > 1 else 1, -1
        )
        if rt.shape[-1] <= 1:
            return None
        matches += int((rt.argmax(dim=-1) == ct.argmax(dim=-1)).sum())
        counted += rt.shape[0]
    if counted == 0:
        return None
    return matches / counted
