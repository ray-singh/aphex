"""Soft-label knowledge distillation.

Trains a student model to match the teacher's soft outputs on a user-provided
dataset. This is the only piece of aphex that performs gradient updates — all
other functionality is post-training. Distillation is opt-in via the dedicated
``aphex distill`` command for that reason.

Loss formulation
----------------
For classification teachers, the loss is the standard Hinton-style mix:

    L = α · KL( softmax(student/T) || softmax(teacher/T) ) · T²
      + (1 - α) · CE(student, hard_label)

When the dataset has no usable hard labels (e.g. unlabeled distillation), set
``alpha=1.0`` and the CE term drops out.

For regression teachers, the loss is mean-squared error between student and
teacher outputs; the temperature and alpha mixing are ignored.

The trainer accepts plain Python callables for the student factory and the
dataset loader to avoid coupling to any particular framework outside torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

logger = logging.getLogger("infermap.distillation")


@dataclass
class DistillConfig:
    epochs: int = 3
    batch_size: int = 32
    lr: float = 1e-3
    temperature: float = 4.0
    alpha: float = 0.7              # weight on KD loss; (1 - alpha) on hard-label CE
    task: str = "classification"    # "classification" | "regression"
    device: str = "cpu"
    log_every: int = 25             # batches between progress logs

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.task not in ("classification", "regression"):
            raise ValueError(
                f"task must be 'classification' or 'regression', got {self.task!r}"
            )


@dataclass
class DistillReport:
    """Per-epoch training history + final teacher/student parameter counts."""

    config: DistillConfig
    teacher_params: int
    student_params: int
    epoch_losses: list[float] = field(default_factory=list)
    teacher_score: float | None = None   # optional baseline metric on eval set
    student_score: float | None = None   # student score on same set (same metric)
    metric: str | None = None

    @property
    def compression_ratio(self) -> float:
        if self.student_params == 0:
            return 0.0
        return self.teacher_params / self.student_params


def distill(
    teacher: Any,
    student: Any,
    inputs: list[Any],
    labels: list[Any] | None,
    config: DistillConfig | None = None,
) -> tuple[Any, DistillReport]:
    """Train *student* to mimic *teacher* on *(inputs, labels)*.

    Parameters
    ----------
    teacher: the trained nn.Module to imitate. Frozen during training.
    student: the smaller nn.Module to update. Returned in trained state.
    inputs: list of input tensors. Each item is one sample, batched on dim 0.
    labels: optional ground-truth labels (same length as ``inputs``). If None,
            the loss collapses to pure KD (``alpha`` is forced to 1.0).
    config: training hyperparameters. Defaults to a small classification run.
    """
    import torch

    config = config or DistillConfig()
    if labels is None:
        if config.alpha < 1.0:
            logger.info("no labels provided; forcing alpha=1.0 (pure KD)")
        config = DistillConfig(
            epochs=config.epochs, batch_size=config.batch_size, lr=config.lr,
            temperature=config.temperature, alpha=1.0, task=config.task,
            device=config.device, log_every=config.log_every,
        )

    device = torch.device(config.device)
    teacher = teacher.to(device).eval()
    student = student.to(device).train()
    for p in teacher.parameters():
        p.requires_grad_(False)

    n = len(inputs)
    if n == 0:
        raise ValueError("distill: inputs is empty")

    optim = torch.optim.Adam(student.parameters(), lr=config.lr)
    loss_history: list[float] = []

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_inputs, batch_labels in _batches(inputs, labels, config.batch_size):
            x = _stack(batch_inputs).to(device)
            with torch.no_grad():
                teacher_out = teacher(x)

            student_out = student(x)
            loss = _distill_loss(
                student_out, teacher_out, batch_labels, config, device,
            )

            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            n_batches += 1
            if n_batches % config.log_every == 0:
                logger.info(
                    "epoch %d batch %d loss=%.4f", epoch + 1, n_batches, loss.item(),
                )

        avg = epoch_loss / max(1, n_batches)
        loss_history.append(avg)
        logger.info("epoch %d/%d avg_loss=%.4f", epoch + 1, config.epochs, avg)

    student.eval()
    report = DistillReport(
        config=config,
        teacher_params=_param_count(teacher),
        student_params=_param_count(student),
        epoch_losses=loss_history,
    )
    return student, report


def score(
    model: Any,
    inputs: list[Any],
    labels: list[Any],
    task: str,
    device: str = "cpu",
) -> float:
    """Compute accuracy (classification) or RMSE (regression) on (inputs, labels)."""
    import numpy as np
    import torch

    dev = torch.device(device)
    model = model.to(dev).eval()
    preds: list[Any] = []
    with torch.no_grad():
        for batch_inputs, _ in _batches(inputs, labels, 32):
            x = _stack(batch_inputs).to(dev)
            out = model(x).detach().cpu()
            preds.append(out)

    stacked = torch.cat(preds, dim=0).numpy()
    gt = np.array(labels)
    if task == "classification":
        if stacked.ndim > 1:
            stacked = stacked.argmax(axis=-1)
        return float((stacked == gt).mean())
    return float(np.sqrt(np.mean((stacked.flatten() - gt.flatten()) ** 2)))


# ── Internals ────────────────────────────────────────────────────────────────


def _distill_loss(
    student_out: Any,
    teacher_out: Any,
    labels: list[Any] | None,
    config: DistillConfig,
    device: Any,
) -> Any:
    import torch
    import torch.nn.functional as F

    if config.task == "regression":
        return F.mse_loss(student_out, teacher_out)

    T = config.temperature
    kd = F.kl_div(
        F.log_softmax(student_out / T, dim=-1),
        F.softmax(teacher_out / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)

    if config.alpha >= 1.0 or labels is None:
        return kd

    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    ce = F.cross_entropy(student_out, y)
    return config.alpha * kd + (1.0 - config.alpha) * ce


def _batches(
    inputs: list[Any],
    labels: list[Any] | None,
    batch_size: int,
) -> Iterable[tuple[list[Any], list[Any] | None]]:
    n = len(inputs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bi = inputs[start:end]
        bl = labels[start:end] if labels is not None else None
        yield bi, bl


def _stack(items: list[Any]) -> Any:
    """Join a list of single-sample tensors into a batch tensor.

    Accepts tensors shaped (1, ...) (the EvalDataset convention) or (...) and
    produces a tensor shaped (batch, ...).
    """
    import torch

    if not items:
        raise ValueError("_stack: empty batch")
    first = items[0]
    if not isinstance(first, torch.Tensor):
        first = torch.as_tensor(first)
    if first.ndim >= 1 and first.shape[0] == 1:
        return torch.cat([
            i if isinstance(i, torch.Tensor) else torch.as_tensor(i)
            for i in items
        ], dim=0)
    return torch.stack([
        i if isinstance(i, torch.Tensor) else torch.as_tensor(i)
        for i in items
    ], dim=0)


def _param_count(model: Any) -> int:
    return sum(p.numel() for p in model.parameters())


def load_student_factory(spec: str) -> Callable[[], Any]:
    """Load a no-argument callable returning an untrained student nn.Module.

    Spec format mirrors --infer-fn: ``path/to/module.py:make_student``. The
    callable must take zero arguments and return a torch.nn.Module.
    """
    from pathlib import Path
    if ":" not in spec:
        raise ValueError(
            f"--student must be 'path/to/module.py:function', got: {spec!r}"
        )
    module_path, fn_name = spec.rsplit(":", 1)
    import importlib.util
    import sys

    path = Path(module_path)
    if not path.exists():
        raise FileNotFoundError(f"Student module not found: {module_path}")

    spec_obj = importlib.util.spec_from_file_location("_aphex_student_module", path)
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

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
    if fn is None or not callable(fn):
        raise AttributeError(
            f"Function {fn_name!r} not found or not callable in {module_path}"
        )
    return fn
