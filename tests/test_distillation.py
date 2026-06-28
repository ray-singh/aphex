"""Tests for the distillation module and the `aphex distill` CLI command."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from aphex.distillation import (
    DistillConfig,
    distill,
    load_student_factory,
    score,
)

# ── DistillConfig validation ─────────────────────────────────────────────────


def test_distill_config_rejects_invalid_epochs() -> None:
    with pytest.raises(ValueError, match="epochs"):
        DistillConfig(epochs=0)


def test_distill_config_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        DistillConfig(alpha=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        DistillConfig(alpha=1.1)


def test_distill_config_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        DistillConfig(temperature=0)


def test_distill_config_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="task must be"):
        DistillConfig(task="generation")


# ── Core training loop ──────────────────────────────────────────────────────


def _trained_teacher_classification() -> tuple[nn.Module, list[torch.Tensor], list[int]]:
    """Returns a teacher trained on a tiny synthetic 3-class problem + eval data."""
    torch.manual_seed(0)
    teacher = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 3))
    X = torch.randn(200, 8)
    # Deterministic 3-class label derived from input
    y = ((X[:, 0] > 0).long() + (X[:, 1] > 0).long()).clamp(max=2)
    optim = torch.optim.Adam(teacher.parameters(), lr=1e-2)
    for _ in range(120):
        loss = nn.functional.cross_entropy(teacher(X), y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    teacher.eval()
    inputs = [X[i:i+1] for i in range(200)]
    return teacher, inputs, y.tolist()


def test_distill_loss_decreases_on_classification() -> None:
    teacher, inputs, labels = _trained_teacher_classification()
    student = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 3))
    cfg = DistillConfig(
        epochs=8, batch_size=16, lr=1e-2,
        temperature=2.0, alpha=0.7, task="classification",
    )
    _, report = distill(teacher, student, inputs, labels, cfg)
    assert len(report.epoch_losses) == 8
    # First-epoch loss strictly greater than last-epoch loss.
    assert report.epoch_losses[0] > report.epoch_losses[-1]


def test_distill_classification_improves_random_student_accuracy() -> None:
    """A distilled student should score above random on the teacher's task."""
    teacher, inputs, labels = _trained_teacher_classification()

    # Fresh random student
    torch.manual_seed(123)
    student = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 3))
    random_acc = score(student, inputs, labels, task="classification")

    cfg = DistillConfig(epochs=15, batch_size=16, lr=1e-2,
                        temperature=2.0, alpha=0.8, task="classification")
    distilled, _ = distill(teacher, student, inputs, labels, cfg)
    distilled_acc = score(distilled, inputs, labels, task="classification")

    # Must exceed both random (~0.33) and the untrained student.
    assert distilled_acc > 0.40
    assert distilled_acc >= random_acc


def test_distill_pure_kd_when_labels_none() -> None:
    """labels=None forces alpha=1.0 (pure KD) and the loop still trains."""
    teacher, inputs, _ = _trained_teacher_classification()
    student = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 3))
    cfg = DistillConfig(epochs=5, batch_size=16, lr=1e-2,
                        temperature=2.0, alpha=0.5, task="classification")
    _, report = distill(teacher, student, inputs, labels=None, config=cfg)
    # Trainer should have flipped alpha to 1.0 internally.
    assert report.config.alpha == 1.0
    assert report.epoch_losses[0] > report.epoch_losses[-1]


def test_distill_regression_loss_decreases() -> None:
    torch.manual_seed(0)
    teacher = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))
    X = torch.randn(100, 4)
    target = X.sum(-1, keepdim=True) * 2 - 1
    optim = torch.optim.Adam(teacher.parameters(), lr=1e-2)
    for _ in range(60):
        loss = nn.functional.mse_loss(teacher(X), target)
        optim.zero_grad()
        loss.backward()
        optim.step()
    teacher.eval()

    student = nn.Linear(4, 1)
    inputs = [X[i:i+1] for i in range(100)]
    cfg = DistillConfig(epochs=10, batch_size=16, lr=1e-2, task="regression")
    _, report = distill(teacher, student, inputs, labels=None, config=cfg)
    assert report.epoch_losses[0] > report.epoch_losses[-1]


def test_distill_report_compression_ratio() -> None:
    teacher, inputs, labels = _trained_teacher_classification()
    student = nn.Linear(8, 3)
    cfg = DistillConfig(epochs=1, batch_size=16, lr=1e-2)
    _, report = distill(teacher, student, inputs, labels, cfg)
    # Teacher 387 params (8*32+32 + 32*3+3), student 27 (8*3+3). Compression > 10×.
    assert report.compression_ratio > 5.0
    assert report.teacher_params > report.student_params


def test_distill_rejects_empty_inputs() -> None:
    teacher = nn.Linear(4, 2)
    student = nn.Linear(4, 2)
    with pytest.raises(ValueError, match="empty"):
        distill(teacher, student, inputs=[], labels=None)


def test_distill_freezes_teacher_parameters() -> None:
    teacher, inputs, labels = _trained_teacher_classification()
    teacher_params_before = [p.detach().clone() for p in teacher.parameters()]
    student = nn.Linear(8, 3)
    cfg = DistillConfig(epochs=2, batch_size=16, lr=1e-1)  # large LR
    distill(teacher, student, inputs, labels, cfg)
    for before, after in zip(teacher_params_before, teacher.parameters(), strict=False):
        assert torch.equal(before, after), "teacher parameters changed during distill"


# ── load_student_factory ─────────────────────────────────────────────────────


def test_load_student_factory_loads_callable(tmp_path: Path) -> None:
    f = tmp_path / "stu.py"
    f.write_text(
        "import torch.nn as nn\n"
        "def make(): return nn.Linear(4, 2)\n"
    )
    factory = load_student_factory(f"{f}:make")
    m = factory()
    assert isinstance(m, nn.Linear)


def test_load_student_factory_rejects_bad_spec() -> None:
    with pytest.raises(ValueError, match="path/to/module.py"):
        load_student_factory("no_colon_here")


def test_load_student_factory_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_student_factory("/nonexistent_path_aphex_test.py:make")


def test_load_student_factory_missing_function(tmp_path: Path) -> None:
    f = tmp_path / "stu.py"
    f.write_text("def other(): pass\n")
    with pytest.raises(AttributeError):
        load_student_factory(f"{f}:does_not_exist")


# ── End-to-end CLI test (subprocess) ─────────────────────────────────────────


def test_cli_distill_end_to_end(tmp_path: Path) -> None:
    """Run `aphex distill` as a subprocess and verify it produces artifacts."""
    teacher, _inputs, _labels = _trained_teacher_classification()

    teacher_path = tmp_path / "teacher.pt"
    eval_path = tmp_path / "eval.pt"
    student_path = tmp_path / "student.pt"
    report_path = tmp_path / "report.json"
    student_factory = tmp_path / "stu.py"

    torch.save(teacher, teacher_path)
    X = torch.randn(60, 8)
    y = ((X[:, 0] > 0).long() + (X[:, 1] > 0).long()).clamp(max=2)
    torch.save({"inputs": X, "labels": y}, eval_path)

    student_factory.write_text(
        "import torch.nn as nn\n"
        "def make(): return nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 3))\n"
    )

    env = {
        **__import__("os").environ,
        "APHEX_TRUST_PICKLE": "1",  # teacher saved as full nn.Module
    }
    result = subprocess.run(
        [
            sys.executable, "-m", "aphex.cli", "distill", str(teacher_path),
            "--student", f"{student_factory}:make",
            "--eval", str(eval_path),
            "--epochs", "4",
            "--batch-size", "16",
            "--lr", "0.01",
            "--output", str(student_path),
            "--report", str(report_path),
        ],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert result.returncode == 0, (
        f"distill CLI failed: stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert student_path.exists()
    assert report_path.exists()

    payload = json.loads(report_path.read_text())
    assert payload["config"]["epochs"] == 4
    assert len(payload["epoch_losses"]) == 4
    assert payload["epoch_losses"][0] > payload["epoch_losses"][-1]
    assert payload["compression_ratio"] > 1.0
    assert payload["teacher_score"] is not None
    assert payload["student_score"] is not None

    # student.pt should be a state_dict, not a full module — verify by loading.
    sd = torch.load(student_path, weights_only=True)
    assert isinstance(sd, dict)
    assert all(isinstance(v, torch.Tensor) for v in sd.values())
