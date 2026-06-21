"""Benchmark case definitions for the APHEX Evaluation Suite.

Each BenchmarkCase describes one study point: a model, its input shape, the
batch sizes to sweep, and optionally a labelled eval dataset loader for
accuracy measurement.  Cases are grouped into SUITE (real models, needs
optional deps + weights) and SMOKE_SUITE (tiny synthetic models, always runs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BenchmarkCase:
    name: str
    # Zero-arg callable returning an nn.Module (or sklearn/etc. model)
    load_model: Callable[[], Any]
    # Input tensor shape without batch dim (for PyTorch models)
    input_shape: list[int]
    batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 8])
    # Optional: zero-arg callable returning (inputs: list, labels: list)
    load_eval: Callable[[], tuple[list[Any], list[Any]]] | None = None
    # Framework hint for get_plugin; None = auto-detect
    framework: str | None = None
    # Human-readable description of the target hardware (informational only)
    hardware_note: str = ""


# ── real model loaders ────────────────────────────────────────────────────────


def _load_resnet50() -> Any:
    from torchvision.models import ResNet50_Weights, resnet50
    m = resnet50(weights=ResNet50_Weights.DEFAULT)
    m.eval()
    return m


def _load_bert_base() -> Any:
    """Returns a BERT-base wrapped to accept a single integer tensor (input_ids)."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class _BertWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bert = AutoModel.from_pretrained("bert-base-uncased")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.bert(input_ids=x.long(), attention_mask=torch.ones_like(x))
            return out.last_hidden_state

    m = _BertWrapper()
    m.eval()
    return m


def _load_lgbm() -> Any:
    """Train a LightGBM classifier on sklearn's breast-cancer dataset."""
    from lightgbm import LGBMClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)
    model = LGBMClassifier(n_estimators=200, max_depth=6, random_state=0, verbose=-1)
    model.fit(X, y)
    return model


def _load_lgbm_eval() -> tuple[list[Any], list[Any]]:
    import numpy as np
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)
    # last 100 samples as eval
    return list(X[-100:]), list(y[-100:].astype(int))


# ── smoke model loaders (tiny synthetic, no optional deps) ────────────────────


def _smoke_cnn() -> Any:
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(16, 10),
    ).eval()


def _smoke_mlp() -> Any:
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    ).eval()


def _smoke_mlp_eval() -> tuple[list[Any], list[Any]]:
    import torch
    inputs = [torch.randn(1, 128) for _ in range(100)]
    # deterministic pseudo-labels
    labels = [int(inp[0, 0].item() > 0) for inp in inputs]
    return inputs, labels


# ── suites ────────────────────────────────────────────────────────────────────


SUITE: list[BenchmarkCase] = [
    BenchmarkCase(
        name="ResNet-50",
        load_model=_load_resnet50,
        input_shape=[3, 224, 224],
        batch_sizes=[1, 4, 8, 16],
        hardware_note="T4 / A10G / CPU",
    ),
    BenchmarkCase(
        name="BERT-base",
        load_model=_load_bert_base,
        input_shape=[128],       # seq_len; wrapper accepts integer tensor
        batch_sizes=[1, 4, 8],
        hardware_note="T4 / A10G",
    ),
    BenchmarkCase(
        name="LightGBM-200",
        load_model=_load_lgbm,
        input_shape=[30],        # breast-cancer feature count
        batch_sizes=[1, 32, 128],
        load_eval=_load_lgbm_eval,
        framework="sklearn",
        hardware_note="CPU",
    ),
]

# Always runnable — used for CI and quick local validation.
SMOKE_SUITE: list[BenchmarkCase] = [
    BenchmarkCase(
        name="smoke-cnn",
        load_model=_smoke_cnn,
        input_shape=[3, 32, 32],
        batch_sizes=[1, 4],
        hardware_note="CPU (synthetic)",
    ),
    BenchmarkCase(
        name="smoke-mlp",
        load_model=_smoke_mlp,
        input_shape=[128],
        batch_sizes=[1, 4],
        load_eval=_smoke_mlp_eval,
        hardware_note="CPU (synthetic)",
    ),
]
