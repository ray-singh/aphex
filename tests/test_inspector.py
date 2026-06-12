"""Tests for model inspector."""

import torch
import torch.nn as nn
import pytest

from infermap.inspector import inspect_model, ModelInfo


class TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(16, 32)
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.out = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = x.unsqueeze(1)
        x, _ = self.attn(x, x, x)
        return self.out(x.squeeze(1))


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_inspect_transformer() -> None:
    model = TinyTransformer()
    info = inspect_model(model)

    assert info.framework == "pytorch"
    assert info.family == "transformer"
    assert info.parameters > 0
    assert info.estimated_memory_fp32_gb > 0
    assert info.estimated_memory_fp16_gb < info.estimated_memory_fp32_gb


def test_inspect_cnn() -> None:
    model = TinyCNN()
    info = inspect_model(model)

    assert info.framework == "pytorch"
    assert info.family == "cnn"
    assert info.parameters > 0


def test_memory_estimation_by_dtype() -> None:
    model = TinyTransformer()
    info = inspect_model(model)

    fp32 = info.estimated_memory_gb("fp32")
    fp16 = info.estimated_memory_gb("fp16")
    int8 = info.estimated_memory_gb("int8")
    int4 = info.estimated_memory_gb("int4")

    assert fp32 > fp16 > int8 > int4


def test_trainable_parameters() -> None:
    model = TinyCNN()
    # Freeze all params
    for p in model.parameters():
        p.requires_grad = False

    info = inspect_model(model)
    assert info.trainable_parameters == 0
    assert info.parameters > 0
