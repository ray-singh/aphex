"""Shared pytest fixtures and hardware-skip markers."""

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_cuda: skip if CUDA unavailable")
    config.addinivalue_line("markers", "requires_mps: skip if MPS unavailable")


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
requires_mps = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
