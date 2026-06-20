"""Post-training pruning helpers — no retraining required.

Two strategies are supported:

- **Unstructured magnitude pruning** (``unstructured_magnitude``): zeros out the
  smallest-magnitude weights in every Linear / Conv2d / Conv1d layer. Storage
  benefit is real; latency on dense hardware usually isn't. We make the sparsity
  permanent (``prune.remove``) so saved models contain real zeros.

- **2:4 structured pruning** (``structured_2_4``): for every group of 4
  consecutive weights along the input dimension, keep the two largest by
  magnitude and zero the rest. Matches NVIDIA Ampere+ sparse Tensor Cores
  (`SM_80+`), where the matmul kernel can use this pattern for a real speedup.

Both functions are *destructive but deepcopy-safe*: pass a copy in, get a
modified module back. They report the achieved sparsity so callers can sanity-
check the result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("infermap.pruning")

_PRUNEABLE_TYPES_ATTR = ("Linear", "Conv1d", "Conv2d", "Conv3d")


@dataclass(frozen=True)
class PruneSpec:
    """Describes a pruning configuration without touching torch at import time."""

    method: str           # "unstructured_magnitude" | "structured_2_4"
    amount: float = 0.0   # for unstructured: fraction to zero; ignored for 2:4

    def __post_init__(self) -> None:
        if self.method == "unstructured_magnitude":
            if not 0.0 < self.amount < 1.0:
                raise ValueError(
                    f"unstructured_magnitude requires 0 < amount < 1, got {self.amount}"
                )
        elif self.method == "structured_2_4":
            pass
        else:
            raise ValueError(
                f"Unknown pruning method {self.method!r}. "
                "Supported: 'unstructured_magnitude', 'structured_2_4'."
            )

    @property
    def label(self) -> str:
        if self.method == "unstructured_magnitude":
            return f"prune_unstructured_{int(self.amount * 100):02d}"
        return "prune_2_4"


@dataclass
class PruneReport:
    """Result of a pruning pass — useful for logging and tests."""

    spec: PruneSpec
    weight_params_total: int
    weight_params_zeroed: int

    @property
    def sparsity(self) -> float:
        if self.weight_params_total == 0:
            return 0.0
        return self.weight_params_zeroed / self.weight_params_total


def prune_model(model: Any, spec: PruneSpec) -> tuple[Any, PruneReport]:
    """Apply *spec* to *model* in place; also return a PruneReport.

    Operates on every Linear / Conv* layer's ``weight`` parameter. Callers that
    don't want their original model mutated should ``copy.deepcopy`` first.
    """
    import torch.nn as nn
    import torch.nn.utils.prune as torch_prune

    targets = _pruneable_modules(model)
    if not targets:
        logger.warning(
            "prune_model: no Linear/Conv layers found in %s; returning unchanged",
            type(model).__name__,
        )
        return model, PruneReport(spec=spec, weight_params_total=0, weight_params_zeroed=0)

    if spec.method == "unstructured_magnitude":
        for mod in targets:
            torch_prune.l1_unstructured(mod, name="weight", amount=spec.amount)
            torch_prune.remove(mod, name="weight")
    elif spec.method == "structured_2_4":
        for mod in targets:
            _apply_2_4(mod)

    total, zeroed = _count_weight_sparsity(targets)
    report = PruneReport(spec=spec, weight_params_total=total, weight_params_zeroed=zeroed)
    logger.info(
        "prune_model: method=%s achieved sparsity=%.3f (%d/%d zero weights, %d layers)",
        spec.method, report.sparsity, zeroed, total, len(targets),
    )
    return model, report


def _pruneable_modules(model: Any) -> list[Any]:
    import torch.nn as nn

    out: list[Any] = []
    for mod in model.modules():
        cls_name = type(mod).__name__
        if cls_name in _PRUNEABLE_TYPES_ATTR and hasattr(mod, "weight"):
            # Skip layers whose weight isn't a leaf Parameter (e.g. already-quantized
            # DynamicQuantizedLinear); pruning APIs require a real Parameter.
            if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                out.append(mod)
    return out


def _apply_2_4(module: Any) -> None:
    """Zero the two smallest-magnitude entries in every contiguous group of 4
    along the input dimension (last dimension for Linear; in-channel dim for Conv).
    """
    import torch

    w = module.weight.data
    if w.ndim < 2:
        return
    flat = w.reshape(w.shape[0], -1)  # (out, in_total)
    in_total = flat.shape[1]
    if in_total < 4:
        return

    truncated = (in_total // 4) * 4
    blocks = flat[:, :truncated].reshape(flat.shape[0], -1, 4)
    abs_blocks = blocks.abs()
    # Indices of the two smallest entries in each group of 4.
    _, idx = abs_blocks.topk(k=2, dim=-1, largest=False)
    mask = torch.ones_like(blocks, dtype=torch.bool)
    mask.scatter_(-1, idx, False)
    blocks_masked = blocks * mask
    flat[:, :truncated] = blocks_masked.reshape(flat.shape[0], -1)
    module.weight.data = flat.reshape_as(w)


def _count_weight_sparsity(modules: list[Any]) -> tuple[int, int]:
    import torch

    total = 0
    zeroed = 0
    for mod in modules:
        w = mod.weight.data
        total += w.numel()
        zeroed += int((w == 0).sum().item())
    return total, zeroed


# ── Catalog used by the candidate generator and benchmark/eval pipeline ──────


PRUNING_BACKENDS: dict[str, PruneSpec] = {
    "pytorch_prune_unstructured_30": PruneSpec("unstructured_magnitude", 0.30),
    "pytorch_prune_unstructured_50": PruneSpec("unstructured_magnitude", 0.50),
    "pytorch_prune_unstructured_70": PruneSpec("unstructured_magnitude", 0.70),
    "pytorch_prune_2_4":            PruneSpec("structured_2_4"),
}


def is_pruning_backend(backend: str) -> bool:
    return backend in PRUNING_BACKENDS


def spec_for_backend(backend: str) -> PruneSpec:
    return PRUNING_BACKENDS[backend]
