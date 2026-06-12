"""Model inspector — identifies framework, family, parameter count, memory footprint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ModelFamily = Literal["transformer", "cnn", "unknown"]
Framework = Literal["pytorch", "unknown"]

BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


@dataclass
class ModelInfo:
    framework: Framework
    family: ModelFamily
    parameters: int
    trainable_parameters: int
    estimated_memory_fp32_gb: float
    estimated_memory_fp16_gb: float
    input_shape: list[int] | None = None
    model_path: str | None = None

    def estimated_memory_gb(self, dtype: str = "fp32") -> float:
        bpp = BYTES_PER_PARAM.get(dtype, 4.0)
        return (self.parameters * bpp * 1.2) / 1e9  # 1.2x overhead for activations


def inspect_model(model_or_path: Any, input_shape: list[int] | None = None) -> ModelInfo:
    """
    Inspect a model and return its profile.

    Args:
        model_or_path: A torch.nn.Module instance or path to a saved model file.
        input_shape: Optional input shape for torchinfo summary (without batch dim).
    """
    import torch
    import torch.nn as nn

    if isinstance(model_or_path, (str, Path)):
        model = _load_model(Path(model_or_path))
    elif isinstance(model_or_path, nn.Module):
        model = model_or_path
    else:
        raise TypeError(f"Expected nn.Module or path, got {type(model_or_path)}")

    framework: Framework = "pytorch"
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    family = _detect_family(model)

    mem_fp32 = (params * 4.0 * 1.2) / 1e9
    mem_fp16 = (params * 2.0 * 1.2) / 1e9

    path_str: str | None = None
    if isinstance(model_or_path, (str, Path)):
        path_str = str(model_or_path)

    return ModelInfo(
        framework=framework,
        family=family,
        parameters=params,
        trainable_parameters=trainable,
        estimated_memory_fp32_gb=mem_fp32,
        estimated_memory_fp16_gb=mem_fp16,
        input_shape=input_shape,
        model_path=path_str,
    )


def _load_model(path: Path) -> Any:
    import zipfile

    import torch
    import torch.nn as nn

    # TorchScript archives are ZIP files with a specific marker; load them directly
    # to avoid a UserWarning from torch.load's auto-dispatch.
    if zipfile.is_zipfile(path):
        try:
            return torch.jit.load(str(path), map_location="cpu")
        except Exception:
            pass  # fall through to torch.load for non-script zip saves

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj
    raise ValueError(
        f"Loaded object from {path} is not an nn.Module. "
        "Pass a full model (not just a state dict) or load the architecture first."
    )


# Layer-name heuristics for family detection
_TRANSFORMER_HINTS = {
    "attention",
    "transformer",
    "bert",
    "gpt",
    "t5",
    "vit",
    "multiheadattn",
    "self_attn",
    "cross_attn",
    "encoder",
    "decoder",
}

_CNN_HINTS = {
    "conv",
    "resnet",
    "efficientnet",
    "convnext",
    "mobilenet",
    "densenet",
    "vgg",
}


def _detect_family(model: Any) -> ModelFamily:
    class_name = type(model).__name__.lower()
    module_names = {type(m).__name__.lower() for m in model.modules()}
    all_names = module_names | {class_name}

    # Check class name first, then scan submodule types
    for hint in _TRANSFORMER_HINTS:
        if any(hint in n for n in all_names):
            return "transformer"
    for hint in _CNN_HINTS:
        if any(hint in n for n in all_names):
            return "cnn"

    # Structural heuristic: if the model has many Linear layers relative to Conv2d, it's
    # likely a transformer-family; if Conv2d dominates, it's a CNN.
    linear_count = sum(1 for m in model.modules() if type(m).__name__ == "Linear")
    conv_count = sum(1 for m in model.modules() if type(m).__name__ == "Conv2d")
    if linear_count > conv_count and linear_count > 3:
        return "transformer"
    if conv_count > 0:
        return "cnn"

    return "unknown"
