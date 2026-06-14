"""Model inspector — identifies framework, family, parameter count, memory footprint."""

from __future__ import annotations

import pickle
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
    import io
    import pickle
    import zipfile

    import torch
    import torch.nn as nn
    import torch.storage as _ts

    # TorchScript archives are ZIP files with a specific marker; load them directly
    # to avoid a UserWarning from torch.load's auto-dispatch.
    if zipfile.is_zipfile(path):
        try:
            return torch.jit.load(str(path), map_location="cpu")
        except Exception:
            pass  # fall through to torch.load for non-script zip saves

    # Some pickles store tensors as embedded byte buffers via _load_from_bytes,
    # which bypasses the outer torch.load's map_location (e.g. models saved on CUDA).
    # Patch it for the duration of this load so all storages land on CPU.
    _orig_lfb = _ts._load_from_bytes

    def _lfb_cpu(b: bytes) -> Any:
        return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)

    _ts._load_from_bytes = _lfb_cpu
    try:
        # torch.load handles .pt / .pth (legacy binary and new ZIP formats).
        # Plain Python pickles (.pkl) don't have the torch magic number header,
        # so we fall back to pickle.load with our stub Unpickler for those.
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False,
                             pickle_module=_StubPickle)
        except RuntimeError:
            with open(path, "rb") as fh:
                obj = _StubPickle.Unpickler(fh).load()
    finally:
        _ts._load_from_bytes = _orig_lfb

    # Some saved objects are non-Module wrappers (e.g. training harnesses) that
    # hold the actual network under a `.net` attribute.  Check this first so that
    # stub nn.Module wrappers don't shadow the real sub-network.
    if hasattr(obj, "net") and isinstance(getattr(obj, "net", None), nn.Module):
        model = obj.net  # type: ignore[union-attr]
    elif isinstance(obj, nn.Module):
        model = obj
    else:
        raise ValueError(
            f"Loaded object from {path} is not an nn.Module. "
            "Pass a full model (not just a state dict) or load the architecture first."
        )

    # Stub classes created by _StubPickle have no forward() — inject best-effort
    # implementations so the model can be called and traced for benchmarking.
    _patch_stub_forwards(model)
    return model


class _StubPickle:
    """pickle_module shim: stubs out missing external classes so torch.load
    can reconstruct objects from pickles that depend on unavailable packages."""

    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
    loads = staticmethod(pickle.loads)
    PickleError = pickle.PickleError
    PicklingError = pickle.PicklingError
    UnpicklingError = pickle.UnpicklingError

    @staticmethod
    def load(f: Any, **kwargs: Any) -> Any:
        return _StubPickle.Unpickler(f).load()

    class Unpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                import sys
                import types
                import torch.nn as nn

                # Ensure the stub module exists in sys.modules so that
                # torch.save can later find the class for pickling.
                if module not in sys.modules:
                    sys.modules[module] = types.ModuleType(module)

                stub_mod = sys.modules[module]
                if hasattr(stub_mod, name):
                    return getattr(stub_mod, name)  # type: ignore[no-any-return]

                _nn_keywords = ("net", "block", "layer", "encoder",
                                "decoder", "head", "embed", "attention")
                if any(k in name.lower() for k in _nn_keywords):
                    cls = type(name, (nn.Module,),
                               {"__init__": lambda self: nn.Module.__init__(self),
                                "__module__": module})
                else:
                    cls = type(name, (), {"__module__": module})

                setattr(stub_mod, name, cls)
                return cls


def _patch_stub_forwards(model: Any) -> None:
    """Inject best-effort forward() into stub nn.Modules that have none.

    Stub classes created by _StubPickle.Unpickler lack a forward() definition,
    which makes the model uncallable.  This function walks the module tree and
    injects concrete forward implementations based on the submodule names that
    are present after pickle restoration.
    """
    import types as _types

    import torch
    import torch.nn as nn

    def _is_stub(m: nn.Module) -> bool:
        return "forward" not in type(m).__dict__

    for module in model.modules():
        if not _is_stub(module):
            continue

        sub = set(module._modules)

        # TCN-style residual block: conv1, conv2, relu (drop/downsample optional).
        # Causal TCNs pad both sides but only use the causal (left-padded) portion,
        # so we clip each conv output back to the original sequence length.
        if {"conv1", "conv2", "relu"}.issubset(sub):
            def _tcn_block_fwd(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
                residual = x
                L = x.shape[2]
                out = self.conv1(x)[:, :, :L]
                out = self.relu(out)
                if "drop" in self._modules:
                    out = self.drop(out)
                out = self.conv2(out)[:, :, :L]
                out = self.relu(out)
                if "drop" in self._modules:
                    out = self.drop(out)
                if "downsample" in self._modules:
                    residual = self.downsample(x)
                return self.relu(out + residual)

            module.forward = _types.MethodType(_tcn_block_fwd, module)

        # TCN/CNN trunk with dual prediction heads
        elif "network" in sub and "rv_head_linear" in sub and "shock_head" in sub:
            def _tcn_net_fwd(self: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                out = self.network(x)
                out = out[:, :, -1]  # last time-step
                return self.rv_head_linear(out), self.shock_head(out)

            module.forward = _types.MethodType(_tcn_net_fwd, module)

        # Transformer trunk with CLS token and dual prediction heads
        elif "input_proj" in sub and "encoder" in sub and "rv_head_linear" in sub:
            def _transformer_net_fwd(
                self: nn.Module, x: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                x = self.input_proj(x)
                b = x.shape[0]
                cls = self.cls_token.expand(b, -1, -1)
                x = torch.cat([cls, x], dim=1)
                pos = torch.arange(x.shape[1], device=x.device)
                x = x + self.pos_emb(pos).unsqueeze(0)
                x = self.encoder(x)
                out = x[:, 0]
                return self.rv_head_linear(out), self.shock_head(out)

            module.forward = _types.MethodType(_transformer_net_fwd, module)


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
