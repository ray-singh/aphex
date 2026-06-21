"""Model inspector — identifies framework, family, parameter count, memory footprint."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("infermap.inspector")

ModelFamily = str  # e.g. "transformer", "cnn", "tree_ensemble", "unknown"
Framework = str   # e.g. "pytorch", "sklearn", "xgboost", "unknown"

# (module_name, class_name, is_nn_module) for every stub class created by _StubPickle.
# Workers that need to unpickle stub models read this to recreate the same classes.
_STUB_REGISTRY: list[tuple[str, str, bool]] = []

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
    # Architecture fingerprint (V2) — populated by framework-specific inspectors
    num_layers: int | None = None
    num_attention_heads: int | None = None
    hidden_size: int | None = None
    vocab_size: int | None = None
    max_sequence_length: int | None = None
    num_conv_layers: int | None = None
    num_features: int | None = None  # sklearn: n_features_in_
    num_trees: int | None = None     # tree ensembles
    max_depth: int | None = None     # tree ensembles

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
    fp = _extract_pytorch_fingerprint(model)

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
        num_layers=fp.get("num_layers"),
        num_attention_heads=fp.get("num_attention_heads"),
        hidden_size=fp.get("hidden_size"),
        vocab_size=fp.get("vocab_size"),
        num_conv_layers=fp.get("num_conv_layers"),
    )


def _extract_pytorch_fingerprint(model: Any) -> dict[str, Any]:
    """Extract architecture fingerprint fields from a PyTorch nn.Module."""
    import torch.nn as nn

    result: dict[str, Any] = {}

    attn_mods = [m for m in model.modules() if isinstance(m, nn.MultiheadAttention)]
    if attn_mods:
        result["num_layers"] = len(attn_mods)
        result["num_attention_heads"] = attn_mods[0].num_heads
        result["hidden_size"] = attn_mods[0].embed_dim

    emb_mods = [m for m in model.modules() if isinstance(m, nn.Embedding)]
    if emb_mods:
        result["vocab_size"] = max(e.num_embeddings for e in emb_mods)

    n_conv = sum(
        1 for m in model.modules()
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
    )
    if n_conv:
        result["num_conv_layers"] = n_conv

    return result


def _load_model(path: Path) -> Any:
    import io
    import zipfile

    import torch
    import torch.nn as nn
    import torch.storage as _ts

    # TorchScript archives are ZIP files with a specific marker; load them directly
    # to avoid a UserWarning from torch.load's auto-dispatch.
    if zipfile.is_zipfile(path):
        try:
            return torch.jit.load(str(path), map_location="cpu")
        except Exception as exc:
            logger.debug("jit.load failed, falling back to torch.load: %s", exc)

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
        model = obj.net
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
                    return getattr(stub_mod, name)

                _nn_keywords = ("net", "block", "layer", "encoder",
                                "decoder", "head", "embed", "attention")
                is_nn = any(k in name.lower() for k in _nn_keywords)
                if is_nn:
                    cls = type(name, (nn.Module,),
                               {"__init__": lambda self: nn.Module.__init__(self),
                                "__module__": module})
                else:
                    cls = type(name, (), {"__module__": module})

                setattr(stub_mod, name, cls)
                _STUB_REGISTRY.append((module, name, is_nn))
                return cls


_RESIDUAL_SUBMODULE_NAMES = frozenset({"downsample", "shortcut", "residual", "skip"})


def _patch_stub_forwards(model: Any) -> None:
    """Inject best-effort forward() into stub nn.Modules that have none.

    Stub classes created by _StubPickle.Unpickler preserve weights and submodule
    structure but lose the Python forward() code.  This function reconstructs a
    best-effort forward using two tiers:

    Tier 1 — leaf stubs (no child modules): infer layer type from parameter shapes.
      weight 2D → Linear  |  3D → Conv1d  |  4D → Conv2d
      1D weight + running_mean buffer → BatchNorm  |  1D weight only → LayerNorm
      no weight → identity (dropout, activation stubs, etc.)

    Tier 2 — container stubs (has child modules):
      has downsample/shortcut/skip/residual child → residual(input) + sequential(rest)
      otherwise → sequential: call each child in _modules order

    Architecture-specific overrides run first for patterns where generic inference
    would produce wrong results (e.g. causal conv length clipping, dual-head routing,
    CLS token insertion).  The generic tiers handle everything else.

    Forwards are attached to the CLASS (not the instance) so that torch.compile,
    deepcopy, and quantize_dynamic all work correctly.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _is_stub(m: nn.Module) -> bool:
        return "forward" not in type(m).__dict__

    for module in model.modules():
        if not _is_stub(module):
            continue

        sub = module._modules
        sub_keys = set(sub.keys())
        own_params = {n: p for n, p in module.named_parameters(recurse=False)}
        own_buffers = {n: b for n, b in module.named_buffers(recurse=False)}

        # ── Architecture-specific overrides ───────────────────────────────────
        # These run first because they handle routing that generic inference
        # can't reconstruct (causal clipping, dual heads, CLS tokens, etc.).

        # Causal TCN residual block: conv outputs are over-padded; clip to input
        # length before the residual add, or shapes will mismatch.
        if {"conv1", "conv2", "relu"}.issubset(sub_keys):
            def _tcn_block_fwd(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
                residual = x
                L = x.shape[2]
                out = self.conv1(x)[:, :, :L]  # type: ignore[operator]
                out = self.relu(out)  # type: ignore[operator]
                if "drop" in self._modules:
                    out = self.drop(out)  # type: ignore[operator]
                out = self.conv2(out)[:, :, :L]  # type: ignore[operator]
                out = self.relu(out)  # type: ignore[operator]
                if "drop" in self._modules:
                    out = self.drop(out)  # type: ignore[operator]
                if "downsample" in self._modules:
                    residual = self.downsample(x)  # type: ignore[operator]
                return self.relu(out + residual)  # type: ignore[operator]
            type(module).forward = _tcn_block_fwd
            continue

        # Sequential trunk → last-timestep slice → two parallel heads.
        if "network" in sub_keys and "rv_head_linear" in sub_keys and "shock_head" in sub_keys:
            def _tcn_net_fwd(self: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                out = self.network(x)[:, :, -1]  # type: ignore[operator]
                return self.rv_head_linear(out), self.shock_head(out)  # type: ignore[operator]
            type(module).forward = _tcn_net_fwd
            continue

        # Transformer trunk: CLS token + positional embedding + encoder + dual heads.
        if "input_proj" in sub_keys and "encoder" in sub_keys and "rv_head_linear" in sub_keys:
            def _transformer_net_fwd(
                self: nn.Module, x: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                x = self.input_proj(x)  # type: ignore[operator]
                b = x.shape[0]
                cls = self.cls_token.expand(b, -1, -1)  # type: ignore[operator]
                x = torch.cat([cls, x], dim=1)
                x = x + self.pos_emb.weight[:x.shape[1]].unsqueeze(0)  # type: ignore[union-attr,index]
                x = self.encoder(x)  # type: ignore[operator]
                out = x[:, 0]
                return self.rv_head_linear(out), self.shock_head(out)  # type: ignore[operator]
            type(module).forward = _transformer_net_fwd
            continue

        # ── Tier 1: leaf stubs (no child modules) ─────────────────────────────
        if not sub:
            w = own_params.get("weight")
            if w is None:
                def _identity_fwd(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
                    return x
                type(module).forward = _identity_fwd

            elif w.dim() == 1:
                if "running_mean" in own_buffers:
                    def _bn_fwd(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
                        return F.batch_norm(
                            x, self.running_mean, self.running_var,  # type: ignore[arg-type]
                            self.weight, self._parameters.get("bias"),  # type: ignore[arg-type]
                            False, 0.1, 1e-5,
                        )
                    type(module).forward = _bn_fwd
                else:
                    def _ln_fwd(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
                        return F.layer_norm(
                            x, self.weight.shape,  # type: ignore[arg-type]
                            self.weight, self._parameters.get("bias"),  # type: ignore[arg-type]
                        )
                    type(module).forward = _ln_fwd

            elif w.dim() == 2:
                def _linear_fwd(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
                    return F.linear(x, self.weight, self._parameters.get("bias"))  # type: ignore[arg-type]
                type(module).forward = _linear_fwd

            elif w.dim() == 3:
                def _conv1d_fwd(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
                    return F.conv1d(  # type: ignore[misc]
                        x, self.weight, self._parameters.get("bias"),  # type: ignore[arg-type]
                        getattr(self, "stride", (1,)),
                        getattr(self, "padding", (0,)),
                        getattr(self, "dilation", (1,)),
                        getattr(self, "groups", 1),
                    )
                type(module).forward = _conv1d_fwd

            elif w.dim() == 4:
                def _conv2d_fwd(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
                    return F.conv2d(  # type: ignore[misc]
                        x, self.weight, self._parameters.get("bias"),  # type: ignore[arg-type]
                        getattr(self, "stride", (1, 1)),
                        getattr(self, "padding", (0, 0)),
                        getattr(self, "dilation", (1, 1)),
                        getattr(self, "groups", 1),
                    )
                type(module).forward = _conv2d_fwd

            else:
                def _identity_fwd(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
                    return x
                type(module).forward = _identity_fwd

            continue

        # ── Tier 2: container stubs (has child modules) ───────────────────────
        residual_keys = _RESIDUAL_SUBMODULE_NAMES & sub_keys
        if residual_keys:
            rk = next(iter(residual_keys))

            def _residual_fwd(self: nn.Module, x: torch.Tensor, _rk: str = rk) -> torch.Tensor:
                branch = getattr(self, _rk)
                residual = branch(x) if branch is not None else x
                out = x
                for name, child in self._modules.items():
                    if name == _rk:
                        continue
                    out = child(out)  # type: ignore[misc]
                return out + residual

            type(module).forward = _residual_fwd

        else:
            def _sequential_fwd(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
                for child in self._modules.values():
                    x = child(x)  # type: ignore[misc]
                return x

            type(module).forward = _sequential_fwd


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
