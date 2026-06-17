"""Tests for the plugin registry — routing and custom registration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import infermap.registry as reg
from infermap.plugins.llm import LLMPlugin
from infermap.plugins.pytorch import PytorchPlugin


def _pt(tmp_path: Path) -> Path:
    p = tmp_path / "model.pt"
    torch.save(nn.Linear(2, 2), p)
    return p


# ── Default routing ────────────────────────────────────────────────────────────


def test_pt_file_routes_to_pytorch_plugin(tmp_path: Path) -> None:
    assert isinstance(reg.get_plugin(_pt(tmp_path)), PytorchPlugin)


def test_pth_file_routes_to_pytorch_plugin(tmp_path: Path) -> None:
    p = tmp_path / "model.pth"
    torch.save(nn.Linear(2, 2), p)
    assert isinstance(reg.get_plugin(p), PytorchPlugin)


def test_nn_module_routes_to_pytorch_plugin() -> None:
    assert isinstance(reg.get_plugin(nn.Linear(4, 2)), PytorchPlugin)


def test_gguf_file_routes_to_llm_plugin(tmp_path: Path) -> None:
    p = tmp_path / "model.gguf"
    p.write_bytes(b"GGUF" + b"\x00" * 16)
    assert isinstance(reg.get_plugin(p), LLMPlugin)


def test_hf_dir_with_llm_config_routes_to_llm_plugin(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"]})
    )
    assert isinstance(reg.get_plugin(str(tmp_path)), LLMPlugin)


def test_unknown_extension_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No plugin"):
        reg.get_plugin(tmp_path / "model.xyz")


def test_unknown_object_raises_value_error() -> None:
    with pytest.raises(ValueError, match="No plugin"):
        reg.get_plugin({"this": "is not a model"})


# ── Custom plugin registration ────────────────────────────────────────────────


class _AlwaysPlugin:
    name = "always"

    def can_handle(self, x: Any) -> bool:
        return True

    def inspect(self, *a: Any, **kw: Any) -> None:
        return None

    def load(self, *a: Any, **kw: Any) -> None:
        return None

    def generate_candidates(self, *a: Any, **kw: Any) -> list:
        return []

    def benchmark(self, *a: Any, **kw: Any) -> None:
        return None


def test_custom_plugin_at_front_takes_priority() -> None:
    original = reg._plugins[:]
    try:
        dummy = _AlwaysPlugin()
        reg._plugins.insert(0, dummy)
        assert reg.get_plugin("anything") is dummy
    finally:
        reg._plugins[:] = original


def test_register_appends_to_plugin_list() -> None:
    original = reg._plugins[:]
    try:
        dummy = _AlwaysPlugin()
        before = len(reg._plugins)
        reg.register(dummy)
        assert len(reg._plugins) == before + 1
        assert reg._plugins[-1] is dummy
    finally:
        reg._plugins[:] = original


def test_default_plugins_always_restored_after_custom_plugin_test() -> None:
    """Confirm the teardown in the tests above correctly restores state."""
    assert any(isinstance(p, PytorchPlugin) for p in reg._plugins)
    assert any(isinstance(p, LLMPlugin) for p in reg._plugins)


# ── Plugin name uniqueness ────────────────────────────────────────────────────


def test_default_plugin_names_are_unique() -> None:
    names = [p.name for p in reg._plugins]
    assert len(names) == len(set(names))
