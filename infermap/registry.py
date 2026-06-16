"""Plugin registry — maps model inputs to the right adapter."""

from __future__ import annotations

from typing import Any

from infermap.plugin import ModelPlugin

_plugins: list[ModelPlugin] = []


def register(plugin: ModelPlugin) -> None:
    """Register a plugin. Plugins registered first take higher priority."""
    _plugins.append(plugin)


def get_plugin(path_or_model: Any) -> ModelPlugin:
    """Return the first registered plugin that can handle this input."""
    for plugin in _plugins:
        if plugin.can_handle(path_or_model):
            return plugin
    raise ValueError(
        f"No plugin can handle {path_or_model!r}. "
        "Install a framework plugin or register one with infermap.registry.register()."
    )


def _register_defaults() -> None:
    from infermap.plugins.pytorch import PytorchPlugin
    register(PytorchPlugin())

    from infermap.plugins.sklearn import SklearnPlugin
    register(SklearnPlugin())

    from infermap.plugins.llm import LLMPlugin
    register(LLMPlugin())


_register_defaults()
