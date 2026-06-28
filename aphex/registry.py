"""Plugin registry — maps model inputs to the right adapter."""

from __future__ import annotations

from typing import Any

from aphex.plugin import ModelPlugin

_plugins: list[ModelPlugin] = []
_defaults_loaded = False


def register(plugin: ModelPlugin) -> None:
    """Register a plugin. Plugins registered first take higher priority."""
    _plugins.append(plugin)


def get_plugin(path_or_model: Any) -> ModelPlugin:
    """Return the first registered plugin that can handle this input.

    Loads the built-in plugins lazily on first use so importing this module
    doesn't pull in torch / sklearn / llm dependencies as a side effect.
    """
    if not _defaults_loaded:
        load_default_plugins()
    for plugin in _plugins:
        if plugin.can_handle(path_or_model):
            return plugin
    raise ValueError(
        f"No plugin can handle {path_or_model!r}. "
        "Install a framework plugin or register one with aphex.registry.register()."
    )


def load_default_plugins() -> None:
    """Register the built-in plugins. Idempotent; safe to call multiple times.

    Called automatically by ``get_plugin`` on first use; can also be called
    explicitly by the CLI entry point or by tests that want plugins ready.
    """
    global _defaults_loaded
    if _defaults_loaded:
        return
    from aphex.plugins.pytorch import PytorchPlugin
    register(PytorchPlugin())

    from aphex.plugins.sklearn import SklearnPlugin
    register(SklearnPlugin())

    from aphex.plugins.llm import LLMPlugin
    register(LLMPlugin())
    _defaults_loaded = True


def reset() -> None:
    """Clear all registered plugins. For tests only."""
    global _defaults_loaded
    _plugins.clear()
    _defaults_loaded = False
