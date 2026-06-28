from __future__ import annotations

from pathlib import Path

from aphex.deployment import DeploymentConfig
from aphex.serving.bentoml import BentoMLGenerator
from aphex.serving.fastapi import FastAPIGenerator
from aphex.serving.torchserve import TorchServeGenerator
from aphex.serving.triton import TritonGenerator

_REGISTRY: dict[str, object] = {
    "triton": TritonGenerator(),
    "torchserve": TorchServeGenerator(),
    "bentoml": BentoMLGenerator(),
    "fastapi": FastAPIGenerator(),
}

SUPPORTED_FRAMEWORKS: list[str] = list(_REGISTRY)


def generate_serving_config(
    framework: str,
    config: DeploymentConfig,
    output_dir: Path,
) -> list[Path]:
    gen = _REGISTRY.get(framework)
    if gen is None:
        supported = ", ".join(_REGISTRY)
        raise ValueError(f"Unknown serving framework {framework!r}. Supported: {supported}")
    from aphex.serving.base import ServingGenerator
    assert isinstance(gen, ServingGenerator)
    return gen.generate(config, output_dir)
