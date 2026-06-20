from __future__ import annotations

from pathlib import Path

from infermap.deployment import DeploymentConfig
from infermap.serving.bentoml import BentoMLGenerator
from infermap.serving.fastapi import FastAPIGenerator
from infermap.serving.torchserve import TorchServeGenerator
from infermap.serving.triton import TritonGenerator

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
    from infermap.serving.base import ServingGenerator
    assert isinstance(gen, ServingGenerator)
    return gen.generate(config, output_dir)
