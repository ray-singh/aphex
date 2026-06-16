from __future__ import annotations
from pathlib import Path
from infermap.deployment import DeploymentConfig
from infermap.serving.base import ServingGenerator

_HANDLER_MAP: dict[str, str] = {
    "cnn": "image_classifier",
    "transformer": "text_classifier",
    "llm": "text_classifier",
    "resnet": "image_classifier",
    "vit": "image_classifier",
}


def _handler(family: str) -> str:
    return _HANDLER_MAP.get(family.lower(), "default")


def _model_name(config: DeploymentConfig) -> str:
    if config.model_path:
        return Path(config.model_path).stem.replace(" ", "_").replace("-", "_")
    return "model"


class TorchServeGenerator(ServingGenerator):
    name = "torchserve"

    def generate(self, config: DeploymentConfig, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        name = _model_name(config)
        num_workers = config.system.num_workers if config.system else 4
        handler = _handler(config.family)
        model_file = config.model_path or f"{name}.pt"

        props = "\n".join([
            "inference_address=http://0.0.0.0:8080",
            "management_address=http://0.0.0.0:8081",
            "metrics_address=http://0.0.0.0:8082",
            f"number_of_netty_threads={num_workers}",
            "job_queue_size=100",
            "model_store=./model_store",
            "load_models=all",
            "",
        ])

        serve_sh = "\n".join([
            "#!/usr/bin/env bash",
            "set -e",
            "",
            "mkdir -p model_store",
            "",
            "torch-model-archiver \\",
            f'  --model-name "{name}" \\',
            "  --version 1.0 \\",
            f'  --serialized-file "{model_file}" \\',
            f"  --handler {handler} \\",
            "  --export-path ./model_store",
            "",
            "torchserve --start \\",
            "  --ncs \\",
            "  --model-store ./model_store \\",
            f'  --models "{name}={name}.mar" \\',
            "  --ts-config config.properties",
            "",
        ])

        paths = []
        p1 = output_dir / "config.properties"
        p1.write_text(props)
        paths.append(p1)

        p2 = output_dir / "serve.sh"
        p2.write_text(serve_sh)
        p2.chmod(0o755)
        paths.append(p2)

        return paths
