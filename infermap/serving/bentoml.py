from __future__ import annotations

from pathlib import Path

from infermap.deployment import DeploymentConfig
from infermap.serving.base import ServingGenerator

_RUNNER_MAP: dict[str, str] = {
    "pytorch": "bentoml.pytorch",
    "sklearn": "bentoml.sklearn",
    "xgboost": "bentoml.xgboost",
    "lightgbm": "bentoml.lightgbm",
    "catboost": "bentoml.catboost",
}

_PACKAGE_MAP: dict[str, list[str]] = {
    "pytorch": ["torch", "torchvision"],
    "sklearn": ["scikit-learn"],
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm"],
    "catboost": ["catboost"],
}


def _model_name(config: DeploymentConfig) -> str:
    if config.model_path:
        return Path(config.model_path).stem.replace(" ", "_").replace("-", "_")
    return "model"


class BentoMLGenerator(ServingGenerator):
    name = "bentoml"

    def generate(self, config: DeploymentConfig, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        name = _model_name(config)
        runner_module = _RUNNER_MAP.get(config.framework, "bentoml.picklable_model")
        packages = _PACKAGE_MAP.get(config.framework, [config.framework])
        batch_size = config.system.recommended_batch_size if config.system else config.batch_size

        bentofile_lines = [
            'service: "service:svc"',
            "include:",
            "  - service.py",
            "python:",
            "  packages:",
            "    - bentoml",
        ]
        for pkg in packages:
            bentofile_lines.append(f"    - {pkg}")
        bentofile_lines.append("")

        service_py = "\n".join([
            "from __future__ import annotations",
            "import numpy as np",
            "import bentoml",
            f'import {runner_module.split(".")[0]}',
            "",
            f'runner = {runner_module}.get("{name}:latest").to_runner()',
            f'svc = bentoml.Service("{name}", runners=[runner])',
            "",
            "",
            "@svc.api(",
            "    input=bentoml.io.NumpyNdarray(),",
            "    output=bentoml.io.NumpyNdarray(),",
            f"    max_batch_size={batch_size},",
            ")",
            "def predict(input_data: np.ndarray) -> np.ndarray:",
            "    return runner.run(input_data)",
            "",
        ])

        paths = []
        p1 = output_dir / "bentofile.yaml"
        p1.write_text("\n".join(bentofile_lines))
        paths.append(p1)

        p2 = output_dir / "service.py"
        p2.write_text(service_py)
        paths.append(p2)

        return paths
