from __future__ import annotations

from pathlib import Path

from aphex.deployment import DeploymentConfig
from aphex.serving.base import ServingGenerator

_FRAMEWORK_PACKAGES: dict[str, str] = {
    "pytorch": "torch",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "onnx": "onnxruntime",
}


def _model_name(config: DeploymentConfig) -> str:
    if config.model_path:
        return Path(config.model_path).stem.replace(" ", "_").replace("-", "_")
    return "model"


class FastAPIGenerator(ServingGenerator):
    name = "fastapi"

    def generate(self, config: DeploymentConfig, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        name = _model_name(config)
        batch_size = config.system.recommended_batch_size if config.system else config.batch_size
        fw_pkg = _FRAMEWORK_PACKAGES.get(config.framework, config.framework)

        app_py = "\n".join([
            "from __future__ import annotations",
            "from fastapi import FastAPI",
            "from pydantic import BaseModel",
            "import numpy as np",
            f"# TODO: import your {config.framework} model loading library",
            "",
            "app = FastAPI()",
            "",
            f"# TODO: load {name} from {config.model_path or 'your model path'}",
            "model = None",
            "",
            "",
            "@app.get(\"/health\")",
            "def health() -> dict:",
            "    return {\"status\": \"ok\"}",
            "",
            "",
            "@app.get(\"/ready\")",
            "def ready() -> dict:",
            "    return {\"status\": \"ready\" if model is not None else \"loading\"}",
            "",
            "",
            "class PredictRequest(BaseModel):",
            "    inputs: list  # shape: [batch, ...]",
            "",
            "",
            "class PredictResponse(BaseModel):",
            "    outputs: list",
            "",
            "",
            "@app.post(\"/predict\", response_model=PredictResponse)",
            "def predict(request: PredictRequest) -> PredictResponse:",
            "    inputs = np.array(request.inputs)",
            f"    # Recommended batch size: {batch_size}  dtype: {config.dtype}  device: {config.device}",
            "    outputs = model(inputs)  # type: ignore",
            "    return PredictResponse(outputs=outputs.tolist())",
            "",
            "",
            'if __name__ == "__main__":',
            "    import uvicorn",
            '    uvicorn.run(app, host="0.0.0.0", port=8080)',
            "",
        ])

        requirements = "\n".join([
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "numpy>=1.24.0",
            "pydantic>=2.0.0",
            fw_pkg,
            "",
        ])

        paths = []
        p1 = output_dir / "app.py"
        p1.write_text(app_py)
        paths.append(p1)

        p2 = output_dir / "requirements.txt"
        p2.write_text(requirements)
        paths.append(p2)

        return paths
