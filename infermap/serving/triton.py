from __future__ import annotations

from pathlib import Path

from infermap.deployment import DeploymentConfig
from infermap.serving.base import ServingGenerator

# Maps aphex backend prefixes to Triton backend names
_BACKEND_MAP: list[tuple[str, str]] = [
    ("tensorrt", "tensorrt_plan"),
    ("onnx", "onnxruntime"),
    ("pytorch", "pytorch_libtorch"),
    ("torchscript", "pytorch_libtorch"),
    ("xgboost", "fil"),
    ("lightgbm", "fil"),
    ("sklearn", "fil"),
    ("openvino", "openvino"),
    ("catboost", "python"),
    ("coreml", "python"),
    ("ctransformers", "python"),
    ("vllm", "python"),
]

_DTYPE_MAP: dict[str, str] = {
    "fp32": "TYPE_FP32",
    "fp16": "TYPE_FP16",
    "bf16": "TYPE_BF16",
    "int8": "TYPE_INT8",
    "int4": "TYPE_INT8",
    "int32": "TYPE_INT32",
}


def _triton_backend(backend: str) -> str:
    lower = backend.lower()
    for prefix, triton_be in _BACKEND_MAP:
        if lower.startswith(prefix):
            return triton_be
    return "python"


def _instance_kind(device: str) -> str:
    return "KIND_GPU" if device in ("cuda", "mps") else "KIND_CPU"


def _data_type(dtype: str) -> str:
    return _DTYPE_MAP.get(dtype.lower(), "TYPE_FP32")


def _model_name(config: DeploymentConfig) -> str:
    if config.model_path:
        return Path(config.model_path).stem.replace(" ", "_").replace("-", "_")
    return "model"


class TritonGenerator(ServingGenerator):
    name = "triton"

    def generate(self, config: DeploymentConfig, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)

        name = _model_name(config)
        backend = _triton_backend(config.backend)
        data_type = _data_type(config.dtype)
        max_batch = config.system.max_safe_batch_size if config.system else config.batch_size
        dynamic = config.system.dynamic_batching if config.system else False
        kind = _instance_kind(config.device)

        lines = [
            f'name: "{name}"',
            f'backend: "{backend}"',
            f"max_batch_size: {max_batch}",
            "",
            "input [",
            "  {",
            '    name: "input_0"',
            f"    data_type: {data_type}",
            "    dims: [ -1 ]  # TODO: replace with actual input shape",
            "  }",
            "]",
            "output [",
            "  {",
            '    name: "output_0"',
            f"    data_type: {data_type}",
            "    dims: [ -1 ]  # TODO: replace with actual output shape",
            "  }",
            "]",
        ]

        if dynamic:
            lines += [
                "",
                "dynamic_batching {",
                "  max_queue_delay_microseconds: 100",
                "}",
            ]

        lines += [
            "",
            "instance_group [",
            "  {",
            "    count: 1",
            f"    kind: {kind}",
            "  }",
            "]",
            "",
        ]

        p = output_dir / "config.pbtxt"
        p.write_text("\n".join(lines))
        return [p]
