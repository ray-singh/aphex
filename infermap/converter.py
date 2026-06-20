"""Convert a model to its target deployment format and write the artifact(s) to disk.

Supported backends
------------------
pytorch_fp32 / fp16 / bf16      → .pt  (cast weights, save with torch.save)
pytorch_int8_dynamic            → .pt  (quantize_dynamic, save with torch.save)
onnx_cpu / onnx_cuda / coreml  → .onnx  (PyTorch or sklearn/XGBoost/LightGBM/CatBoost)
onnx_int8_cpu                   → .onnx (PyTorch FP32 export → onnxruntime dynamic quant)
tensorrt_fp32 / fp16 / int8     → .engine (requires tensorrt)
openvino_fp32 / int8            → .xml + .bin (requires openvino[, nncf])
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from infermap.benchmark import (
    _ensure_quantization_engine,
    _export_to_onnx_bytes,
    _safe_quantize_dynamic,
)

_PYTORCH_BACKENDS = frozenset({
    "pytorch_fp32",
    "pytorch_fp16",
    "pytorch_bf16",
    "pytorch_int8_dynamic",
})
_ONNX_BACKENDS = frozenset({
    "onnx_cpu",
    "onnx_cuda",
    "onnx_coreml",
    "onnx_int8_cpu",
})
_TRT_BACKENDS = frozenset({
    "tensorrt_fp32",
    "tensorrt_fp16",
    "tensorrt_int8",
})
_OV_BACKENDS = frozenset({
    "openvino_fp32",
    "openvino_int8",
})

ALL_CONVERTIBLE: frozenset[str] = _PYTORCH_BACKENDS | _ONNX_BACKENDS | _TRT_BACKENDS | _OV_BACKENDS

_EXT_MAP: dict[str, str] = {
    "pytorch_fp32": ".pt",
    "pytorch_fp16": ".pt",
    "pytorch_bf16": ".pt",
    "pytorch_int8_dynamic": ".pt",
    "onnx_cpu": ".onnx",
    "onnx_cuda": ".onnx",
    "onnx_coreml": ".onnx",
    "onnx_int8_cpu": ".onnx",
    "tensorrt_fp32": ".engine",
    "tensorrt_fp16": ".engine",
    "tensorrt_int8": ".engine",
    "openvino_fp32": ".xml",
    "openvino_int8": ".xml",
}


def convert(
    model: Any,
    backend: str,
    input_shape: list[int],
    output_path: Path,
    calibration_inputs: list[Any] | None = None,
) -> list[Path]:
    """Export *model* to *backend* format and write to *output_path*.

    *model* may be a ``torch.nn.Module`` or a traditional ML model
    (sklearn, XGBoost, LightGBM, CatBoost).  ONNX backends support both;
    PyTorch/TensorRT/OpenVINO backends require ``nn.Module``.

    Returns a list of written file paths (OpenVINO produces .xml + .bin).
    Raises ValueError for unsupported backends, RuntimeError on conversion failure.
    """
    if backend in _PYTORCH_BACKENDS:
        return _convert_pytorch(model, backend, output_path)
    if backend in _ONNX_BACKENDS:
        return _convert_onnx(model, backend, input_shape, output_path)
    if backend in _TRT_BACKENDS:
        return _convert_tensorrt(model, backend, input_shape, output_path, calibration_inputs)
    if backend in _OV_BACKENDS:
        return _convert_openvino(model, backend, input_shape, output_path, calibration_inputs)
    raise ValueError(
        f"Unsupported backend for conversion: {backend!r}. "
        f"Convertible backends: {sorted(ALL_CONVERTIBLE)}"
    )


def default_output_path(model_path: Path, backend: str) -> Path:
    """Derive a sensible output artifact path from the model path and backend."""
    ext = _EXT_MAP.get(backend, ".bin")
    return model_path.parent / f"{model_path.stem}_{backend}{ext}"


# ── backend converters ────────────────────────────────────────────────────────


def _convert_pytorch(model: Any, backend: str, output_path: Path) -> list[Path]:
    import torch
    import torch.nn as nn
    m = copy.deepcopy(model).cpu().eval()

    if backend == "pytorch_fp16":
        m = m.half()
    elif backend == "pytorch_bf16":
        m = m.to(torch.bfloat16)
    elif backend == "pytorch_int8_dynamic":
        if isinstance(m, torch.jit.ScriptModule):
            raise RuntimeError(
                "pytorch_int8_dynamic is not compatible with TorchScript models. "
                "Save an eager nn.Module instead."
            )
        _ensure_quantization_engine()
        m = _safe_quantize_dynamic(m)
    # pytorch_fp32: save as-is

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m, output_path)
    return [output_path]


def _convert_onnx(
    model: Any,
    backend: str,
    input_shape: list[int],
    output_path: Path,
) -> list[Path]:
    import torch
    import torch.nn as nn
    if not isinstance(model, nn.Module):
        return _convert_sklearn_onnx(model, backend, input_shape, output_path)

    m = copy.deepcopy(model).cpu().eval().float()
    dummy = torch.zeros(1, *input_shape)
    onnx_bytes = _export_to_onnx_bytes(m, dummy)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if backend == "onnx_int8_cpu":
        import logging
        import tempfile
        import warnings

        from onnxruntime.quantization import QuantType, quantize_dynamic

        with tempfile.TemporaryDirectory() as tmpdir:
            fp32_path = Path(tmpdir) / "model.onnx"
            fp32_path.write_bytes(onnx_bytes)
            ort_logger = logging.getLogger("onnxruntime")
            prev_level = ort_logger.level
            ort_logger.setLevel(logging.ERROR)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    quantize_dynamic(str(fp32_path), str(output_path), weight_type=QuantType.QInt8)
            finally:
                ort_logger.setLevel(prev_level)
    else:
        output_path.write_bytes(onnx_bytes)

    return [output_path]


def _convert_sklearn_onnx(
    model: Any,
    backend: str,
    input_shape: list[int],
    output_path: Path,
) -> list[Path]:
    if backend == "onnx_int8_cpu":
        raise RuntimeError(
            "onnx_int8_cpu is not supported for traditional ML models. "
            "Use onnx_cpu instead — sklearn/tree models export directly to a "
            "portable ONNX graph without a separate quantization step."
        )

    from infermap.plugins.sklearn import _detect_framework, _export_to_onnx, _get_n_features

    framework = _detect_framework(model)
    n_features = input_shape[0] if input_shape else (_get_n_features(model) or 1)
    try:
        onnx_bytes = _export_to_onnx(model, framework, n_features)
    except ImportError as exc:
        raise RuntimeError(
            f"ONNX export for {framework} models requires an additional package. "
            f"Install it with: pip install skl2onnx  # for sklearn\n"
            f"  or: pip install onnxmltools  # for LightGBM\n"
            f"Original error: {exc}"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_bytes)
    return [output_path]


def _convert_tensorrt(
    model: Any,
    backend: str,
    input_shape: list[int],
    output_path: Path,
    calibration_inputs: list[Any] | None = None,
) -> list[Path]:
    import torch
    import tensorrt as trt

    from infermap.benchmark import _make_trt_calibrator

    if backend == "tensorrt_int8" and not calibration_inputs:
        raise RuntimeError(
            "tensorrt_int8 requires calibration data. "
            "Pass --calibration-data to supply representative inputs."
        )

    m = copy.deepcopy(model).cpu().eval()
    dummy = torch.zeros(1, *input_shape)
    onnx_bytes = _export_to_onnx_bytes(m, dummy)

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        errors = [parser.get_error(i).desc() for i in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT ONNX parse failed: {'; '.join(errors)}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if backend == "tensorrt_fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif backend == "tensorrt_int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _make_trt_calibrator(calibration_inputs)  # type: ignore[arg-type]

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed — check GPU capabilities.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(plan))
    return [output_path]


def _convert_openvino(
    model: Any,
    backend: str,
    input_shape: list[int],
    output_path: Path,
    calibration_inputs: list[Any] | None = None,
) -> list[Path]:
    import io
    import torch
    import openvino as ov

    if backend == "openvino_int8" and not calibration_inputs:
        raise RuntimeError(
            "openvino_int8 requires calibration data. "
            "Pass --calibration-data to supply representative inputs."
        )

    m = copy.deepcopy(model).cpu().eval().float()
    dummy = torch.zeros(1, *input_shape)
    onnx_bytes = _export_to_onnx_bytes(m, dummy)
    ov_model = ov.convert_model(io.BytesIO(onnx_bytes))

    if backend == "openvino_int8":
        try:
            import nncf

            def _transform(item: Any) -> dict:
                return {"input": item.float().cpu().numpy()}

            dataset = nncf.Dataset(calibration_inputs, _transform)
            ov_model = nncf.quantize(ov_model, dataset)
        except ImportError as exc:
            raise RuntimeError(
                "openvino_int8 requires the nncf package: pip install nncf"
            ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path = output_path.with_suffix(".xml")
    ov.save_model(ov_model, str(xml_path))

    written = [xml_path]
    bin_path = xml_path.with_suffix(".bin")
    if bin_path.exists():
        written.append(bin_path)
    return written


# ── deployment.yaml reader ────────────────────────────────────────────────────


_KNOWN_TOP_LEVEL_SECTIONS = frozenset(
    {"model", "hardware", "recommendation", "performance",
     "constraints", "meta", "serving"}
)


def read_deployment_yaml(path: Path, *, strict: bool = False) -> dict:
    """Parse a deployment.yaml written by aphex optimize.

    Uses a tiny hand-rolled parser so no third-party yaml library is required.
    Only handles the subset our writer produces: nested dicts of scalar values.

    Validates that the file is structurally a deployment.yaml: the top level
    must be a mapping that contains at least one of the known sections. If
    ``strict=True``, unknown top-level keys raise; otherwise a warning is
    emitted via the logging module and parsing continues.
    """
    import logging
    _log = logging.getLogger("infermap.converter")

    result: dict = {}
    stack: list[tuple[int, dict]] = [(-1, result)]

    for line_no, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        content = line.strip()

        if ":" not in content:
            raise ValueError(
                f"{path}:{line_no}: malformed line (no ':' separator): {content!r}"
            )

        key, _, rest = content.partition(":")
        key = key.strip()
        rest = rest.strip()

        # Pop to the right parent
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()

        parent = stack[-1][1]

        if not rest:
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(rest)

    if not result:
        raise ValueError(f"{path}: empty or not a YAML mapping")

    known_present = _KNOWN_TOP_LEVEL_SECTIONS & set(result)
    if not known_present:
        raise ValueError(
            f"{path}: does not look like a deployment.yaml written by aphex "
            f"(no recognised top-level section among "
            f"{sorted(_KNOWN_TOP_LEVEL_SECTIONS)})"
        )
    unknown = set(result) - _KNOWN_TOP_LEVEL_SECTIONS
    if unknown:
        msg = f"{path}: unknown top-level section(s): {sorted(unknown)}"
        if strict:
            raise ValueError(msg)
        _log.warning(msg)

    return result


def _parse_scalar(val: str) -> Any:
    if val == "null":
        return None
    if val == "true":
        return True
    if val == "false":
        return False
    if val.startswith('"') and val.endswith('"'):
        return val[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val
