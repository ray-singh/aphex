"""Tests for serving config generators."""
from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from infermap.benchmark import BenchmarkResult
from infermap.candidates import DeploymentCandidate
from infermap.deployment import DeploymentConfig
from infermap.inspector import ModelInfo
from infermap.profiler import AcceleratorProfile, CPUProfile, HardwareProfile
from infermap.recommender import Recommendation
from infermap.serving import SUPPORTED_FRAMEWORKS, generate_serving_config
from infermap.serving.triton import TritonGenerator, _triton_backend, _instance_kind
from infermap.serving.torchserve import TorchServeGenerator
from infermap.serving.bentoml import BentoMLGenerator
from infermap.serving.fastapi import FastAPIGenerator


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _config(
    framework: str = "pytorch",
    family: str = "cnn",
    backend: str = "tensorrt_fp16",
    dtype: str = "fp16",
    device: str = "cuda",
    model_path: str = "/models/resnet50.pt",
    batch_size: int = 4,
) -> DeploymentConfig:
    return DeploymentConfig(
        model_path=model_path,
        framework=framework,
        family=family,
        parameters=25_000_000,
        accelerator_kind="cuda",
        accelerator_name="NVIDIA A100-SXM4-80GB",
        accelerator_memory_gb=80.0,
        memory_bandwidth_gbps=2039.0,
        fp16_tflops=312.0,
        backend=backend,
        dtype=dtype,
        device=device,
        batch_size=batch_size,
        description=backend,
        latency_p50_ms=3.5,
        latency_p95_ms=4.0,
        latency_p99_ms=4.5,
        throughput_rps=285.0,
        memory_mb=420.0,
        accuracy_drop=None,
        objective="latency",
        max_latency_ms=None,
        max_memory_mb=None,
        min_throughput_rps=None,
        max_quality_loss=None,
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        system=None,
    )


# ── Registry / dispatch ───────────────────────────────────────────────────────

def test_supported_frameworks_non_empty() -> None:
    assert len(SUPPORTED_FRAMEWORKS) >= 4


def test_supported_frameworks_contains_expected() -> None:
    for fw in ("triton", "torchserve", "bentoml", "fastapi"):
        assert fw in SUPPORTED_FRAMEWORKS


def test_unknown_framework_raises() -> None:
    with pytest.raises(ValueError, match="Unknown serving framework"):
        generate_serving_config("nonexistent", _config(), Path("/tmp"))


def test_generate_returns_paths(tmp_path: Path) -> None:
    paths = generate_serving_config("triton", _config(), tmp_path / "triton")
    assert all(isinstance(p, Path) for p in paths)
    assert len(paths) >= 1


def test_all_frameworks_create_files(tmp_path: Path) -> None:
    cfg = _config()
    for fw in SUPPORTED_FRAMEWORKS:
        out = tmp_path / fw
        paths = generate_serving_config(fw, cfg, out)
        assert all(p.exists() for p in paths), f"{fw} generator created non-existent path"


# ── Triton backend mapping ────────────────────────────────────────────────────

def test_triton_backend_tensorrt() -> None:
    assert _triton_backend("tensorrt_fp16") == "tensorrt_plan"


def test_triton_backend_onnx() -> None:
    assert _triton_backend("onnx_fp32") == "onnxruntime"


def test_triton_backend_pytorch() -> None:
    assert _triton_backend("pytorch_fp16") == "pytorch_libtorch"


def test_triton_backend_xgboost() -> None:
    assert _triton_backend("xgboost_native") == "fil"


def test_triton_backend_lightgbm() -> None:
    assert _triton_backend("lightgbm_native") == "fil"


def test_triton_backend_sklearn() -> None:
    assert _triton_backend("sklearn_native") == "fil"


def test_triton_backend_openvino() -> None:
    assert _triton_backend("openvino_fp32") == "openvino"


def test_triton_backend_unknown_falls_back_to_python() -> None:
    assert _triton_backend("custom_backend") == "python"


def test_triton_instance_kind_cuda() -> None:
    assert _instance_kind("cuda") == "KIND_GPU"


def test_triton_instance_kind_mps() -> None:
    assert _instance_kind("mps") == "KIND_GPU"


def test_triton_instance_kind_cpu() -> None:
    assert _instance_kind("cpu") == "KIND_CPU"


# ── Triton generator ──────────────────────────────────────────────────────────

def test_triton_creates_config_pbtxt(tmp_path: Path) -> None:
    paths = TritonGenerator().generate(_config(), tmp_path)
    assert any(p.name == "config.pbtxt" for p in paths)


def test_triton_config_contains_backend(tmp_path: Path) -> None:
    TritonGenerator().generate(_config(backend="onnx_fp32"), tmp_path)
    content = (tmp_path / "config.pbtxt").read_text()
    assert "onnxruntime" in content


def test_triton_config_contains_model_name(tmp_path: Path) -> None:
    TritonGenerator().generate(_config(model_path="/models/resnet50.pt"), tmp_path)
    content = (tmp_path / "config.pbtxt").read_text()
    assert "resnet50" in content


def test_triton_config_gpu_instance_for_cuda(tmp_path: Path) -> None:
    TritonGenerator().generate(_config(device="cuda"), tmp_path)
    content = (tmp_path / "config.pbtxt").read_text()
    assert "KIND_GPU" in content


def test_triton_config_cpu_instance_for_cpu(tmp_path: Path) -> None:
    TritonGenerator().generate(_config(device="cpu"), tmp_path)
    content = (tmp_path / "config.pbtxt").read_text()
    assert "KIND_CPU" in content


def test_triton_no_dynamic_batching_by_default(tmp_path: Path) -> None:
    TritonGenerator().generate(_config(), tmp_path)
    content = (tmp_path / "config.pbtxt").read_text()
    assert "dynamic_batching" not in content


def test_triton_dynamic_batching_when_system_config_set(tmp_path: Path) -> None:
    from infermap.system_recommender import SystemConfig
    cfg = _config()
    cfg.system = SystemConfig(
        max_safe_batch_size=32,
        recommended_batch_size=8,
        dynamic_batching=True,
        num_workers=4,
        tensor_parallel_size=1,
        kv_cache_fraction=None,
        continuous_batching=None,
        prefill_chunk_size=None,
        kv_cache_dtype=None,
        enable_result_cache=False,
    )
    TritonGenerator().generate(cfg, tmp_path)
    content = (tmp_path / "config.pbtxt").read_text()
    assert "dynamic_batching" in content


def test_triton_max_batch_from_system_config(tmp_path: Path) -> None:
    from infermap.system_recommender import SystemConfig
    cfg = _config(batch_size=4)
    cfg.system = SystemConfig(
        max_safe_batch_size=64,
        recommended_batch_size=8,
        dynamic_batching=False,
        num_workers=4,
        tensor_parallel_size=1,
        kv_cache_fraction=None,
        continuous_batching=None,
        prefill_chunk_size=None,
        kv_cache_dtype=None,
        enable_result_cache=False,
    )
    TritonGenerator().generate(cfg, tmp_path)
    content = (tmp_path / "config.pbtxt").read_text()
    assert "max_batch_size: 64" in content


def test_triton_max_batch_falls_back_to_batch_size(tmp_path: Path) -> None:
    TritonGenerator().generate(_config(batch_size=8), tmp_path)
    content = (tmp_path / "config.pbtxt").read_text()
    assert "max_batch_size: 8" in content


# ── TorchServe generator ──────────────────────────────────────────────────────

def test_torchserve_creates_config_properties(tmp_path: Path) -> None:
    paths = TorchServeGenerator().generate(_config(), tmp_path)
    assert any(p.name == "config.properties" for p in paths)


def test_torchserve_creates_serve_sh(tmp_path: Path) -> None:
    paths = TorchServeGenerator().generate(_config(), tmp_path)
    assert any(p.name == "serve.sh" for p in paths)


def test_torchserve_config_has_inference_address(tmp_path: Path) -> None:
    TorchServeGenerator().generate(_config(), tmp_path)
    content = (tmp_path / "config.properties").read_text()
    assert "inference_address" in content


def test_torchserve_serve_sh_contains_model_name(tmp_path: Path) -> None:
    TorchServeGenerator().generate(_config(model_path="/models/resnet50.pt"), tmp_path)
    content = (tmp_path / "serve.sh").read_text()
    assert "resnet50" in content


def test_torchserve_serve_sh_has_archiver_command(tmp_path: Path) -> None:
    TorchServeGenerator().generate(_config(), tmp_path)
    content = (tmp_path / "serve.sh").read_text()
    assert "torch-model-archiver" in content


# ── BentoML generator ─────────────────────────────────────────────────────────

def test_bentoml_creates_bentofile(tmp_path: Path) -> None:
    paths = BentoMLGenerator().generate(_config(), tmp_path)
    assert any(p.name == "bentofile.yaml" for p in paths)


def test_bentoml_creates_service_py(tmp_path: Path) -> None:
    paths = BentoMLGenerator().generate(_config(), tmp_path)
    assert any(p.name == "service.py" for p in paths)


def test_bentoml_bentofile_has_service_ref(tmp_path: Path) -> None:
    BentoMLGenerator().generate(_config(), tmp_path)
    content = (tmp_path / "bentofile.yaml").read_text()
    assert "service" in content


def test_bentoml_service_py_has_predict_endpoint(tmp_path: Path) -> None:
    BentoMLGenerator().generate(_config(), tmp_path)
    content = (tmp_path / "service.py").read_text()
    assert "def predict" in content


def test_bentoml_sklearn_uses_sklearn_runner(tmp_path: Path) -> None:
    BentoMLGenerator().generate(_config(framework="sklearn", backend="sklearn_native"), tmp_path)
    content = (tmp_path / "service.py").read_text()
    assert "bentoml.sklearn" in content


def test_bentoml_pytorch_uses_pytorch_runner(tmp_path: Path) -> None:
    BentoMLGenerator().generate(_config(framework="pytorch"), tmp_path)
    content = (tmp_path / "service.py").read_text()
    assert "bentoml.pytorch" in content


# ── FastAPI generator ─────────────────────────────────────────────────────────

def test_fastapi_creates_app_py(tmp_path: Path) -> None:
    paths = FastAPIGenerator().generate(_config(), tmp_path)
    assert any(p.name == "app.py" for p in paths)


def test_fastapi_creates_requirements_txt(tmp_path: Path) -> None:
    paths = FastAPIGenerator().generate(_config(), tmp_path)
    assert any(p.name == "requirements.txt" for p in paths)


def test_fastapi_app_has_predict_route(tmp_path: Path) -> None:
    FastAPIGenerator().generate(_config(), tmp_path)
    content = (tmp_path / "app.py").read_text()
    assert '"/predict"' in content


def test_fastapi_requirements_has_fastapi(tmp_path: Path) -> None:
    FastAPIGenerator().generate(_config(), tmp_path)
    content = (tmp_path / "requirements.txt").read_text()
    assert "fastapi" in content


def test_fastapi_requirements_has_framework_package(tmp_path: Path) -> None:
    FastAPIGenerator().generate(_config(framework="xgboost"), tmp_path)
    content = (tmp_path / "requirements.txt").read_text()
    assert "xgboost" in content


def test_fastapi_app_mentions_dtype_and_device(tmp_path: Path) -> None:
    FastAPIGenerator().generate(_config(dtype="int8", device="cuda"), tmp_path)
    content = (tmp_path / "app.py").read_text()
    assert "int8" in content
    assert "cuda" in content
