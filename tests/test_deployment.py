"""Tests for deployment config generation."""
from __future__ import annotations

from pathlib import Path

import pytest

from infermap.benchmark import BenchmarkResult
from infermap.candidates import DeploymentCandidate
from infermap.deployment import (
    DeploymentConfig,
    _scalar,
    build_config,
    config_to_dict,
    write_yaml,
)
from infermap.inspector import ModelInfo
from infermap.profiler import AcceleratorProfile, CPUProfile, HardwareProfile
from infermap.recommender import Recommendation

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _cand(backend: str = "tensorrt_fp16", dtype: str = "fp16", device: str = "cuda") -> DeploymentCandidate:
    return DeploymentCandidate(
        backend=backend, dtype=dtype,
        description="TensorRT FP16 (CUDA)",
        requires_export=True, device=device,
    )


def _result(
    backend: str = "tensorrt_fp16",
    latency: float = 3.5,
    memory: float = 420.0,
    accuracy_drop: float | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        candidate=_cand(backend),
        latency_p50_ms=latency,
        latency_p95_ms=latency * 1.1,
        latency_p99_ms=latency * 1.2,
        throughput_rps=1000.0 / latency,
        memory_mb=memory,
        accuracy_drop=accuracy_drop,
        batch_size=4,
    )


def _rec(result: BenchmarkResult | None = None) -> Recommendation:
    r = result or _result()
    return Recommendation(
        result=r,
        rationale="Best latency on Pareto frontier.",
        pareto_frontier=[r],
        all_results=[r],
    )


def _info() -> ModelInfo:
    return ModelInfo(
        framework="pytorch",
        family="transformer",
        parameters=7_000_000_000,
        trainable_parameters=7_000_000_000,
        estimated_memory_fp32_gb=28.0,
        estimated_memory_fp16_gb=14.0,
        model_path="/models/llama.pt",
        num_layers=32,
        num_attention_heads=32,
        hidden_size=4096,
    )


def _hw() -> HardwareProfile:
    cpu = CPUProfile(name="Intel Xeon", physical_cores=32, logical_cores=64, ram_gb=256.0)
    acc = AcceleratorProfile(  # type: ignore[arg-type]
        kind="cuda", name="NVIDIA A100-SXM4-80GB",
        memory_gb=80.0, bf16=True,
        cuda_compute="8.0",
        memory_bandwidth_gbps=2039.0,
        fp16_tflops=312.0,
        int8_tops=624.0,
    )
    return HardwareProfile(cpu=cpu, accelerator=acc)


# ── build_config ──────────────────────────────────────────────────────────────

def test_build_config_returns_deployment_config() -> None:
    cfg = build_config(_rec(), _info(), _hw())
    assert isinstance(cfg, DeploymentConfig)


def test_build_config_model_fields() -> None:
    cfg = build_config(_rec(), _info(), _hw())
    assert cfg.framework == "pytorch"
    assert cfg.family == "transformer"
    assert cfg.parameters == 7_000_000_000
    assert cfg.model_path == "/models/llama.pt"


def test_build_config_hardware_fields() -> None:
    cfg = build_config(_rec(), _info(), _hw())
    assert cfg.accelerator_kind == "cuda"
    assert cfg.accelerator_name == "NVIDIA A100-SXM4-80GB"
    assert cfg.memory_bandwidth_gbps == 2039.0
    assert cfg.fp16_tflops == 312.0


def test_build_config_recommendation_fields() -> None:
    cfg = build_config(_rec(), _info(), _hw())
    assert cfg.backend == "tensorrt_fp16"
    assert cfg.dtype == "fp16"
    assert cfg.device == "cuda"
    assert cfg.batch_size == 4


def test_build_config_performance_fields() -> None:
    r = _result(latency=3.5, memory=420.0)
    cfg = build_config(_rec(r), _info(), _hw())
    assert cfg.latency_p50_ms == pytest.approx(3.5)
    assert cfg.memory_mb == pytest.approx(420.0)
    assert cfg.throughput_rps == pytest.approx(1000.0 / 3.5)


def test_build_config_accuracy_drop_propagated() -> None:
    r = _result(accuracy_drop=0.023)
    cfg = build_config(_rec(r), _info(), _hw())
    assert cfg.accuracy_drop == pytest.approx(0.023)


def test_build_config_accuracy_drop_none_when_not_measured() -> None:
    cfg = build_config(_rec(), _info(), _hw())
    assert cfg.accuracy_drop is None


def test_build_config_constraints_stored() -> None:
    cfg = build_config(
        _rec(), _info(), _hw(),
        objective="throughput",
        max_latency_ms=10.0,
        max_memory_mb=500.0,
        min_throughput_rps=100.0,
        max_quality_loss=0.01,
    )
    assert cfg.objective == "throughput"
    assert cfg.max_latency_ms == 10.0
    assert cfg.max_memory_mb == 500.0
    assert cfg.min_throughput_rps == 100.0
    assert cfg.max_quality_loss == 0.01


def test_build_config_null_constraints_default_to_none() -> None:
    cfg = build_config(_rec(), _info(), _hw())
    assert cfg.max_latency_ms is None
    assert cfg.max_memory_mb is None
    assert cfg.min_throughput_rps is None
    assert cfg.max_quality_loss is None


def test_build_config_generated_at_is_iso8601() -> None:
    import re
    cfg = build_config(_rec(), _info(), _hw())
    # e.g. "2026-06-16T10:00:00+00:00"
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", cfg.generated_at)


# ── config_to_dict ────────────────────────────────────────────────────────────

def test_config_to_dict_has_required_sections() -> None:
    d = config_to_dict(build_config(_rec(), _info(), _hw()))
    assert set(d.keys()) == {"model", "hardware", "recommendation", "performance", "constraints", "meta", "serving"}


def test_config_to_dict_model_section() -> None:
    d = config_to_dict(build_config(_rec(), _info(), _hw()))
    assert d["model"]["framework"] == "pytorch"
    assert d["model"]["parameters"] == 7_000_000_000


def test_config_to_dict_recommendation_section() -> None:
    d = config_to_dict(build_config(_rec(), _info(), _hw()))
    assert d["recommendation"]["backend"] == "tensorrt_fp16"
    assert d["recommendation"]["batch_size"] == 4


def test_config_to_dict_performance_section() -> None:
    r = _result(latency=3.5)
    d = config_to_dict(build_config(_rec(r), _info(), _hw()))
    assert d["performance"]["latency_p50_ms"] == pytest.approx(3.5)


def test_config_to_dict_meta_has_tool() -> None:
    d = config_to_dict(build_config(_rec(), _info(), _hw()))
    assert d["meta"]["tool"] == "aphex"


# ── write_yaml ────────────────────────────────────────────────────────────────

def test_write_yaml_creates_file(tmp_path: Path) -> None:
    p = tmp_path / "deployment.yaml"
    write_yaml(build_config(_rec(), _info(), _hw()), p)
    assert p.exists()


def test_write_yaml_contains_key_fields(tmp_path: Path) -> None:
    p = tmp_path / "deployment.yaml"
    write_yaml(build_config(_rec(), _info(), _hw()), p)
    content = p.read_text()
    assert "tensorrt_fp16" in content
    assert "pytorch" in content
    assert "A100" in content


def test_write_yaml_has_header_comment(tmp_path: Path) -> None:
    p = tmp_path / "deployment.yaml"
    write_yaml(build_config(_rec(), _info(), _hw()), p)
    content = p.read_text()
    assert content.startswith("# Generated by aphex optimize")


def test_write_yaml_null_values_are_yaml_null(tmp_path: Path) -> None:
    p = tmp_path / "deployment.yaml"
    write_yaml(build_config(_rec(), _info(), _hw()), p)
    content = p.read_text()
    assert "null" in content  # None constraints render as null


def test_write_yaml_can_be_round_tripped(tmp_path: Path) -> None:
    """Verify the output is valid YAML by parsing it."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        pytest.skip("pyyaml not installed — skipping round-trip check")

    p = tmp_path / "deployment.yaml"
    cfg = build_config(_rec(), _info(), _hw())
    write_yaml(cfg, p)
    data = yaml.safe_load(p.read_text())
    assert data["recommendation"]["backend"] == cfg.backend
    assert data["model"]["parameters"] == cfg.parameters


def test_write_yaml_float_formatting(tmp_path: Path) -> None:
    p = tmp_path / "deployment.yaml"
    r = _result(latency=3.1415926)
    write_yaml(build_config(_rec(r), _info(), _hw()), p)
    content = p.read_text()
    # Should not emit scientific notation for typical latency values
    assert "e+" not in content
    assert "e-" not in content


# ── _scalar ──────────────────────────────────────────────────────────────────

def test_scalar_none_is_null() -> None:
    assert _scalar(None) == "null"


def test_scalar_bool_true() -> None:
    assert _scalar(True) == "true"


def test_scalar_bool_false() -> None:
    assert _scalar(False) == "false"


def test_scalar_int() -> None:
    assert _scalar(42) == "42"


def test_scalar_float_no_scientific_notation() -> None:
    result = _scalar(3.14)
    assert "e" not in result.lower()


def test_scalar_string_is_quoted() -> None:
    result = _scalar("pytorch_fp32")
    assert result.startswith('"') and result.endswith('"')


def test_scalar_string_with_special_chars_quoted() -> None:
    result = _scalar("NVIDIA A100: 80GB")
    assert '"' in result
