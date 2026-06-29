"""Deployment config generation — serializes the optimize recommendation to YAML."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aphex.inspector import ModelInfo
from aphex.profiler import HardwareProfile
from aphex.recommender import Recommendation
from aphex.system_recommender import SystemConfig


@dataclass
class DeploymentConfig:
    # model
    model_path: str | None
    framework: str
    family: str
    parameters: int
    # hardware
    accelerator_kind: str
    accelerator_name: str
    accelerator_memory_gb: float
    memory_bandwidth_gbps: float | None
    fp16_tflops: float | None
    # recommendation
    backend: str
    dtype: str
    device: str
    batch_size: int
    description: str
    # performance
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_rps: float
    memory_mb: float
    accuracy_drop: float | None
    # search parameters
    objective: str
    max_latency_ms: float | None
    max_memory_mb: float | None
    min_throughput_rps: float | None
    max_quality_loss: float | None
    # meta
    generated_at: str
    # Input spec string (parseable by :meth:`InputSpec.parse`) — stored so
    # ``aphex convert --from-config`` / ``aphex check --from-config`` are
    # self-contained. Single-input round-trips as ``"3,224,224"``; multi-input
    # as ``"input_ids:128:long;attention_mask:128:long"``.
    input_shape: str | None = None
    # system-level serving config (optional)
    system: SystemConfig | None = None
    # eval (optional — populated when --eval + --infer-fn are provided)
    eval_metric: str | None = None
    eval_score: float | None = None


def build_config(
    rec: Recommendation,
    info: ModelInfo,
    hw: HardwareProfile,
    objective: str = "latency",
    max_latency_ms: float | None = None,
    max_memory_mb: float | None = None,
    min_throughput_rps: float | None = None,
    max_quality_loss: float | None = None,
    system: SystemConfig | None = None,
    eval_metric: str | None = None,
    eval_score: float | None = None,
    input_shape: str | None = None,
) -> DeploymentConfig:
    r = rec.result
    return DeploymentConfig(
        model_path=info.model_path,
        framework=info.framework,
        family=info.family,
        parameters=info.parameters,
        accelerator_kind=hw.accelerator.kind,
        accelerator_name=hw.accelerator.name,
        accelerator_memory_gb=hw.accelerator.memory_gb,
        memory_bandwidth_gbps=hw.accelerator.memory_bandwidth_gbps,
        fp16_tflops=hw.accelerator.fp16_tflops,
        backend=r.candidate.backend,
        dtype=r.candidate.dtype,
        device=r.candidate.device,
        batch_size=r.batch_size,
        description=r.candidate.description,
        latency_p50_ms=r.latency_p50_ms,
        latency_p95_ms=r.latency_p95_ms,
        latency_p99_ms=r.latency_p99_ms,
        throughput_rps=r.throughput_rps,
        memory_mb=r.memory_mb,
        accuracy_drop=r.accuracy_drop,
        objective=objective,
        max_latency_ms=max_latency_ms,
        max_memory_mb=max_memory_mb,
        min_throughput_rps=min_throughput_rps,
        max_quality_loss=max_quality_loss,
        generated_at=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        input_shape=input_shape,
        system=system,
        eval_metric=eval_metric,
        eval_score=eval_score,
    )


def config_to_dict(config: DeploymentConfig) -> dict[str, Any]:
    return {
        "model": {
            "path": config.model_path,
            "framework": config.framework,
            "family": config.family,
            "parameters": config.parameters,
            "input_shape": config.input_shape,
        },
        "hardware": {
            "accelerator": config.accelerator_kind,
            "device": config.accelerator_name,
            "memory_gb": config.accelerator_memory_gb,
            "memory_bandwidth_gbps": config.memory_bandwidth_gbps,
            "fp16_tflops": config.fp16_tflops,
        },
        "recommendation": {
            "backend": config.backend,
            "dtype": config.dtype,
            "device": config.device,
            "batch_size": config.batch_size,
            "description": config.description,
        },
        "performance": {
            "latency_p50_ms": config.latency_p50_ms,
            "latency_p95_ms": config.latency_p95_ms,
            "latency_p99_ms": config.latency_p99_ms,
            "throughput_rps": config.throughput_rps,
            "memory_mb": config.memory_mb,
            "accuracy_drop": config.accuracy_drop,
            "eval_metric": config.eval_metric,
            "eval_score": config.eval_score,
        },
        "constraints": {
            "objective": config.objective,
            "max_latency_ms": config.max_latency_ms,
            "max_memory_mb": config.max_memory_mb,
            "min_throughput_rps": config.min_throughput_rps,
            "max_quality_loss": config.max_quality_loss,
        },
        "meta": {
            "generated_at": config.generated_at,
            "tool": "aphex",
        },
        "serving": _system_config_dict(config.system),
    }


def write_yaml(config: DeploymentConfig, path: Path) -> None:
    """Write deployment config as YAML. No external dependencies."""
    d = config_to_dict(config)
    lines = [
        "# Generated by aphex optimize",
        f"# {config.generated_at}",
        "",
    ]
    lines.append(_section_to_yaml(d))
    path.write_text("\n".join(lines) + "\n")


# ── YAML serialiser ───────────────────────────────────────────────────────────

def _section_to_yaml(d: dict[str, Any], indent: int = 0) -> str:
    parts: list[str] = []
    pad = "  " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            parts.append(f"{pad}{key}:")
            parts.append(_section_to_yaml(value, indent + 1))
        else:
            parts.append(f"{pad}{key}: {_scalar(value)}")
    return "\n".join(parts)


def _system_config_dict(system: SystemConfig | None) -> dict[str, Any]:
    if system is None:
        return {k: None for k in (
            "max_safe_batch_size", "recommended_batch_size", "dynamic_batching",
            "num_workers", "tensor_parallel_size", "kv_cache_fraction",
            "continuous_batching", "prefill_chunk_size", "kv_cache_dtype",
            "enable_result_cache",
        )}
    return {
        "max_safe_batch_size": system.max_safe_batch_size,
        "recommended_batch_size": system.recommended_batch_size,
        "dynamic_batching": system.dynamic_batching,
        "num_workers": system.num_workers,
        "tensor_parallel_size": system.tensor_parallel_size,
        "kv_cache_fraction": system.kv_cache_fraction,
        "continuous_batching": system.continuous_batching,
        "prefill_chunk_size": system.prefill_chunk_size,
        "kv_cache_dtype": system.kv_cache_dtype,
        "enable_result_cache": system.enable_result_cache,
    }


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Avoid scientific notation for typical deployment numbers
        return f"{value:.4g}"
    # Quote all strings — safe and unambiguous
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
