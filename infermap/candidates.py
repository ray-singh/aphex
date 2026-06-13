"""Candidate generation — produces deployment strategies based on model type and hardware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from infermap.inspector import ModelInfo
from infermap.profiler import HardwareProfile

Backend = Literal[
    "pytorch_fp32",
    "pytorch_fp16",
    "pytorch_bf16",
    "torch_compile_fp32",
    "torch_compile_fp16",
    "onnx_cpu",
    "onnx_cuda",
    "onnx_coreml",
    "pytorch_int8_dynamic",
    "onnx_int8_cpu",
]


@dataclass
class DeploymentCandidate:
    backend: Backend
    dtype: str
    description: str
    requires_export: bool  # True if model must be exported to ONNX first
    device: str  # "cpu", "cuda", "mps"

    @property
    def id(self) -> str:
        return self.backend


def generate_candidates(
    model_info: ModelInfo, hardware: HardwareProfile
) -> list[DeploymentCandidate]:
    kind = hardware.accelerator.kind

    if kind == "cuda":
        return _cuda_candidates(hardware)
    if kind == "mps":
        return _mps_candidates(hardware)
    return _cpu_candidates()


def _cuda_candidates(hardware: HardwareProfile) -> list[DeploymentCandidate]:
    candidates: list[DeploymentCandidate] = [
        DeploymentCandidate(
            backend="pytorch_fp32",
            dtype="fp32",
            description="PyTorch FP32 baseline",
            requires_export=False,
            device="cuda",
        ),
        DeploymentCandidate(
            backend="pytorch_fp16",
            dtype="fp16",
            description="PyTorch FP16 (half precision)",
            requires_export=False,
            device="cuda",
        ),
        DeploymentCandidate(
            backend="torch_compile_fp32",
            dtype="fp32",
            description="torch.compile FP32",
            requires_export=False,
            device="cuda",
        ),
        DeploymentCandidate(
            backend="onnx_cpu",
            dtype="fp32",
            description="ONNX Runtime CPU",
            requires_export=True,
            device="cpu",
        ),
        DeploymentCandidate(
            backend="onnx_cuda",
            dtype="fp32",
            description="ONNX Runtime CUDA",
            requires_export=True,
            device="cuda",
        ),
    ]

    # BF16 requires sm_80+ (Ampere). T4 is sm_75 — skip BF16 there.
    if hardware.accelerator.bf16:
        candidates.append(
            DeploymentCandidate(
                backend="pytorch_bf16",
                dtype="bf16",
                description="PyTorch BF16 (bfloat16)",
                requires_export=False,
                device="cuda",
            )
        )

    candidates += [
        DeploymentCandidate(
            backend="pytorch_int8_dynamic",
            dtype="int8",
            description="PyTorch INT8 dynamic (CPU)",
            requires_export=False,
            device="cpu",
        ),
        DeploymentCandidate(
            backend="onnx_int8_cpu",
            dtype="int8",
            description="ONNX Runtime INT8 (CPU)",
            requires_export=True,
            device="cpu",
        ),
    ]

    return candidates


def _mps_candidates(hardware: HardwareProfile) -> list[DeploymentCandidate]:
    candidates: list[DeploymentCandidate] = [
        DeploymentCandidate(
            backend="pytorch_fp32",
            dtype="fp32",
            description="PyTorch FP32 CPU",
            requires_export=False,
            device="cpu",
        ),
        DeploymentCandidate(
            backend="pytorch_fp32",
            dtype="fp32",
            description="PyTorch FP32 MPS",
            requires_export=False,
            device="mps",
        ),
        DeploymentCandidate(
            backend="pytorch_fp16",
            dtype="fp16",
            description="PyTorch FP16 MPS",
            requires_export=False,
            device="mps",
        ),
        DeploymentCandidate(
            backend="onnx_coreml",
            dtype="fp32",
            description="ONNX Runtime + CoreML (Apple Silicon)",
            requires_export=True,
            device="cpu",
        ),
        DeploymentCandidate(
            backend="torch_compile_fp32",
            dtype="fp32",
            description="torch.compile (CPU mode)",
            requires_export=False,
            device="cpu",
        ),
        DeploymentCandidate(
            backend="pytorch_int8_dynamic",
            dtype="int8",
            description="PyTorch INT8 dynamic (CPU)",
            requires_export=False,
            device="cpu",
        ),
        DeploymentCandidate(
            backend="onnx_int8_cpu",
            dtype="int8",
            description="ONNX Runtime INT8 (CPU)",
            requires_export=True,
            device="cpu",
        ),
    ]
    return candidates


def _cpu_candidates() -> list[DeploymentCandidate]:
    return [
        DeploymentCandidate(
            backend="pytorch_fp32",
            dtype="fp32",
            description="PyTorch FP32 CPU baseline",
            requires_export=False,
            device="cpu",
        ),
        DeploymentCandidate(
            backend="torch_compile_fp32",
            dtype="fp32",
            description="torch.compile FP32 CPU",
            requires_export=False,
            device="cpu",
        ),
        DeploymentCandidate(
            backend="onnx_cpu",
            dtype="fp32",
            description="ONNX Runtime CPU",
            requires_export=True,
            device="cpu",
        ),
        DeploymentCandidate(
            backend="pytorch_int8_dynamic",
            dtype="int8",
            description="PyTorch INT8 dynamic (CPU)",
            requires_export=False,
            device="cpu",
        ),
        DeploymentCandidate(
            backend="onnx_int8_cpu",
            dtype="int8",
            description="ONNX Runtime INT8 (CPU)",
            requires_export=True,
            device="cpu",
        ),
    ]
