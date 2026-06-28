"""Candidate generation — produces deployment strategies based on model type and hardware."""

from __future__ import annotations

from dataclasses import dataclass

from aphex.inspector import ModelInfo
from aphex.profiler import HardwareProfile

Backend = str  # e.g. "pytorch_fp32", "sklearn_predict", "treelite_cpu", ...


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
            backend="tensorrt_fp32",
            dtype="fp32",
            description="TensorRT FP32 (CUDA)",
            requires_export=True,
            device="cuda",
        ),
        DeploymentCandidate(
            backend="tensorrt_fp16",
            dtype="fp16",
            description="TensorRT FP16 (CUDA)",
            requires_export=True,
            device="cuda",
        ),
        DeploymentCandidate(
            backend="tensorrt_int8",
            dtype="int8",
            description="TensorRT INT8 (CUDA, requires --calibration-data)",
            requires_export=True,
            device="cuda",
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

    candidates += _prune_candidates("cuda")
    candidates += _data_parallel_candidates(hardware)
    return candidates


def _data_parallel_candidates(hardware: HardwareProfile) -> list[DeploymentCandidate]:
    """Single-process nn.DataParallel candidates, one per supported dtype × world size.

    Only emitted when the host actually has >=2 CUDA devices. DP shards the batch
    dimension across GPUs, so it's a throughput win at large batch and a latency
    no-op (or slight loss) at batch=1 — the runner enforces ``batch_size >= N``.
    """
    n = hardware.accelerator.device_count
    if hardware.accelerator.kind != "cuda" or n < 2:
        return []
    world_sizes = [w for w in (2, 4, 8) if w <= n]
    dtypes: list[tuple[str, str]] = [("fp32", "FP32"), ("fp16", "FP16")]
    if hardware.accelerator.bf16:
        dtypes.append(("bf16", "BF16"))
    out: list[DeploymentCandidate] = []
    for world in world_sizes:
        for dtype, dtype_label in dtypes:
            out.append(
                DeploymentCandidate(
                    backend=f"pytorch_dp{world}_{dtype}",
                    dtype=dtype,
                    description=(
                        f"PyTorch {dtype_label} on {world}× GPU via nn.DataParallel "
                        f"(throughput-oriented; requires batch >= {world})"
                    ),
                    requires_export=False,
                    device="cuda",
                )
            )
    return out


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
        DeploymentCandidate(
            backend="openvino_fp32",
            dtype="fp32",
            description="OpenVINO FP32 (CPU)",
            requires_export=True,
            device="cpu",
        ),
    ]
    candidates += _prune_candidates("cpu")
    return candidates


def _prune_candidates(device: str) -> list[DeploymentCandidate]:
    """Magnitude-pruned PyTorch candidates available on any device.

    2:4 structured pruning is only meaningful as a latency win on CUDA Ampere+
    (sm_80+ sparse Tensor Cores). On other devices it still produces a valid
    50%-sparse model — useful as a storage / accuracy reference — but with no
    speedup expected.
    """
    return [
        DeploymentCandidate(
            backend="pytorch_prune_unstructured_30",
            dtype="fp32",
            description="PyTorch FP32 + 30% magnitude prune",
            requires_export=False,
            device=device,
        ),
        DeploymentCandidate(
            backend="pytorch_prune_unstructured_50",
            dtype="fp32",
            description="PyTorch FP32 + 50% magnitude prune",
            requires_export=False,
            device=device,
        ),
        DeploymentCandidate(
            backend="pytorch_prune_unstructured_70",
            dtype="fp32",
            description="PyTorch FP32 + 70% magnitude prune",
            requires_export=False,
            device=device,
        ),
        DeploymentCandidate(
            backend="pytorch_prune_2_4",
            dtype="fp32",
            description="PyTorch FP32 + 2:4 structured prune",
            requires_export=False,
            device=device,
        ),
    ]


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
        DeploymentCandidate(
            backend="openvino_fp32",
            dtype="fp32",
            description="OpenVINO FP32 (CPU)",
            requires_export=True,
            device="cpu",
        ),
        DeploymentCandidate(
            backend="openvino_int8",
            dtype="int8",
            description="OpenVINO INT8 (CPU, requires nncf)",
            requires_export=True,
            device="cpu",
        ),
        *_prune_candidates("cpu"),
    ]
