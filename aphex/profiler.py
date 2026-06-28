"""Hardware profiler — detects CPU, accelerator type, memory, and capabilities."""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger("aphex.profiler")


@dataclass
class CPUProfile:
    name: str
    physical_cores: int
    logical_cores: int
    ram_gb: float


@dataclass
class AcceleratorProfile:
    kind: Literal["cuda", "mps", "none"]
    name: str
    memory_gb: float
    bf16: bool
    # CUDA-only fields
    cuda_compute: str = ""
    device_count: int = 1
    # Hardware capability fingerprint (V3) — None when unknown
    memory_bandwidth_gbps: float | None = None
    fp16_tflops: float | None = None
    int8_tops: float | None = None
    # MPS-only: memory is shared with CPU RAM


@dataclass
class HardwareProfile:
    cpu: CPUProfile
    accelerator: AcceleratorProfile
    platform: str = field(default_factory=platform.platform)

    @property
    def has_gpu(self) -> bool:
        return self.accelerator.kind in ("cuda", "mps")

    @property
    def available_memory_gb(self) -> float:
        """Best estimate of memory available for model loading."""
        if self.accelerator.kind == "cuda":
            return self.accelerator.memory_gb
        # MPS and CPU-only: use RAM, reserving ~4 GB for system
        return max(0.0, self.cpu.ram_gb - 4.0)


# GPU specs ordered longest-match-first so "a100" doesn't swallow "a10".
# Values from NVIDIA product pages (fp16 Tensor Core peak, HBM bandwidth).
_GPU_SPECS: list[tuple[str, float, float, float]] = [
    # (name_substr, memory_bandwidth_gbps, fp16_tflops, int8_tops)
    ("h200",    4800.0, 1979.0, 3958.0),
    ("h100",    3350.0,  989.0, 1979.0),
    ("b200",    8000.0, 4500.0, 9000.0),
    ("a100",    2039.0,  312.0,  624.0),
    ("a30",      933.0,  165.0,  330.0),
    ("a10g",     600.0,  125.0,  250.0),
    ("a10",      600.0,  125.0,  250.0),
    ("l40s",     864.0,  362.0,  724.0),
    ("l40",      864.0,  181.0,  362.0),
    ("l4",       300.0,  121.0,  242.0),
    ("v100",     900.0,  112.0,    0.0),
    ("t4",       320.0,   65.0,  130.0),
    ("p100",     720.0,   18.7,    0.0),
    ("rtx 4090", 1008.0, 330.0,  660.0),
    ("rtx 4080",  717.0, 245.0,  490.0),
    ("rtx 3090",  936.0, 142.0,  284.0),
    ("rtx 3080",  760.0, 119.0,  238.0),
    ("rtx 3070",  448.0,  82.0,  164.0),
    ("4090",     1008.0, 330.0,  660.0),
    ("4080",      717.0, 245.0,  490.0),
    ("3090",      936.0, 142.0,  284.0),
    ("3080",      760.0, 119.0,  238.0),
    ("3070",      448.0,  82.0,  164.0),
]

# Apple Silicon — longer variant names must precede shorter ones (m4 max before m4).
# Values from Apple product pages (GPU bandwidth, Neural Engine excluded from fp16_tflops).
_APPLE_SPECS: list[tuple[str, float, float]] = [
    # (name_substr, memory_bandwidth_gbps, fp16_tflops)
    ("m4 max",    546.0, 18.4),
    ("m4 pro",    273.0,  9.2),
    ("m4",        120.0,  4.6),
    ("m3 max",    300.0, 14.2),
    ("m3 pro",    150.0,  7.4),
    ("m3",        100.0,  3.6),
    ("m2 ultra",  800.0, 27.2),
    ("m2 max",    400.0, 13.6),
    ("m2 pro",    200.0,  6.8),
    ("m2",        100.0,  3.6),
    ("m1 ultra",  800.0, 20.8),
    ("m1 max",    400.0, 10.4),
    ("m1 pro",    200.0,  5.2),
    ("m1",         68.3,  2.6),
]


def _lookup_gpu_specs(name: str) -> tuple[float | None, float | None, float | None]:
    lower = name.lower()
    for substr, bw, fp16, int8 in _GPU_SPECS:
        if substr in lower:
            return bw, fp16, int8
    return None, None, None


def _lookup_apple_specs(name: str) -> tuple[float | None, float | None]:
    lower = name.lower()
    for substr, bw, fp16 in _APPLE_SPECS:
        if substr in lower:
            return bw, fp16
    return None, None


def profile_hardware() -> HardwareProfile:
    cpu = _profile_cpu()
    accelerator = _detect_accelerator()
    return HardwareProfile(cpu=cpu, accelerator=accelerator)


def _profile_cpu() -> CPUProfile:
    import psutil

    name = _cpu_name()
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or 1
    ram_gb = psutil.virtual_memory().total / 1e9
    return CPUProfile(name=name, physical_cores=physical, logical_cores=logical, ram_gb=ram_gb)


def _cpu_name() -> str:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            return out
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception as exc:
        logger.debug("CPU name detection failed: %s", exc)
    return platform.processor() or "Unknown CPU"


def _detect_accelerator() -> AcceleratorProfile:
    try:
        import torch

        if torch.cuda.is_available():
            return _profile_cuda(torch)
        if torch.backends.mps.is_available():
            return _profile_mps()
    except ImportError:
        pass
    return AcceleratorProfile(kind="none", name="CPU-only", memory_gb=0.0, bf16=False)


def _profile_cuda(torch: object) -> AcceleratorProfile:
    import torch as t

    idx = t.cuda.current_device()
    name = t.cuda.get_device_name(idx)
    props = t.cuda.get_device_properties(idx)
    memory_gb = props.total_memory / 1e9
    bf16 = props.major >= 8  # Ampere (sm_80) and above support BF16
    compute = f"{props.major}.{props.minor}"
    bw, fp16, int8 = _lookup_gpu_specs(name)
    return AcceleratorProfile(
        kind="cuda",
        name=name,
        memory_gb=memory_gb,
        bf16=bf16,
        cuda_compute=compute,
        device_count=t.cuda.device_count(),
        memory_bandwidth_gbps=bw,
        fp16_tflops=fp16,
        int8_tops=int8,
    )


def _profile_mps() -> AcceleratorProfile:
    import psutil

    chip = _apple_chip_name()
    ram_gb = psutil.virtual_memory().total / 1e9
    # M1 Pro and earlier do not support BF16; M2+ does
    bf16 = _apple_supports_bf16(chip)
    bw, fp16 = _lookup_apple_specs(chip)
    return AcceleratorProfile(
        kind="mps",
        name=chip,
        memory_gb=ram_gb,  # unified memory
        bf16=bf16,
        memory_bandwidth_gbps=bw,
        fp16_tflops=fp16,
    )


def _apple_chip_name() -> str:
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True
        )
        for line in out.splitlines():
            if "Chip" in line or "Processor Name" in line:
                return line.split(":", 1)[1].strip()
    except Exception as exc:
        logger.debug("Apple chip name detection failed: %s", exc)
    return "Apple Silicon"


def _apple_supports_bf16(chip_name: str) -> bool:
    name_lower = chip_name.lower()
    return any(gen in name_lower for gen in ("m2", "m3", "m4", "m5"))


def collect_environment(hw: HardwareProfile) -> dict[str, Any]:
    """Return a snapshot of the runtime environment for benchmark reproducibility."""
    acc = hw.accelerator
    acc_info: dict[str, Any] = {
        "kind": acc.kind,
        "name": acc.name,
        "memory_gb": acc.memory_gb,
    }
    if acc.cuda_compute:
        acc_info["cuda_compute"] = acc.cuda_compute

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "os": platform.system(),
        "os_version": platform.release(),
        "machine": platform.machine(),
        "cpu": {
            "name": hw.cpu.name,
            "physical_cores": hw.cpu.physical_cores,
            "logical_cores": hw.cpu.logical_cores,
            "ram_gb": hw.cpu.ram_gb,
        },
        "accelerator": acc_info,
        "libraries": _collect_lib_versions(),
    }


def _collect_lib_versions() -> dict[str, str]:
    libs: dict[str, str] = {}
    _try_lib(libs, "torch")
    _try_lib(libs, "onnxruntime")
    _try_lib(libs, "numpy")
    _try_lib(libs, "sklearn")
    try:
        import torch

        if torch.cuda.is_available() and torch.version.cuda:
            libs["cuda_toolkit"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        logger.debug("cuda_toolkit version detection failed", exc_info=True)
    return libs


def _try_lib(out: dict[str, str], name: str) -> None:
    try:
        mod = __import__(name)
        out[name] = getattr(mod, "__version__", "unknown")
    except ImportError:
        pass
