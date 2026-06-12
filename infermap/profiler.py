"""Hardware profiler — detects CPU, accelerator type, memory, and capabilities."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from typing import Literal


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
    except Exception:
        pass
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
    return AcceleratorProfile(
        kind="cuda",
        name=name,
        memory_gb=memory_gb,
        bf16=bf16,
        cuda_compute=compute,
    )


def _profile_mps() -> AcceleratorProfile:
    import psutil

    chip = _apple_chip_name()
    ram_gb = psutil.virtual_memory().total / 1e9
    # M1 Pro and earlier do not support BF16; M2+ does
    bf16 = _apple_supports_bf16(chip)
    return AcceleratorProfile(
        kind="mps",
        name=chip,
        memory_gb=ram_gb,  # unified memory
        bf16=bf16,
    )


def _apple_chip_name() -> str:
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True
        )
        for line in out.splitlines():
            if "Chip" in line or "Processor Name" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "Apple Silicon"


def _apple_supports_bf16(chip_name: str) -> bool:
    name_lower = chip_name.lower()
    # M2 and later support BF16
    for gen in ("m2", "m3", "m4", "m5"):
        if gen in name_lower:
            return True
    return False
