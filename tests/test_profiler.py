"""Tests for hardware profiler."""

from infermap.profiler import profile_hardware, HardwareProfile


def test_profile_returns_hardware_profile() -> None:
    hw = profile_hardware()
    assert isinstance(hw, HardwareProfile)


def test_cpu_profile_populated() -> None:
    hw = profile_hardware()
    assert hw.cpu.ram_gb > 0
    assert hw.cpu.physical_cores >= 1
    assert hw.cpu.logical_cores >= hw.cpu.physical_cores
    assert hw.cpu.name != ""


def test_accelerator_kind_valid() -> None:
    hw = profile_hardware()
    assert hw.accelerator.kind in ("cuda", "mps", "none")


def test_available_memory_positive() -> None:
    hw = profile_hardware()
    assert hw.available_memory_gb >= 0
