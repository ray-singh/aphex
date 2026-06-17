"""Tests for infermap.cloud.instances — cloud instance hardware profiles."""
from __future__ import annotations

import pytest

from infermap.cloud.instances import get_instance_profile, list_targets
from infermap.profiler import HardwareProfile


# ── get_instance_profile: known instances ─────────────────────────────────────


def test_returns_hardware_profile() -> None:
    hw = get_instance_profile("ml.g4dn.xlarge")
    assert isinstance(hw, HardwareProfile)


def test_g4dn_xlarge_is_t4() -> None:
    hw = get_instance_profile("ml.g4dn.xlarge")
    assert "T4" in hw.accelerator.name


def test_g4dn_xlarge_vram() -> None:
    hw = get_instance_profile("ml.g4dn.xlarge")
    assert hw.accelerator.memory_gb == 16.0


def test_g4dn_no_bf16() -> None:
    hw = get_instance_profile("ml.g4dn.xlarge")
    assert hw.accelerator.bf16 is False


def test_a2_highgpu_is_a100() -> None:
    hw = get_instance_profile("a2-highgpu-1g")
    assert "A100" in hw.accelerator.name


def test_a2_highgpu_bf16() -> None:
    hw = get_instance_profile("a2-highgpu-1g")
    assert hw.accelerator.bf16 is True


def test_a2_highgpu_vram() -> None:
    hw = get_instance_profile("a2-highgpu-1g")
    assert hw.accelerator.memory_gb == 40.0


def test_a3_highgpu_is_h100() -> None:
    hw = get_instance_profile("a3-highgpu-1g")
    assert "H100" in hw.accelerator.name


def test_p3_2xlarge_is_v100() -> None:
    hw = get_instance_profile("ml.p3.2xlarge")
    assert "V100" in hw.accelerator.name


def test_g5_xlarge_is_a10g() -> None:
    hw = get_instance_profile("ml.g5.xlarge")
    assert "A10G" in hw.accelerator.name


def test_g6_xlarge_is_l4() -> None:
    hw = get_instance_profile("ml.g6.xlarge")
    assert "L4" in hw.accelerator.name


def test_azure_nc6s_v3_is_v100() -> None:
    hw = get_instance_profile("Standard_NC6s_v3")
    assert "V100" in hw.accelerator.name


def test_azure_nc_a100_v4_is_a100() -> None:
    hw = get_instance_profile("Standard_NC24ads_A100_v4")
    assert "A100" in hw.accelerator.name


# ── case-insensitive ──────────────────────────────────────────────────────────


def test_case_insensitive_upper() -> None:
    hw = get_instance_profile("ML.G4DN.XLARGE")
    assert "T4" in hw.accelerator.name


def test_case_insensitive_mixed() -> None:
    hw = get_instance_profile("A2-HighGPU-1g")
    assert "A100" in hw.accelerator.name


# ── short GPU aliases ─────────────────────────────────────────────────────────


def test_alias_t4() -> None:
    hw = get_instance_profile("t4")
    assert "T4" in hw.accelerator.name


def test_alias_a100() -> None:
    hw = get_instance_profile("a100")
    assert "A100" in hw.accelerator.name


def test_alias_a100_80gb() -> None:
    hw = get_instance_profile("a100-80gb")
    assert hw.accelerator.memory_gb == 80.0


def test_alias_h100() -> None:
    hw = get_instance_profile("h100")
    assert "H100" in hw.accelerator.name


def test_alias_v100() -> None:
    hw = get_instance_profile("v100")
    assert "V100" in hw.accelerator.name


def test_alias_l4() -> None:
    hw = get_instance_profile("l4")
    assert "L4" in hw.accelerator.name


def test_alias_a10g() -> None:
    hw = get_instance_profile("a10g")
    assert "A10G" in hw.accelerator.name


# ── gpu specs populated ───────────────────────────────────────────────────────


def test_t4_has_bandwidth() -> None:
    hw = get_instance_profile("t4")
    assert hw.accelerator.memory_bandwidth_gbps is not None
    assert hw.accelerator.memory_bandwidth_gbps > 0


def test_a100_has_fp16_tflops() -> None:
    hw = get_instance_profile("a100")
    assert hw.accelerator.fp16_tflops is not None
    assert hw.accelerator.fp16_tflops > 0


def test_t4_has_int8_tops() -> None:
    hw = get_instance_profile("t4")
    assert hw.accelerator.int8_tops is not None
    assert hw.accelerator.int8_tops > 0


def test_accelerator_kind_is_cuda() -> None:
    hw = get_instance_profile("ml.g4dn.xlarge")
    assert hw.accelerator.kind == "cuda"


# ── unknown target ────────────────────────────────────────────────────────────


def test_unknown_target_raises() -> None:
    with pytest.raises(ValueError, match="Unknown target"):
        get_instance_profile("ml.quantum.1xlarge")


def test_unknown_target_mentions_targets_command() -> None:
    with pytest.raises(ValueError, match="aphex targets"):
        get_instance_profile("totally-fake-instance")


# ── list_targets ──────────────────────────────────────────────────────────────


def test_list_targets_has_all_providers() -> None:
    groups = list_targets()
    assert "aws" in groups
    assert "gcp" in groups
    assert "azure" in groups
    assert "aliases" in groups


def test_list_targets_aws_non_empty() -> None:
    assert len(list_targets()["aws"]) > 0


def test_list_targets_gcp_non_empty() -> None:
    assert len(list_targets()["gcp"]) > 0


def test_list_targets_aliases_includes_t4() -> None:
    assert "t4" in list_targets()["aliases"]


def test_list_targets_aliases_includes_h100() -> None:
    assert "h100" in list_targets()["aliases"]
