"""Known cloud instance hardware profiles.

Maps canonical instance type strings and short GPU aliases to HardwareProfile
objects, so aphex optimize can reason about target hardware without needing to
run on that machine.

Usage
-----
    from aphex.cloud.instances import get_instance_profile, list_targets

    hw = get_instance_profile("ml.g4dn.xlarge")
    hw = get_instance_profile("a2-highgpu-1g")
    hw = get_instance_profile("a100-80gb")   # short alias
"""

from __future__ import annotations

from aphex.profiler import AcceleratorProfile, CPUProfile, HardwareProfile, _lookup_gpu_specs

# ── raw instance data ─────────────────────────────────────────────────────────
#
# Each entry: (gpu_name, vram_gb, cpu_ram_gb, vcpus, bf16)
# bf16 = True for Ampere (A100, A10G, A30) and later; False for Volta/Turing.
# GPU bandwidth/tflops/tops come from the shared _lookup_gpu_specs table in profiler.py.

_INSTANCES: dict[str, tuple[str, float, float, int, bool]] = {

    # ── AWS EC2 / SageMaker ───────────────────────────────────────────────────
    # g4dn — NVIDIA T4 (Turing, no BF16)
    "ml.g4dn.xlarge":    ("NVIDIA Tesla T4",  16.0,  16.0,  4, False),
    "ml.g4dn.2xlarge":   ("NVIDIA Tesla T4",  16.0,  32.0,  8, False),
    "ml.g4dn.4xlarge":   ("NVIDIA Tesla T4",  16.0,  64.0, 16, False),
    "ml.g4dn.8xlarge":   ("NVIDIA Tesla T4",  16.0, 128.0, 32, False),
    "ml.g4dn.12xlarge":  ("NVIDIA Tesla T4",  64.0, 192.0, 48, False),  # 4× T4
    "g4dn.xlarge":       ("NVIDIA Tesla T4",  16.0,  16.0,  4, False),
    "g4dn.2xlarge":      ("NVIDIA Tesla T4",  16.0,  32.0,  8, False),
    "g4dn.4xlarge":      ("NVIDIA Tesla T4",  16.0,  64.0, 16, False),
    "g4dn.12xlarge":     ("NVIDIA Tesla T4",  64.0, 192.0, 48, False),

    # g5 — NVIDIA A10G (Ampere, BF16)
    "ml.g5.xlarge":      ("NVIDIA A10G",      24.0,  16.0,  4, True),
    "ml.g5.2xlarge":     ("NVIDIA A10G",      24.0,  32.0,  8, True),
    "ml.g5.4xlarge":     ("NVIDIA A10G",      24.0,  64.0, 16, True),
    "ml.g5.8xlarge":     ("NVIDIA A10G",      24.0, 128.0, 32, True),
    "ml.g5.12xlarge":    ("NVIDIA A10G",      96.0, 192.0, 48, True),  # 4× A10G
    "g5.xlarge":         ("NVIDIA A10G",      24.0,  16.0,  4, True),
    "g5.2xlarge":        ("NVIDIA A10G",      24.0,  32.0,  8, True),
    "g5.4xlarge":        ("NVIDIA A10G",      24.0,  64.0, 16, True),
    "g5.12xlarge":       ("NVIDIA A10G",      96.0, 192.0, 48, True),

    # g6 — NVIDIA L4 (Ada Lovelace, BF16)
    "ml.g6.xlarge":      ("NVIDIA L4",        24.0,  16.0,  4, True),
    "ml.g6.2xlarge":     ("NVIDIA L4",        24.0,  32.0,  8, True),
    "ml.g6.4xlarge":     ("NVIDIA L4",        24.0,  64.0, 16, True),
    "ml.g6.8xlarge":     ("NVIDIA L4",        24.0, 128.0, 32, True),
    "g6.xlarge":         ("NVIDIA L4",        24.0,  16.0,  4, True),
    "g6.2xlarge":        ("NVIDIA L4",        24.0,  32.0,  8, True),

    # p3 — NVIDIA V100 (Volta, no BF16)
    "ml.p3.2xlarge":     ("NVIDIA Tesla V100", 16.0,  61.0,  8, False),
    "ml.p3.8xlarge":     ("NVIDIA Tesla V100", 64.0, 244.0, 32, False),  # 4× V100
    "ml.p3.16xlarge":    ("NVIDIA Tesla V100", 128.0, 488.0, 64, False), # 8× V100
    "p3.2xlarge":        ("NVIDIA Tesla V100", 16.0,  61.0,  8, False),
    "p3.8xlarge":        ("NVIDIA Tesla V100", 64.0, 244.0, 32, False),

    # p4d — NVIDIA A100 40 GB (Ampere, BF16)
    "ml.p4d.24xlarge":   ("NVIDIA A100",      320.0, 1152.0, 96, True), # 8× A100 40GB
    "p4d.24xlarge":      ("NVIDIA A100",      320.0, 1152.0, 96, True),

    # p4de — NVIDIA A100 80 GB
    "ml.p4de.24xlarge":  ("NVIDIA A100 80GB", 640.0, 1152.0, 96, True), # 8× A100 80GB
    "p4de.24xlarge":     ("NVIDIA A100 80GB", 640.0, 1152.0, 96, True),

    # p5 — NVIDIA H100 (Hopper, BF16)
    "ml.p5.48xlarge":    ("NVIDIA H100",      640.0, 2048.0, 192, True), # 8× H100 80GB
    "p5.48xlarge":       ("NVIDIA H100",      640.0, 2048.0, 192, True),

    # ── GCP Compute Engine / Vertex AI ────────────────────────────────────────
    # n1 + T4 accelerator
    "n1-standard-4-t4":  ("NVIDIA Tesla T4",  16.0,  15.0,  4, False),
    "n1-standard-8-t4":  ("NVIDIA Tesla T4",  16.0,  30.0,  8, False),
    "n1-standard-16-t4": ("NVIDIA Tesla T4",  64.0,  60.0, 16, False),  # 4× T4

    # n1 + V100
    "n1-standard-8-v100":  ("NVIDIA Tesla V100", 16.0,  30.0,  8, False),
    "n1-standard-16-v100": ("NVIDIA Tesla V100", 64.0,  60.0, 16, False),

    # a2 — A100 40 GB
    "a2-highgpu-1g":     ("NVIDIA A100",       40.0,  85.0, 12, True),
    "a2-highgpu-2g":     ("NVIDIA A100",       80.0, 170.0, 24, True),
    "a2-highgpu-4g":     ("NVIDIA A100",      160.0, 340.0, 48, True),
    "a2-highgpu-8g":     ("NVIDIA A100",      320.0, 680.0, 96, True),

    # a2 ultra — A100 80 GB
    "a2-ultragpu-1g":    ("NVIDIA A100 80GB",  80.0,  170.0, 12, True),
    "a2-ultragpu-2g":    ("NVIDIA A100 80GB", 160.0,  340.0, 24, True),
    "a2-ultragpu-4g":    ("NVIDIA A100 80GB", 320.0,  680.0, 48, True),

    # a3 — H100 80 GB
    "a3-highgpu-1g":     ("NVIDIA H100",       80.0,  234.0, 26, True),
    "a3-highgpu-2g":     ("NVIDIA H100",      160.0,  468.0, 52, True),
    "a3-highgpu-4g":     ("NVIDIA H100",      320.0,  936.0, 104, True),
    "a3-highgpu-8g":     ("NVIDIA H100",      640.0, 1872.0, 208, True),

    # g2 — L4
    "g2-standard-4":     ("NVIDIA L4",         24.0,  16.0,  4, True),
    "g2-standard-8":     ("NVIDIA L4",         24.0,  32.0,  8, True),
    "g2-standard-16":    ("NVIDIA L4",         48.0,  64.0, 16, True),  # 2× L4

    # ── Azure ─────────────────────────────────────────────────────────────────
    # NC v3 — V100
    "standard_nc6s_v3":  ("NVIDIA Tesla V100", 16.0, 112.0,  6, False),
    "standard_nc12s_v3": ("NVIDIA Tesla V100", 32.0, 224.0, 12, False),
    "standard_nc24s_v3": ("NVIDIA Tesla V100", 64.0, 448.0, 24, False),

    # NC A100 v4
    "standard_nc24ads_a100_v4":  ("NVIDIA A100 80GB",  80.0, 220.0, 24, True),
    "standard_nc48ads_a100_v4":  ("NVIDIA A100 80GB", 160.0, 440.0, 48, True),
    "standard_nc96ads_a100_v4":  ("NVIDIA A100 80GB", 320.0, 880.0, 96, True),

    # ND A100 v4
    "standard_nd96asr_v4":       ("NVIDIA A100",      320.0, 900.0,  96, True),  # 8× A100 40GB

    # NC H100 v5
    "standard_nc40ads_h100_v5":  ("NVIDIA H100",       80.0, 320.0,  40, True),
    "standard_nc80adis_h100_v5": ("NVIDIA H100",      160.0, 640.0,  80, True),
}

# Short GPU aliases → canonical lookup key
_ALIASES: dict[str, str] = {
    "t4":        "ml.g4dn.xlarge",
    "a10g":      "ml.g5.xlarge",
    "l4":        "ml.g6.xlarge",
    "v100":      "ml.p3.2xlarge",
    "a100":      "a2-highgpu-1g",
    "a100-40gb": "a2-highgpu-1g",
    "a100-80gb": "a2-ultragpu-1g",
    "h100":      "a3-highgpu-1g",
}


# ── public API ────────────────────────────────────────────────────────────────


def get_instance_profile(target: str) -> HardwareProfile:
    """Return a HardwareProfile for *target*.

    *target* may be a canonical instance type (e.g. ``ml.g4dn.xlarge``,
    ``a2-highgpu-1g``) or a short GPU alias (``t4``, ``a100``, ``h100``).
    Case-insensitive.

    Raises ValueError if the target is not recognised.
    """
    key = target.strip().lower()
    key = _ALIASES.get(key, key)

    entry = _INSTANCES.get(key)
    if entry is None:
        raise ValueError(
            f"Unknown target: {target!r}.\n"
            f"Run 'aphex targets' to see available instance types and aliases."
        )

    gpu_name, vram_gb, ram_gb, vcpus, bf16 = entry
    bw, fp16_tflops, int8_tops = _lookup_gpu_specs(gpu_name)

    cpu = CPUProfile(
        name=f"Cloud vCPU ({vcpus} vCPU)",
        physical_cores=vcpus // 2,
        logical_cores=vcpus,
        ram_gb=ram_gb,
    )
    accelerator = AcceleratorProfile(
        kind="cuda",
        name=gpu_name,
        memory_gb=vram_gb,
        bf16=bf16,
        cuda_compute="",
        memory_bandwidth_gbps=bw,
        fp16_tflops=fp16_tflops,
        int8_tops=int8_tops,
    )
    return HardwareProfile(cpu=cpu, accelerator=accelerator, platform=f"cloud/{key}")


def list_targets() -> dict[str, list[str]]:
    """Return all supported targets grouped by provider."""
    aws = sorted(k for k in _INSTANCES if k.startswith(("ml.", "g4", "g5", "g6", "p3", "p4", "p5")))
    gcp = sorted(k for k in _INSTANCES if k.startswith(("n1-", "a2-", "a3-", "g2-")))
    azure = sorted(k for k in _INSTANCES if k.startswith("standard_"))
    aliases = sorted(_ALIASES)
    return {"aws": aws, "gcp": gcp, "azure": azure, "aliases": aliases}
