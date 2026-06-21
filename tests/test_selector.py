"""Tests for infermap.selector — candidate selector."""
from __future__ import annotations

from infermap.candidates import generate_candidates
from infermap.inspector import ModelInfo
from infermap.selector import select_candidates

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _info(family: str = "cnn", params: int = 5_000_000) -> ModelInfo:
    return ModelInfo(
        framework="pytorch",
        family=family,
        parameters=params,
        trainable_parameters=params,
        estimated_memory_fp32_gb=(params * 4.0 * 1.2) / 1e9,
        estimated_memory_fp16_gb=(params * 2.0 * 1.2) / 1e9,
    )


def _cuda_hw(bf16: bool = False):
    from infermap.profiler import AcceleratorProfile, CPUProfile, HardwareProfile
    return HardwareProfile(
        cpu=CPUProfile(name="Intel Xeon", physical_cores=8, logical_cores=16, ram_gb=32.0),
        accelerator=AcceleratorProfile(
            kind="cuda",
            name="NVIDIA Tesla T4" if not bf16 else "NVIDIA A100",
            memory_gb=16.0 if not bf16 else 40.0,
            bf16=bf16,
            memory_bandwidth_gbps=320.0 if not bf16 else 2000.0,
            fp16_tflops=65.0 if not bf16 else 312.0,
            int8_tops=130.0 if not bf16 else 624.0,
        ),
        platform="test",
    )


def _mps_hw():
    from infermap.profiler import AcceleratorProfile, CPUProfile, HardwareProfile
    return HardwareProfile(
        cpu=CPUProfile(name="Apple M2", physical_cores=10, logical_cores=10, ram_gb=16.0),
        accelerator=AcceleratorProfile(
            kind="mps",
            name="Apple M2",
            memory_gb=16.0,
            bf16=False,
            memory_bandwidth_gbps=100.0,
            fp16_tflops=10.0,
            int8_tops=20.0,
        ),
        platform="test",
    )


def _cpu_hw():
    from infermap.profiler import AcceleratorProfile, CPUProfile, HardwareProfile
    return HardwareProfile(
        cpu=CPUProfile(name="Intel Xeon", physical_cores=8, logical_cores=16, ram_gb=32.0),
        accelerator=AcceleratorProfile(
            kind="none",
            name="CPU",
            memory_gb=0.0,
            bf16=False,
            memory_bandwidth_gbps=50.0,
            fp16_tflops=None,
            int8_tops=None,
        ),
        platform="test",
    )


# ── Basic contract ─────────────────────────────────────────────────────────────


def test_select_returns_at_most_k() -> None:
    info = _info("cnn")
    hw = _cuda_hw()
    candidates = generate_candidates(info, hw)
    selected, _ = select_candidates(candidates, info, hw, k=4)
    assert len(selected) <= 4


def test_select_returns_subset_of_candidates() -> None:
    info = _info("cnn")
    hw = _cuda_hw()
    candidates = generate_candidates(info, hw)
    selected, _ = select_candidates(candidates, info, hw, k=4)
    candidate_ids = {c.backend for c in candidates}
    for c in selected:
        assert c.backend in candidate_ids


def test_select_no_duplicates() -> None:
    info = _info("cnn")
    hw = _cuda_hw()
    candidates = generate_candidates(info, hw)
    selected, _ = select_candidates(candidates, info, hw, k=4)
    ids = [c.backend for c in selected]
    assert len(ids) == len(set(ids))


def test_select_empty_candidates_returns_empty() -> None:
    info = _info("cnn")
    hw = _cuda_hw()
    selected, rationale = select_candidates([], info, hw, k=4)
    assert selected == []
    assert "no candidates" in rationale


def test_select_k_larger_than_candidates_returns_all() -> None:
    info = _info("cnn")
    hw = _cpu_hw()
    candidates = generate_candidates(info, hw)
    selected, _ = select_candidates(candidates, info, hw, k=100)
    assert len(selected) == len(candidates)


# ── CNN + CUDA rules ──────────────────────────────────────────────────────────


def test_cnn_cuda_no_bf16_prefers_tensorrt_fp16() -> None:
    info = _info("cnn")
    hw = _cuda_hw(bf16=False)
    candidates = generate_candidates(info, hw)
    selected, _ = select_candidates(candidates, info, hw, k=4)
    assert selected[0].backend == "tensorrt_fp16"


def test_cnn_cuda_bf16_prefers_tensorrt_fp16() -> None:
    info = _info("cnn")
    hw = _cuda_hw(bf16=True)
    candidates = generate_candidates(info, hw)
    selected, _ = select_candidates(candidates, info, hw, k=4)
    assert selected[0].backend == "tensorrt_fp16"


def test_cnn_cuda_bf16_includes_pytorch_bf16() -> None:
    info = _info("cnn")
    hw = _cuda_hw(bf16=True)
    candidates = generate_candidates(info, hw)
    selected, _ = select_candidates(candidates, info, hw, k=4)
    ids = [c.backend for c in selected]
    assert "pytorch_bf16" in ids


def test_cnn_cuda_no_bf16_excludes_pytorch_bf16() -> None:
    # No bf16 hardware → pytorch_bf16 not generated at all
    info = _info("cnn")
    hw = _cuda_hw(bf16=False)
    candidates = generate_candidates(info, hw)
    selected, _ = select_candidates(candidates, info, hw, k=4)
    ids = [c.backend for c in selected]
    assert "pytorch_bf16" not in ids


# ── Transformer + CUDA rules ──────────────────────────────────────────────────


def test_transformer_cuda_bf16_prefers_pytorch_bf16() -> None:
    info = _info("transformer", params=120_000_000)
    hw = _cuda_hw(bf16=True)
    candidates = generate_candidates(info, hw)
    selected, _ = select_candidates(candidates, info, hw, k=4)
    assert selected[0].backend == "pytorch_bf16"


def test_transformer_cuda_no_bf16_prefers_fp16() -> None:
    info = _info("transformer", params=120_000_000)
    hw = _cuda_hw(bf16=False)
    candidates = generate_candidates(info, hw)
    selected, _ = select_candidates(candidates, info, hw, k=4)
    assert selected[0].backend == "pytorch_fp16"


# ── MPS rules ─────────────────────────────────────────────────────────────────


def test_cnn_mps_prefers_pytorch_fp16() -> None:
    info = _info("cnn")
    hw = _mps_hw()
    candidates = generate_candidates(info, hw)
    selected, _ = select_candidates(candidates, info, hw, k=4)
    assert selected[0].backend == "pytorch_fp16"


def test_mps_includes_onnx_coreml() -> None:
    info = _info("cnn")
    hw = _mps_hw()
    candidates = generate_candidates(info, hw)
    selected, _ = select_candidates(candidates, info, hw, k=4)
    ids = [c.backend for c in selected]
    assert "onnx_coreml" in ids


# ── CPU rules ─────────────────────────────────────────────────────────────────


def test_cnn_cpu_prefers_onnx_int8() -> None:
    info = _info("cnn")
    hw = _cpu_hw()
    candidates = generate_candidates(info, hw)
    selected, _ = select_candidates(candidates, info, hw, k=4)
    assert selected[0].backend == "onnx_int8_cpu"


def test_transformer_cpu_prefers_pytorch_int8_dynamic() -> None:
    info = _info("transformer", params=120_000_000)
    hw = _cpu_hw()
    candidates = generate_candidates(info, hw)
    selected, _ = select_candidates(candidates, info, hw, k=4)
    assert selected[0].backend == "pytorch_int8_dynamic"


# ── Unknown family falls back gracefully ──────────────────────────────────────


def test_unknown_family_cuda_still_selects_candidates() -> None:
    info = _info("unknown")
    hw = _cuda_hw()
    candidates = generate_candidates(info, hw)
    selected, rationale = select_candidates(candidates, info, hw, k=4)
    assert len(selected) >= 1
    assert "tensorrt_fp16" in [c.backend for c in selected]


def test_unknown_family_cpu_still_selects_candidates() -> None:
    info = _info("unknown")
    hw = _cpu_hw()
    candidates = generate_candidates(info, hw)
    selected, _ = select_candidates(candidates, info, hw, k=4)
    assert len(selected) >= 1


# ── Rationale string ──────────────────────────────────────────────────────────


def test_rationale_contains_family() -> None:
    info = _info("cnn")
    hw = _cuda_hw()
    candidates = generate_candidates(info, hw)
    _, rationale = select_candidates(candidates, info, hw, k=4)
    assert "cnn" in rationale


def test_rationale_contains_hw_kind() -> None:
    info = _info("cnn")
    hw = _cuda_hw()
    candidates = generate_candidates(info, hw)
    _, rationale = select_candidates(candidates, info, hw, k=4)
    assert "CUDA" in rationale


def test_rationale_contains_param_count() -> None:
    info = _info("cnn", params=5_300_000)
    hw = _cuda_hw()
    candidates = generate_candidates(info, hw)
    _, rationale = select_candidates(candidates, info, hw, k=4)
    assert "5.3M" in rationale


def test_rationale_mentions_top_backend() -> None:
    info = _info("cnn")
    hw = _cuda_hw()
    candidates = generate_candidates(info, hw)
    selected, rationale = select_candidates(candidates, info, hw, k=4)
    assert selected[0].backend in rationale


def test_rationale_shows_n_of_total() -> None:
    info = _info("cnn")
    hw = _cuda_hw()
    candidates = generate_candidates(info, hw)
    selected, rationale = select_candidates(candidates, info, hw, k=4)
    assert f"{len(selected)} of {len(candidates)}" in rationale
