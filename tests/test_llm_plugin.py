"""Tests for the LLM plugin — GGUF parsing, HF config inspection, candidate generation."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from infermap.benchmark import BenchmarkResult
from infermap.candidates import DeploymentCandidate
from infermap.inspector import ModelInfo
from infermap.plugins.llm import (
    LLMPlugin,
    _LLM_MEASURE_CAP,
    _LLM_WARMUP_CAP,
    _config_is_llm,
    _inspect_gguf,
    _inspect_hf_config,
    _make_result,
    _read_gguf_metadata,
)
from infermap.profiler import AcceleratorProfile, CPUProfile, HardwareProfile


# ── Helpers ────────────────────────────────────────────────────────────────────

def _gguf_bytes(version: int = 3, kv_pairs: list[tuple[str, int, Any]] | None = None) -> bytes:
    """Build a minimal GGUF binary (no tensor data) for unit tests."""
    buf = b"GGUF" + struct.pack("<I", version)
    pairs = kv_pairs or []
    if version >= 2:
        buf += struct.pack("<Q", 0)           # n_tensors (uint64)
        buf += struct.pack("<Q", len(pairs))  # n_kv (uint64)
    else:
        buf += struct.pack("<I", 0)           # n_tensors (uint32 in v1)
        buf += struct.pack("<I", len(pairs))  # n_kv (uint32 in v1)
    for key, vtype, value in pairs:
        key_b = key.encode()
        buf += struct.pack("<Q", len(key_b)) + key_b
        buf += struct.pack("<I", vtype)
        if vtype == 8:  # string: uint64 length + bytes
            val_b = value.encode()
            buf += struct.pack("<Q", len(val_b)) + val_b
        elif vtype == 4:  # uint32
            buf += struct.pack("<I", int(value))
        elif vtype == 10:  # uint64
            buf += struct.pack("<Q", int(value))
    return buf


def _hw(kind: str = "none", bf16: bool = False) -> HardwareProfile:
    cpu = CPUProfile(name="x", physical_cores=4, logical_cores=8, ram_gb=16.0)
    acc = AcceleratorProfile(kind=kind, name="y", memory_gb=16.0, bf16=bf16)  # type: ignore[arg-type]
    return HardwareProfile(cpu=cpu, accelerator=acc)


def _llm_info(framework: str = "llama_cpp") -> ModelInfo:
    return ModelInfo(
        framework=framework,
        family="llm",
        parameters=1_000_000,
        trainable_parameters=0,
        estimated_memory_fp32_gb=0.004,
        estimated_memory_fp16_gb=0.002,
    )


# ── _read_gguf_metadata ────────────────────────────────────────────────────────

def test_read_gguf_metadata_empty_on_bad_magic(tmp_path: Path) -> None:
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"NOTG" + b"\x00" * 20)
    assert _read_gguf_metadata(p) == {}


def test_read_gguf_metadata_empty_on_truncated_file(tmp_path: Path) -> None:
    p = tmp_path / "tiny.gguf"
    p.write_bytes(b"")
    assert _read_gguf_metadata(p) == {}


def test_read_gguf_metadata_parses_v3_string_kv(tmp_path: Path) -> None:
    p = tmp_path / "model.gguf"
    p.write_bytes(_gguf_bytes(version=3, kv_pairs=[("general.architecture", 8, "llama")]))
    assert _read_gguf_metadata(p).get("general.architecture") == "llama"


def test_read_gguf_metadata_parses_v3_uint32_kv(tmp_path: Path) -> None:
    p = tmp_path / "model.gguf"
    p.write_bytes(_gguf_bytes(version=3, kv_pairs=[("general.file_type", 4, 15)]))
    assert _read_gguf_metadata(p).get("general.file_type") == 15


def test_read_gguf_metadata_parses_v1_with_uint32_counts(tmp_path: Path) -> None:
    # Version 1 uses uint32 for n_tensors/n_kv instead of uint64.
    p = tmp_path / "v1.gguf"
    p.write_bytes(_gguf_bytes(version=1, kv_pairs=[("general.architecture", 8, "gpt2")]))
    assert _read_gguf_metadata(p).get("general.architecture") == "gpt2"


def test_read_gguf_metadata_multiple_kv(tmp_path: Path) -> None:
    pairs = [
        ("general.architecture", 8, "llama"),
        ("general.file_type", 4, 15),
    ]
    p = tmp_path / "model.gguf"
    p.write_bytes(_gguf_bytes(version=3, kv_pairs=pairs))
    meta = _read_gguf_metadata(p)
    assert meta["general.architecture"] == "llama"
    assert meta["general.file_type"] == 15


# ── LLMPlugin.can_handle ──────────────────────────────────────────────────────

class TestCanHandle:
    plugin = LLMPlugin()

    def test_gguf_suffix_str(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        p.write_bytes(b"GGUF")
        assert self.plugin.can_handle(str(p))

    def test_gguf_suffix_path_object(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        p.write_bytes(b"GGUF")
        assert self.plugin.can_handle(p)

    def test_hf_model_id_string(self) -> None:
        # Slash-separated string that doesn't exist locally
        assert self.plugin.can_handle("meta-llama/Llama-3.2-1B")

    def test_local_dir_with_llm_config(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"architectures": ["LlamaForCausalLM"]})
        )
        assert self.plugin.can_handle(str(tmp_path))

    def test_local_dir_with_non_llm_config(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"architectures": ["ResNet50"]})
        )
        assert not self.plugin.can_handle(str(tmp_path))

    def test_nn_module_returns_false(self) -> None:
        import torch.nn as nn

        assert not self.plugin.can_handle(nn.Linear(4, 4))

    def test_plain_pt_file_returns_false(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pt"
        p.write_bytes(b"fake")
        assert not self.plugin.can_handle(str(p))


# ── _config_is_llm ────────────────────────────────────────────────────────────

def test_config_is_llm_true_for_llama(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    assert _config_is_llm(cfg)


def test_config_is_llm_true_for_gpt(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"architectures": ["GPTNeoXForCausalLM"]}))
    assert _config_is_llm(cfg)


def test_config_is_llm_true_for_mistral(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"architectures": ["MistralForCausalLM"]}))
    assert _config_is_llm(cfg)


def test_config_is_llm_false_for_resnet(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"architectures": ["ResNet50"]}))
    assert not _config_is_llm(cfg)


def test_config_is_llm_false_for_missing_file(tmp_path: Path) -> None:
    assert not _config_is_llm(tmp_path / "nonexistent.json")


def test_config_is_llm_false_for_empty_architectures(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"architectures": []}))
    assert not _config_is_llm(cfg)


# ── _inspect_hf_config ────────────────────────────────────────────────────────

def test_inspect_hf_config_computes_params(tmp_path: Path) -> None:
    hidden, n_layers, vocab, intermediate = 512, 12, 32000, 2048
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({
            "hidden_size": hidden,
            "num_hidden_layers": n_layers,
            "vocab_size": vocab,
            "intermediate_size": intermediate,
        })
    )
    info = _inspect_hf_config(config_path, str(tmp_path))
    expected = vocab * hidden + n_layers * (4 * hidden**2 + 3 * hidden * intermediate)
    assert info.parameters == expected
    assert info.framework == "transformers"
    assert info.family == "llm"


def test_inspect_hf_config_zero_params_when_fields_absent(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({}))
    info = _inspect_hf_config(config_path, "some/model")
    assert info.parameters == 0
    assert info.framework == "transformers"


def test_inspect_hf_config_uses_n_embd_alias(tmp_path: Path) -> None:
    # GPT-2 style config uses n_embd / n_layer / vocab_size
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"n_embd": 256, "n_layer": 6, "vocab_size": 50257})
    )
    info = _inspect_hf_config(config_path, "gpt2-mini")
    assert info.parameters > 0


# ── _inspect_gguf ─────────────────────────────────────────────────────────────

def test_inspect_gguf_sets_framework_and_family(tmp_path: Path) -> None:
    p = tmp_path / "model.gguf"
    p.write_bytes(_gguf_bytes(version=3))
    info = _inspect_gguf(p)
    assert info.framework == "llama_cpp"
    assert info.family == "llm"


def test_inspect_gguf_file_size_fallback(tmp_path: Path) -> None:
    # Without block_count/embedding_length, falls back to file_size * 8 / bits.
    # file_type=15 → Q4_K_M = 4.5 bits
    data = _gguf_bytes(version=3, kv_pairs=[("general.file_type", 4, 15)]) + b"\x00" * 1000
    p = tmp_path / "model.gguf"
    p.write_bytes(data)
    info = _inspect_gguf(p)
    expected = int(p.stat().st_size * 8 / 4.5)
    assert info.parameters == expected


def test_inspect_gguf_unknown_file_type_defaults_to_8bit(tmp_path: Path) -> None:
    data = _gguf_bytes(version=3, kv_pairs=[("general.file_type", 4, 99)]) + b"\x00" * 800
    p = tmp_path / "model.gguf"
    p.write_bytes(data)
    info = _inspect_gguf(p)
    # _GGUF_BITS.get(99, 8.0) → 8.0
    expected = int(p.stat().st_size * 8 / 8.0)
    assert info.parameters == expected


# ── LLM candidate generation ──────────────────────────────────────────────────

class TestLLMCandidates:
    plugin = LLMPlugin()

    def test_gguf_cpu_only(self) -> None:
        cands = self.plugin.generate_candidates(_llm_info("llama_cpp"), _hw(kind="none"))
        assert [c.backend for c in cands] == ["llama_cpp_cpu"]

    def test_gguf_cuda_adds_gpu_candidate(self) -> None:
        backends = [c.backend for c in self.plugin.generate_candidates(_llm_info("llama_cpp"), _hw(kind="cuda"))]
        assert "llama_cpp_cpu" in backends
        assert "llama_cpp_gpu" in backends

    def test_gguf_mps_adds_gpu_candidate(self) -> None:
        backends = [c.backend for c in self.plugin.generate_candidates(_llm_info("llama_cpp"), _hw(kind="mps"))]
        assert "llama_cpp_gpu" in backends

    def test_gguf_gpu_candidate_device_matches_kind(self) -> None:
        cands = self.plugin.generate_candidates(_llm_info("llama_cpp"), _hw(kind="cuda"))
        gpu = next(c for c in cands if c.backend == "llama_cpp_gpu")
        assert gpu.device == "cuda"

    def test_transformers_cpu_only(self) -> None:
        backends = [c.backend for c in self.plugin.generate_candidates(_llm_info("transformers"), _hw(kind="none"))]
        assert "transformers_cpu_fp32" in backends
        assert not any("vllm" in b for b in backends)

    def test_transformers_cuda_includes_vllm_fp16(self) -> None:
        backends = [c.backend for c in self.plugin.generate_candidates(_llm_info("transformers"), _hw(kind="cuda"))]
        assert "vllm_cuda_fp16" in backends
        assert "transformers_cpu_fp32" in backends

    def test_transformers_cuda_bf16_includes_vllm_bf16(self) -> None:
        backends = [c.backend for c in self.plugin.generate_candidates(_llm_info("transformers"), _hw(kind="cuda", bf16=True))]
        assert "vllm_cuda_bf16" in backends

    def test_transformers_cuda_no_bf16_omits_vllm_bf16(self) -> None:
        backends = [c.backend for c in self.plugin.generate_candidates(_llm_info("transformers"), _hw(kind="cuda", bf16=False))]
        assert "vllm_cuda_bf16" not in backends

    def test_all_candidates_non_empty_description(self) -> None:
        for framework in ("llama_cpp", "transformers"):
            for kind in ("none", "cuda", "mps"):
                for c in self.plugin.generate_candidates(_llm_info(framework), _hw(kind=kind)):
                    assert c.description.strip(), f"Empty description for {c.backend}"


# ── Iteration cap ─────────────────────────────────────────────────────────────

def test_benchmark_caps_warmup_and_measure_iters() -> None:
    plugin = LLMPlugin()
    candidate = DeploymentCandidate(
        backend="llama_cpp_cpu",
        dtype="q_native",
        description="test",
        requires_export=False,
        device="cpu",
    )
    captured: dict[str, int] = {}

    def _fake_benchmark(cand: Any, model_path: str, n_warm: int, n_meas: int) -> BenchmarkResult:
        captured["n_warm"] = n_warm
        captured["n_meas"] = n_meas
        return BenchmarkResult(
            candidate=cand, latency_p50_ms=1.0, latency_p95_ms=1.0, latency_p99_ms=1.0,
            throughput_rps=10.0, memory_mb=100.0, batch_size=1,
        )

    with patch("infermap.plugins.llm._benchmark_llm", side_effect=_fake_benchmark):
        plugin.benchmark(
            candidate=candidate,
            model=Path("/tmp/fake.gguf"),
            info=_llm_info("llama_cpp"),
            input_shape=[],
            batch_size=1,
            warmup_iters=100,
            measure_iters=100,
            timeout_s=None,
            calibration_inputs=None,
        )

    assert captured["n_warm"] == _LLM_WARMUP_CAP
    assert captured["n_meas"] == _LLM_MEASURE_CAP


def test_benchmark_does_not_exceed_cap_when_iters_below_cap() -> None:
    plugin = LLMPlugin()
    candidate = DeploymentCandidate(
        backend="llama_cpp_cpu", dtype="q_native", description="x",
        requires_export=False, device="cpu",
    )
    captured: dict[str, int] = {}

    def _fake_benchmark(cand: Any, model_path: str, n_warm: int, n_meas: int) -> BenchmarkResult:
        captured["n_warm"] = n_warm
        captured["n_meas"] = n_meas
        return BenchmarkResult(
            candidate=cand, latency_p50_ms=1.0, latency_p95_ms=1.0, latency_p99_ms=1.0,
            throughput_rps=1.0, memory_mb=0.0, batch_size=1,
        )

    with patch("infermap.plugins.llm._benchmark_llm", side_effect=_fake_benchmark):
        plugin.benchmark(
            candidate=candidate,
            model=Path("/tmp/fake.gguf"),
            info=_llm_info("llama_cpp"),
            input_shape=[],
            batch_size=1,
            warmup_iters=1,
            measure_iters=2,
            timeout_s=None,
            calibration_inputs=None,
        )

    assert captured["n_warm"] == 1
    assert captured["n_meas"] == 2


# ── _make_result ──────────────────────────────────────────────────────────────

def _dummy_cand() -> DeploymentCandidate:
    return DeploymentCandidate(
        backend="llama_cpp_cpu", dtype="q_native", description="x",
        requires_export=False, device="cpu",
    )


def test_make_result_percentiles_from_sorted_list() -> None:
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    s = sorted(latencies)
    n = len(s)
    result = _make_result(_dummy_cand(), latencies, [5.0, 6.0], 200.0)
    assert result.latency_p50_ms == s[int(n * 0.50)]
    assert result.latency_p95_ms == s[int(n * 0.95)]
    assert result.latency_p99_ms == s[int(n * 0.99)]
    assert result.throughput_rps == pytest.approx(5.5)
    assert result.memory_mb == 200.0
    assert result.batch_size == 1


def test_make_result_throughput_zero_when_no_tps() -> None:
    result = _make_result(_dummy_cand(), [5.0], [], 0.0)
    assert result.throughput_rps == 0.0


def test_make_result_empty_latencies_raises() -> None:
    with pytest.raises(RuntimeError, match="no output"):
        _make_result(_dummy_cand(), [], [], 0.0)


def test_make_result_single_latency() -> None:
    result = _make_result(_dummy_cand(), [42.0], [10.0], 50.0)
    assert result.latency_p50_ms == 42.0
    assert result.latency_p95_ms == 42.0
    assert result.latency_p99_ms == 42.0


# ── LLMPlugin.load ────────────────────────────────────────────────────────────

def test_load_returns_path_object(tmp_path: Path) -> None:
    p = tmp_path / "model.gguf"
    p.write_bytes(b"GGUF")
    plugin = LLMPlugin()
    result = plugin.load(p)
    assert isinstance(result, Path)
    assert result == p
