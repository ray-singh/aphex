"""LLM plugin — handles .gguf files (llama.cpp) and HuggingFace model directories/IDs."""

from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Any

from aphex.benchmark import BenchmarkResult
from aphex.candidates import DeploymentCandidate
from aphex.inspector import ModelInfo
from aphex.plugin import ModelPlugin
from aphex.profiler import HardwareProfile

_GGUF_MAGIC = b"GGUF"

# Iteration caps: each LLM run generates many tokens — default 100 iters would be minutes.
_LLM_WARMUP_CAP = 2
_LLM_MEASURE_CAP = 5
_LLM_N_GEN = 64  # tokens to generate per measurement run

# ~20-token prompt used during benchmarking
_BENCH_PROMPT = (
    "Explain the key differences between supervised and unsupervised machine learning."
)

# GGUF general.file_type → approximate bits per model parameter
_GGUF_BITS: dict[int, float] = {
    0: 32.0,   # F32
    1: 16.0,   # F16
    2: 4.0,    # Q4_0
    3: 4.0,    # Q4_1
    6: 5.0,    # Q5_0
    7: 5.0,    # Q5_1
    8: 8.0,    # Q8_0
    10: 2.0,   # Q2_K
    11: 3.0,   # Q3_K_S
    12: 3.5,   # Q3_K_M
    13: 3.5,   # Q3_K_L
    14: 4.0,   # Q4_K_S
    15: 4.5,   # Q4_K_M
    16: 4.0,   # Q4_K_S (duplicate alias)
    17: 5.5,   # Q5_K_M
    18: 5.0,   # Q5_K_S
    19: 6.5,   # Q6_K
}

_LLM_ARCH_HINTS = frozenset({
    "causallm", "llama", "gpt", "mistral", "falcon", "qwen",
    "phi", "gemma", "bloom", "opt", "codegen", "mpt", "rwkv",
})


class LLMPlugin(ModelPlugin):
    name = "llm"

    def can_handle(self, path_or_model: Any) -> bool:
        if not isinstance(path_or_model, (str, Path)):
            return False
        p = Path(str(path_or_model))
        if p.suffix.lower() == ".gguf":
            return True
        if p.is_dir() and (p / "config.json").exists():
            return _config_is_llm(p / "config.json")
        # HuggingFace model ID pattern (org/model) that doesn't exist as a local path
        return isinstance(path_or_model, str) and "/" in path_or_model and not p.exists()

    def inspect(self, path_or_model: Any, input_shape: list[int] | None = None) -> ModelInfo:
        p = Path(str(path_or_model))
        if p.suffix.lower() == ".gguf" and p.exists():
            return _inspect_gguf(p)
        if p.is_dir() and (p / "config.json").exists():
            return _inspect_hf_config(p / "config.json", str(p))
        return _inspect_hf_id(str(path_or_model))

    def load(self, path: Path) -> Any:
        # Return the path as-is — LLM runtimes load lazily inside benchmark().
        return Path(path)

    def generate_candidates(
        self, info: ModelInfo, hardware: HardwareProfile
    ) -> list[DeploymentCandidate]:
        return _llm_candidates(info, hardware)

    def benchmark(
        self,
        candidate: DeploymentCandidate,
        model: Any,
        info: ModelInfo,
        input_shape: list[int],
        batch_size: int,
        warmup_iters: int,
        measure_iters: int,
        timeout_s: float | None,
        calibration_inputs: list[Any] | None,
        input_spec: Any = None,
    ) -> BenchmarkResult:
        n_warm = min(warmup_iters, _LLM_WARMUP_CAP)
        n_meas = min(measure_iters, _LLM_MEASURE_CAP)
        return _benchmark_llm(candidate, str(model), n_warm, n_meas)


# ── GGUF header parser ─────────────────────────────────────────────────────────

def _gguf_read_string(f: Any) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def _gguf_read_value(f: Any, vtype: int) -> Any:
    _SCALAR_FMT = {
        0: "<B", 1: "<b", 2: "<H", 3: "<h",
        4: "<I", 5: "<i", 6: "<f", 7: "<B",
        10: "<Q", 11: "<q", 12: "<d",
    }
    if vtype == 8:
        return _gguf_read_string(f)
    if vtype == 9:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        return [_gguf_read_value(f, elem_type) for _ in range(min(count, 4096))]
    fmt = _SCALAR_FMT.get(vtype, "<I")
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def _read_gguf_metadata(path: Path) -> dict[str, Any]:
    """Parse GGUF header KV metadata; return {} on any parse failure."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != _GGUF_MAGIC:
                return {}
            (version,) = struct.unpack("<I", f.read(4))
            if version >= 2:
                (n_tensors,) = struct.unpack("<Q", f.read(8))
                (n_kv,) = struct.unpack("<Q", f.read(8))
            else:
                (n_tensors,) = struct.unpack("<I", f.read(4))
                (n_kv,) = struct.unpack("<I", f.read(4))

            meta: dict[str, Any] = {}
            for _ in range(min(n_kv, 512)):
                key = _gguf_read_string(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                meta[key] = _gguf_read_value(f, vtype)
            return meta
    except Exception:
        return {}


# ── Inspection ─────────────────────────────────────────────────────────────────

def _inspect_gguf(path: Path) -> ModelInfo:
    meta = _read_gguf_metadata(path)
    arch = str(meta.get("general.architecture", "unknown"))
    file_type = int(meta.get("general.file_type", 1))
    bits = _GGUF_BITS.get(file_type, 8.0)

    # Estimate from file size first, then refine from architecture dimensions if available.
    file_bytes = path.stat().st_size
    estimated_params = int(file_bytes * 8 / bits) if bits > 0 else 0

    block_count = meta.get(f"{arch}.block_count", 0)
    embed_len = meta.get(f"{arch}.embedding_length", 0)
    ff_len = meta.get(f"{arch}.feed_forward_length", embed_len * 4)
    if block_count and embed_len:
        per_layer = 4 * embed_len ** 2 + 3 * embed_len * ff_len
        vocab_size = meta.get(f"{arch}.vocab_size", meta.get("tokenizer.ggml.tokens", 0))
        if isinstance(vocab_size, list):
            vocab_size = len(vocab_size)
        estimated_params = int(block_count * per_layer + int(vocab_size) * embed_len)

    vocab_size = meta.get(f"{arch}.vocab_size", meta.get("tokenizer.ggml.tokens", 0))
    if isinstance(vocab_size, list):
        vocab_size = len(vocab_size)

    n_heads = meta.get(f"{arch}.attention.head_count")

    return ModelInfo(
        framework="llama_cpp",
        family="llm",
        parameters=estimated_params,
        trainable_parameters=0,
        estimated_memory_fp32_gb=(estimated_params * 4.0) / 1e9,
        estimated_memory_fp16_gb=(estimated_params * 2.0) / 1e9,
        model_path=str(path),
        num_layers=int(block_count) if block_count else None,
        num_attention_heads=int(n_heads) if n_heads else None,
        hidden_size=int(embed_len) if embed_len else None,
        vocab_size=int(vocab_size) if vocab_size else None,
    )


def _inspect_hf_config(config_path: Path, model_path: str) -> ModelInfo:
    import json

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except Exception:
        return _minimal_info(model_path, "transformers")

    hidden = cfg.get("hidden_size", cfg.get("n_embd", cfg.get("d_model", 0)))
    n_layers = cfg.get("num_hidden_layers", cfg.get("n_layer", cfg.get("num_layers", 0)))
    n_heads = cfg.get("num_attention_heads", cfg.get("n_head", cfg.get("num_heads")))
    vocab = cfg.get("vocab_size", 0)
    max_seq = cfg.get("max_position_embeddings", cfg.get("n_positions", cfg.get("seq_length")))
    intermediate = cfg.get("intermediate_size", hidden * 4)
    params = vocab * hidden + n_layers * (4 * hidden ** 2 + 3 * hidden * intermediate)

    return ModelInfo(
        framework="transformers",
        family="llm",
        parameters=params,
        trainable_parameters=params,
        estimated_memory_fp32_gb=(params * 4.0) / 1e9,
        estimated_memory_fp16_gb=(params * 2.0) / 1e9,
        model_path=model_path,
        num_layers=int(n_layers) if n_layers else None,
        num_attention_heads=int(n_heads) if n_heads else None,
        hidden_size=int(hidden) if hidden else None,
        vocab_size=int(vocab) if vocab else None,
        max_sequence_length=int(max_seq) if max_seq else None,
    )


def _inspect_hf_id(model_id: str) -> ModelInfo:
    try:
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(repo_id=model_id, filename="config.json"))
        return _inspect_hf_config(config_path, model_id)
    except Exception:
        return _minimal_info(model_id, "transformers")


def _minimal_info(model_path: str, framework: str) -> ModelInfo:
    return ModelInfo(
        framework=framework,
        family="llm",
        parameters=0,
        trainable_parameters=0,
        estimated_memory_fp32_gb=0.0,
        estimated_memory_fp16_gb=0.0,
        model_path=model_path,
    )


def _config_is_llm(config_path: Path) -> bool:
    try:
        import json

        with open(config_path) as f:
            cfg = json.load(f)
        arch = ((cfg.get("architectures") or [""])[0]).lower()
        return any(h in arch for h in _LLM_ARCH_HINTS)
    except Exception:
        return False


# ── Candidate generation ───────────────────────────────────────────────────────

def _llm_candidates(
    info: ModelInfo, hardware: HardwareProfile
) -> list[DeploymentCandidate]:
    kind = hardware.accelerator.kind
    candidates: list[DeploymentCandidate] = []

    if info.framework == "llama_cpp":
        candidates.append(
            DeploymentCandidate(
                backend="llama_cpp_cpu",
                dtype="q_native",
                description="llama.cpp CPU (native quantization)",
                requires_export=False,
                device="cpu",
            )
        )
        if kind in ("cuda", "mps"):
            candidates.append(
                DeploymentCandidate(
                    backend="llama_cpp_gpu",
                    dtype="q_native",
                    description=f"llama.cpp GPU-offloaded ({kind.upper()})",
                    requires_export=False,
                    device=kind,
                )
            )
    else:
        # HuggingFace / vLLM path
        if kind == "cuda":
            candidates.append(
                DeploymentCandidate(
                    backend="vllm_cuda_fp16",
                    dtype="fp16",
                    description="vLLM FP16 (PagedAttention, CUDA)",
                    requires_export=False,
                    device="cuda",
                )
            )
            if hardware.accelerator.bf16:
                candidates.append(
                    DeploymentCandidate(
                        backend="vllm_cuda_bf16",
                        dtype="bf16",
                        description="vLLM BF16 (PagedAttention, CUDA, Ampere+)",
                        requires_export=False,
                        device="cuda",
                    )
                )
        candidates.append(
            DeploymentCandidate(
                backend="transformers_cpu_fp32",
                dtype="fp32",
                description="HuggingFace Transformers FP32 (CPU)",
                requires_export=False,
                device="cpu",
            )
        )

    return candidates


# ── Benchmark runners ─────────────────────────────────────────────────────────

def _benchmark_llm(
    candidate: DeploymentCandidate,
    model_path: str,
    warmup_iters: int,
    measure_iters: int,
) -> BenchmarkResult:
    try:
        b = candidate.backend
        if b in ("llama_cpp_cpu", "llama_cpp_gpu"):
            return _run_llama_cpp(candidate, model_path, warmup_iters, measure_iters)
        if b in ("vllm_cuda_fp16", "vllm_cuda_bf16"):
            return _run_vllm(candidate, model_path, warmup_iters, measure_iters)
        if b == "transformers_cpu_fp32":
            return _run_transformers(candidate, model_path, warmup_iters, measure_iters)
        raise RuntimeError(f"Unknown LLM backend: {b!r}")
    except Exception as exc:
        return BenchmarkResult(
            candidate=candidate,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            throughput_rps=0.0,
            memory_mb=0.0,
            batch_size=1,
            error=str(exc),
        )


def _make_result(
    candidate: DeploymentCandidate,
    latencies_ms: list[float],
    tps: list[float],
    memory_mb: float,
) -> BenchmarkResult:
    if not latencies_ms:
        raise RuntimeError("All LLM benchmark runs produced no output.")
    s = sorted(latencies_ms)
    n = len(s)
    return BenchmarkResult(
        candidate=candidate,
        latency_p50_ms=s[int(n * 0.50)],
        latency_p95_ms=s[int(n * 0.95)],
        latency_p99_ms=s[int(n * 0.99)],
        throughput_rps=sum(tps) / len(tps) if tps else 0.0,
        memory_mb=memory_mb,
        batch_size=1,
    )


def _run_llama_cpp(
    candidate: DeploymentCandidate,
    model_path: str,
    warmup_iters: int,
    measure_iters: int,
) -> BenchmarkResult:
    from llama_cpp import Llama

    n_gpu = -1 if candidate.backend == "llama_cpp_gpu" else 0
    llm = Llama(model_path=model_path, n_gpu_layers=n_gpu, verbose=False, n_ctx=2048)

    # latency_p50_ms tracks TTFT (time-to-first-token, includes prompt prefill).
    # throughput_rps tracks decode tokens/sec.
    ttft_ms: list[float] = []
    tps: list[float] = []

    for i in range(warmup_iters + measure_iters):
        t0 = time.perf_counter()
        first: float | None = None
        n_tok = 0
        for _ in llm(_BENCH_PROMPT, max_tokens=_LLM_N_GEN, stream=True, echo=False):
            if first is None:
                first = (time.perf_counter() - t0) * 1000.0
            n_tok += 1
        elapsed = time.perf_counter() - t0
        if i >= warmup_iters and first is not None and n_tok > 0:
            ttft_ms.append(first)
            tps.append(n_tok / elapsed)

    memory_mb = Path(model_path).stat().st_size / 1e6
    return _make_result(candidate, ttft_ms, tps, memory_mb)


def _run_vllm(
    candidate: DeploymentCandidate,
    model_path: str,
    warmup_iters: int,
    measure_iters: int,
) -> BenchmarkResult:
    import torch
    from vllm import LLM, SamplingParams

    dtype = "bfloat16" if "bf16" in candidate.backend else "float16"
    llm = LLM(model=model_path, dtype=dtype)
    params = SamplingParams(max_tokens=_LLM_N_GEN, temperature=0.0)

    # Reset so the reported VRAM reflects this candidate, not a prior one's peak.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # vLLM offline mode reports total generation latency, not TTFT.
    latencies_ms: list[float] = []
    tps: list[float] = []

    for i in range(warmup_iters + measure_iters):
        t0 = time.perf_counter()
        outputs = llm.generate([_BENCH_PROMPT], params)
        elapsed = time.perf_counter() - t0
        if i >= warmup_iters:
            latencies_ms.append(elapsed * 1000.0)
            n_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
            tps.append(n_tok / elapsed)

    memory_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    return _make_result(candidate, latencies_ms, tps, memory_mb)


def _run_transformers(
    candidate: DeploymentCandidate,
    model_path: str,
    warmup_iters: int,
    measure_iters: int,
) -> BenchmarkResult:
    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="cpu"
    )
    model.eval()
    inputs = tokenizer(_BENCH_PROMPT, return_tensors="pt")
    n_prompt = inputs["input_ids"].shape[1]

    latencies_ms: list[float] = []
    tps: list[float] = []

    for i in range(warmup_iters + measure_iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=_LLM_N_GEN,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0
        if i >= warmup_iters:
            latencies_ms.append(elapsed * 1000.0)
            n_new = out_ids.shape[1] - n_prompt
            tps.append(n_new / max(elapsed, 1e-9))

    return _make_result(candidate, latencies_ms, tps, 0.0)
