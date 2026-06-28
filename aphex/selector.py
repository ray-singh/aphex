"""Candidate selector — uses model fingerprint + hardware profile to predict the top-k
most likely winning backends and benchmark only those instead of all candidates.

The selector sits between generate_candidates() and the benchmark loop in `aphex optimize`.
`aphex benchmark` remains exhaustive and does not use this module.

Design
------
A rule table maps (family, hw_kind, bf16) → ordered list of preferred backend IDs.
select_candidates() filters the already-generated (hardware-feasible) candidates list
to the top k matches, then fills any remaining slots using cost-model order so that
the returned list always contains min(k, len(candidates)) entries.
"""

from __future__ import annotations

from aphex.candidates import DeploymentCandidate
from aphex.inspector import ModelInfo
from aphex.profiler import HardwareProfile

# Rule table: (family, hw_kind, bf16_support) → ordered preference list.
# Keys use "*" as a wildcard for bf16 when the rule applies regardless.
# More specific keys (explicit bf16 value) are checked first.
_RULES: dict[tuple[str, str, str], list[str]] = {
    # ── CUDA, CNN ─────────────────────────────────────────────────────────────
    ("cnn", "cuda", "true"): [
        "tensorrt_fp16",
        "tensorrt_int8",
        "pytorch_bf16",
        "pytorch_fp16",
        "onnx_cuda",
        "torch_compile_fp32",
        "pytorch_fp32",
    ],
    ("cnn", "cuda", "false"): [
        "tensorrt_fp16",
        "tensorrt_int8",
        "pytorch_fp16",
        "onnx_cuda",
        "torch_compile_fp32",
        "pytorch_fp32",
    ],
    # ── CUDA, Transformer / LLM ───────────────────────────────────────────────
    ("transformer", "cuda", "true"): [
        "pytorch_bf16",
        "tensorrt_fp16",
        "pytorch_fp16",
        "torch_compile_fp32",
        "onnx_cuda",
        "pytorch_fp32",
    ],
    ("transformer", "cuda", "false"): [
        "pytorch_fp16",
        "tensorrt_fp16",
        "onnx_cuda",
        "torch_compile_fp32",
        "pytorch_fp32",
    ],
    ("llm", "cuda", "true"): [
        "pytorch_bf16",
        "tensorrt_fp16",
        "pytorch_fp16",
        "torch_compile_fp32",
        "onnx_cuda",
        "pytorch_fp32",
    ],
    ("llm", "cuda", "false"): [
        "pytorch_fp16",
        "tensorrt_fp16",
        "torch_compile_fp32",
        "onnx_cuda",
        "pytorch_fp32",
    ],
    # ── MPS (Apple Silicon) ───────────────────────────────────────────────────
    ("cnn", "mps", "*"): [
        "pytorch_fp16",
        "onnx_coreml",
        "pytorch_fp32",
        "torch_compile_fp32",
        "pytorch_int8_dynamic",
    ],
    ("transformer", "mps", "*"): [
        "pytorch_fp16",
        "pytorch_fp32",
        "onnx_coreml",
        "torch_compile_fp32",
        "pytorch_int8_dynamic",
    ],
    ("llm", "mps", "*"): [
        "pytorch_fp16",
        "pytorch_fp32",
        "onnx_coreml",
        "torch_compile_fp32",
    ],
    # ── CPU only ──────────────────────────────────────────────────────────────
    ("cnn", "cpu", "*"): [
        "onnx_int8_cpu",
        "openvino_int8",
        "openvino_fp32",
        "pytorch_int8_dynamic",
        "onnx_cpu",
        "torch_compile_fp32",
        "pytorch_fp32",
    ],
    ("transformer", "cpu", "*"): [
        "pytorch_int8_dynamic",
        "onnx_int8_cpu",
        "torch_compile_fp32",
        "openvino_fp32",
        "onnx_cpu",
        "pytorch_fp32",
    ],
    ("llm", "cpu", "*"): [
        "pytorch_int8_dynamic",
        "onnx_int8_cpu",
        "torch_compile_fp32",
        "onnx_cpu",
        "pytorch_fp32",
    ],
}

# Fallback generic rules when family is "unknown" or has no specific rule.
_GENERIC_RULES: dict[tuple[str, str], list[str]] = {
    ("cuda", "true"):  ["tensorrt_fp16", "pytorch_bf16", "pytorch_fp16", "onnx_cuda", "torch_compile_fp32"],
    ("cuda", "false"): ["tensorrt_fp16", "pytorch_fp16", "onnx_cuda", "torch_compile_fp32", "pytorch_fp32"],
    ("mps",  "*"):     ["pytorch_fp16", "onnx_coreml", "pytorch_fp32", "torch_compile_fp32"],
    ("cpu",  "*"):     ["onnx_int8_cpu", "pytorch_int8_dynamic", "openvino_fp32", "onnx_cpu", "pytorch_fp32"],
}


def select_candidates(
    candidates: list[DeploymentCandidate],
    info: ModelInfo,
    hw: HardwareProfile,
    k: int = 4,
) -> tuple[list[DeploymentCandidate], str]:
    """Return (selected, rationale) where selected is at most k candidates.

    The selected list is ordered so that the most promising backend is first.
    If the rule table produces fewer than k matches, remaining slots are filled
    from the cost-model-ranked tail so the caller always has something to benchmark.
    """
    if not candidates:
        return [], _rationale(info, hw, [], total=0)

    raw_kind = hw.accelerator.kind  # "cuda", "mps", "none" (CPU-only)
    hw_kind: str = "cpu" if raw_kind == "none" else raw_kind
    bf16 = "true" if hw.accelerator.bf16 else "false"
    family = info.family.lower()

    priority = (
        _RULES.get((family, hw_kind, bf16))
        or _RULES.get((family, hw_kind, "*"))
        or _GENERIC_RULES.get((hw_kind, bf16))
        or _GENERIC_RULES.get((hw_kind, "*"))
        or []
    )

    by_id: dict[str, DeploymentCandidate] = {c.backend: c for c in candidates}

    selected: list[DeploymentCandidate] = []
    seen: set[str] = set()

    for backend_id in priority:
        if len(selected) >= k:
            break
        if backend_id in by_id and backend_id not in seen:
            selected.append(by_id[backend_id])
            seen.add(backend_id)

    # Fill remaining slots from cost model order (unranked remainder).
    if len(selected) < k:
        from aphex.cost_model import estimate
        remainder = [c for c in candidates if c.backend not in seen]
        remainder.sort(key=lambda c: estimate(c, info, hw).predicted_latency_ms)
        for c in remainder:
            if len(selected) >= k:
                break
            selected.append(c)
            seen.add(c.backend)

    rationale = _rationale(info, hw, selected, total=len(candidates))
    return selected, rationale


def _rationale(
    info: ModelInfo,
    hw: HardwareProfile,
    selected: list[DeploymentCandidate],
    total: int,
) -> str:
    family_label = info.family if info.family != "unknown" else "unknown arch"
    params_m = info.parameters / 1e6
    hw_label = hw.accelerator.kind.upper() if hw.accelerator.kind != "none" else "CPU"

    if not selected:
        return f"fingerprint: {family_label} · {params_m:.1f}M params · {hw_label} → no candidates"

    top = selected[0].backend
    n = len(selected)
    return (
        f"fingerprint: {family_label} · {params_m:.1f}M params · {hw_label}"
        f" → predicting [bold]{top}[/bold] wins; benchmarking {n} of {total} candidates"
    )
