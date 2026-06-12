"""
examples/optimize_tiny_mlp.py
─────────────────────────────
End-to-end demo: build a small MLP, benchmark every deployment candidate
on the current hardware, and print the optimal strategy.

Run:
    python examples/optimize_tiny_mlp.py

No CLI flags needed — the script sets everything up itself.
"""

from __future__ import annotations
import sys
import tempfile
from pathlib import Path
import torch
import torch.nn as nn


# ── 1. Define and save a model ────────────────────────────────────────────────

class TinyMLP(nn.Module):
    """3-layer MLP: 128 → 256 → 256 → 64."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_model(path: Path) -> None:
    model = TinyMLP()
    scripted = torch.jit.script(model)
    scripted.save(str(path))
    print(f"[aphex demo]  model saved → {path}")


# ── 2. Run the full aphex optimize pipeline programmatically ──────────────────

def run_optimize(model_path: Path, input_shape: list[int]) -> None:
    import torch
    from infermap.inspector import inspect_model
    from infermap.profiler import profile_hardware
    from infermap.preflight import run_preflight
    from infermap.candidates import generate_candidates
    from infermap.benchmark import benchmark_candidate
    from infermap.recommender import recommend

    print("\n[aphex demo]  profiling hardware …")
    hw = profile_hardware()
    print(f"             cpu   : {hw.cpu.name}")
    print(f"             accel : {hw.accelerator.kind.upper()}")

    print("\n[aphex demo]  inspecting model …")
    info = inspect_model(model_path, input_shape=input_shape)
    print(f"             params : {info.parameters:,}")
    print(f"             fp32   : {info.estimated_memory_fp32_gb:.3f} GB")

    pf = run_preflight(info, hw)
    print(f"\n[aphex demo]  preflight → {pf.category}  ({pf.message})")
    if pf.category == "impossible":
        print("             aborting — model won't fit in available memory.")
        sys.exit(1)

    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()

    candidates = generate_candidates(info, hw)
    print(f"\n[aphex demo]  benchmarking {len(candidates)} candidate(s) …\n")

    results = []
    for cand in candidates:
        r = benchmark_candidate(
            cand, model, info,
            input_shape=input_shape,
            batch_size=1,
            warmup_iters=10,
            measure_iters=50,
            timeout_s=60.0,
        )
        status = "✓" if r.ok else "✗"
        if r.ok:
            print(f"  {status}  {cand.description:<44}  {r.latency_p50_ms:>7.2f} ms  "
                  f"{r.throughput_rps:>6.0f} req/s")
        else:
            print(f"  {status}  {cand.description:<44}  FAILED: {r.error}")
        results.append(r)

    rec = recommend(results, objective="latency")
    r = rec.result
    print(f"\n[aphex demo]  ── recommendation ──────────────────────────────")
    print(f"             backend : {r.candidate.description}")
    print(f"             p50     : {r.latency_p50_ms:.2f} ms")
    print(f"             p95     : {r.latency_p95_ms:.2f} ms")
    print(f"             req/s   : {r.throughput_rps:.0f}")
    print(f"             memory  : {r.memory_mb:.0f} MB")
    print(f"             why     : {rec.rationale}")
    print()


# ── 3. Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "tiny_mlp.pt"
        save_model(model_path)
        run_optimize(model_path, input_shape=[128])
