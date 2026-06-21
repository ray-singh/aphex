"""Formats EvalResult(s) into a summary table and JSON file."""

from __future__ import annotations

import json
from pathlib import Path

from eval.runner import EvalResult


def _fmt_pct(v: float | None, invert: bool = False) -> str:
    if v is None:
        return "n/a"
    pct = v * 100
    if invert:
        pct = -pct
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def _fmt_x(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2f}×"


def print_summary(results: list[EvalResult]) -> None:
    """Print the four-metric summary table to stdout."""
    col_widths = [16, 22, 12, 10, 10, 16, 14, 10]
    headers = [
        "Model", "Hardware", "Generated", "Kept",
        "Speedup", "Search reduction", "Mem reduction", "Acc drop",
    ]

    sep = "  ".join("-" * w for w in col_widths)
    header_row = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=False))
    print()
    print("APHEX Evaluation Suite")
    print("=" * len(sep))
    print(header_row)
    print(sep)

    for r in results:
        hw = r.hardware.accelerator
        hw_label = f"{hw.name} ({hw.kind})" if hw.kind != "none" else "CPU"
        row = [
            r.case.name[:col_widths[0]],
            hw_label[:col_widths[1]],
            str(r.n_generated),
            str(r.n_after_cost_model),
            _fmt_x(r.speedup),
            _fmt_pct(r.search_reduction_pct),
            _fmt_pct(r.memory_reduction_pct),
            _fmt_pct(r.accuracy_delta, invert=True) if r.accuracy_delta is not None else "n/a",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(row, col_widths, strict=False)))

    print(sep)
    print()
    _print_detail(results)


def _print_detail(results: list[EvalResult]) -> None:
    for r in results:
        rec = r.recommendation.result
        print(f"  {r.case.name}")
        print(f"    winner   : {rec.candidate.backend}  bs={rec.batch_size}")
        print(f"    latency  : {rec.latency_p50_ms:.2f} ms p50  "
              f"(baseline {r.baseline.latency_p50_ms:.2f} ms)" if r.baseline else
              f"    latency  : {rec.latency_p50_ms:.2f} ms p50")
        print(f"    throughput: {rec.throughput_rps:.0f} req/s")
        print(f"    memory   : {rec.memory_mb:.1f} MB")
        if r.errors:
            for e in r.errors:
                print(f"    [warn] {e}")
        print(f"    elapsed  : {r.elapsed_s:.1f}s")
        print()


def write_json(results: list[EvalResult], path: Path) -> None:
    """Write full results to a JSON file for downstream analysis."""
    payload = []
    for r in results:
        rec = r.recommendation.result
        entry: dict = {
            "model": r.case.name,
            "hardware": {
                "kind": r.hardware.accelerator.kind,
                "name": r.hardware.accelerator.name,
                "memory_gb": r.hardware.accelerator.memory_gb,
            },
            "candidates": {
                "generated": r.n_generated,
                "after_cost_model": r.n_after_cost_model,
                "search_reduction_pct": round(r.search_reduction_pct * 100, 1),
            },
            "recommendation": {
                "backend": rec.candidate.backend,
                "dtype": rec.candidate.dtype,
                "batch_size": rec.batch_size,
                "latency_p50_ms": rec.latency_p50_ms,
                "latency_p95_ms": rec.latency_p95_ms,
                "throughput_rps": rec.throughput_rps,
                "memory_mb": rec.memory_mb,
                "accuracy_drop": rec.accuracy_drop,
            },
            "baseline": {
                "backend": r.baseline.candidate.backend,
                "latency_p50_ms": r.baseline.latency_p50_ms,
                "memory_mb": r.baseline.memory_mb,
            } if r.baseline else None,
            "metrics": {
                "speedup": round(r.speedup, 3) if r.speedup else None,
                "memory_reduction_pct": round((r.memory_reduction_pct or 0) * 100, 1),
                "accuracy_delta": rec.accuracy_drop,
            },
            "elapsed_s": round(r.elapsed_s, 1),
            "errors": r.errors,
            "all_results": [
                {
                    "backend": res.candidate.backend,
                    "batch_size": res.batch_size,
                    "latency_p50_ms": res.latency_p50_ms,
                    "throughput_rps": res.throughput_rps,
                    "memory_mb": res.memory_mb,
                    "accuracy_drop": res.accuracy_drop,
                    "ok": res.ok,
                    "error": res.error,
                }
                for res in r.all_results
            ],
        }
        payload.append(entry)

    path.write_text(json.dumps(payload, indent=2))
    print(f"results written → {path}")
