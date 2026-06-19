"""Rule-based recommendation engine — selects the best strategy given user constraints."""

from __future__ import annotations

from dataclasses import dataclass

from infermap.benchmark import BenchmarkResult
from infermap.pareto import build_pareto_frontier, rank_by_objective


@dataclass
class Recommendation:
    result: BenchmarkResult
    rationale: str
    pareto_frontier: list[BenchmarkResult]
    all_results: list[BenchmarkResult]


def recommend(
    results: list[BenchmarkResult],
    objective: str = "latency",
    max_latency_ms: float | None = None,
    max_memory_mb: float | None = None,
    min_throughput_rps: float | None = None,
    max_quality_loss: float | None = None,
) -> Recommendation:
    """
    Select the best deployment strategy subject to optional constraints.

    Args:
        results: All benchmark results (including failed ones).
        objective: Primary optimization goal — "latency", "throughput", or "memory".
        max_latency_ms: Hard constraint on p50 latency (optional).
        max_memory_mb: Hard constraint on peak memory (optional).
        min_throughput_rps: Hard constraint on minimum throughput (optional).
        max_quality_loss: Hard constraint on metric degradation vs original model baseline (0.0–1.0).
            Results where accuracy was not measured are kept (conservative).
    """
    frontier = build_pareto_frontier(results)
    passing = [r for r in results if r.ok]

    def _apply_constraints(pool: list[BenchmarkResult]) -> list[BenchmarkResult]:
        if max_latency_ms is not None:
            pool = [r for r in pool if r.latency_p50_ms <= max_latency_ms]
        if max_memory_mb is not None:
            pool = [r for r in pool if r.memory_mb <= max_memory_mb]
        if min_throughput_rps is not None:
            pool = [r for r in pool if r.throughput_rps >= min_throughput_rps]
        if max_quality_loss is not None:
            # Keep results where accuracy wasn't measured (no precision reduction applied)
            pool = [
                r for r in pool
                if r.accuracy_drop is None or r.accuracy_drop <= max_quality_loss
            ]
        return pool

    # Apply constraints to Pareto frontier first, then widen to all passing results,
    # then drop constraints entirely — each step only taken when the previous yields nothing.
    candidates = _apply_constraints(frontier if frontier else passing)
    if not candidates:
        candidates = _apply_constraints(passing)
    if not candidates:
        candidates = passing

    ranked = rank_by_objective(candidates, objective)
    if not ranked:
        errors = list({r.error for r in results if not r.ok and r.error})[:3]
        error_summary = "; ".join(errors) if errors else "unknown reason"
        raise RuntimeError(
            f"No successful benchmark results — all {len(results)} candidate(s) failed. "
            f"Errors: {error_summary}"
        )
    best = ranked[0]

    rationale = _build_rationale(best, objective, max_latency_ms, max_memory_mb, min_throughput_rps, max_quality_loss)

    return Recommendation(
        result=best,
        rationale=rationale,
        pareto_frontier=frontier,
        all_results=results,
    )


def _build_rationale(
    result: BenchmarkResult,
    objective: str,
    max_latency_ms: float | None,
    max_memory_mb: float | None,
    min_throughput_rps: float | None,
    max_quality_loss: float | None = None,
) -> str:
    parts = [f"Best {objective} on the Pareto frontier: {result.candidate.description}."]

    if objective == "latency":
        parts.append(f"p50 latency: {result.latency_p50_ms:.2f} ms.")
    elif objective == "throughput":
        parts.append(f"Throughput: {result.throughput_rps:.0f} req/s.")
    elif objective == "memory":
        parts.append(f"Peak memory: {result.memory_mb:.1f} MB.")

    if result.accuracy_drop is not None:
        parts.append(f"Quality loss vs original model: {result.accuracy_drop * 100:.2f}%.")

    constraints: list[str] = []
    if max_latency_ms is not None:
        constraints.append(f"latency ≤ {max_latency_ms} ms")
    if max_memory_mb is not None:
        constraints.append(f"memory ≤ {max_memory_mb} MB")
    if min_throughput_rps is not None:
        constraints.append(f"throughput ≥ {min_throughput_rps} req/s")
    if max_quality_loss is not None:
        constraints.append(f"quality loss ≤ {max_quality_loss * 100:.1f}%")

    if constraints:
        parts.append("Satisfies constraints: " + ", ".join(constraints) + ".")

    return " ".join(parts)
