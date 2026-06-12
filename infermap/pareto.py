"""Pareto frontier builder — identifies non-dominated deployment candidates."""

from __future__ import annotations

from dataclasses import dataclass

from infermap.benchmark import BenchmarkResult


@dataclass
class ParetoPoint:
    result: BenchmarkResult
    dominated: bool = False


def build_pareto_frontier(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    """
    Return the subset of results that form the Pareto frontier over
    (minimize latency_p50_ms, minimize memory_mb).

    A result is dominated if another result is strictly better on ALL objectives.
    """
    ok_results = [r for r in results if r.ok]
    if not ok_results:
        return []

    points = [ParetoPoint(r) for r in ok_results]

    for i, pi in enumerate(points):
        for j, pj in enumerate(points):
            if i == j:
                continue
            if _dominates(pj.result, pi.result):
                pi.dominated = True
                break

    return [p.result for p in points if not p.dominated]


def _dominates(a: BenchmarkResult, b: BenchmarkResult) -> bool:
    """Return True if result `a` dominates result `b` (a is at least as good on all and better on one)."""
    a_lat = a.latency_p50_ms
    b_lat = b.latency_p50_ms
    a_mem = a.memory_mb
    b_mem = b.memory_mb

    at_least_as_good = a_lat <= b_lat and a_mem <= b_mem
    strictly_better = a_lat < b_lat or a_mem < b_mem
    return at_least_as_good and strictly_better


def rank_by_objective(
    results: list[BenchmarkResult],
    objective: str = "latency",
) -> list[BenchmarkResult]:
    """Sort results by a single objective."""
    if objective == "latency":
        return sorted(results, key=lambda r: r.latency_p50_ms)
    if objective == "throughput":
        return sorted(results, key=lambda r: r.throughput_rps, reverse=True)
    if objective == "memory":
        return sorted(results, key=lambda r: r.memory_mb)
    raise ValueError(f"Unknown objective: {objective!r}. Choose latency, throughput, or memory.")
