"""Pareto frontier builder — identifies non-dominated deployment candidates."""

from __future__ import annotations

from dataclasses import dataclass

from aphex.benchmark import BenchmarkResult


@dataclass
class ParetoPoint:
    result: BenchmarkResult
    dominated: bool = False


def build_pareto_frontier(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    """
    Return the subset of results that form the Pareto frontier over
    (minimize latency_p50_ms, minimize memory_mb, minimize accuracy_drop).

    accuracy_drop is included as a 3rd axis only when both candidates in a
    comparison have it measured; otherwise the pair falls back to 2D comparison.
    A result is dominated if another result is at least as good on all axes
    and strictly better on at least one.
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
    axes: list[tuple[float, float]] = [
        (a.latency_p50_ms, b.latency_p50_ms),
        (a.memory_mb, b.memory_mb),
    ]
    if a.accuracy_drop is not None and b.accuracy_drop is not None:
        axes.append((a.accuracy_drop, b.accuracy_drop))
    at_least_as_good = all(av <= bv for av, bv in axes)
    strictly_better = any(av < bv for av, bv in axes)
    return at_least_as_good and strictly_better


def accuracy_in_frontier(results: list[BenchmarkResult]) -> bool:
    """Return True if any successful result has an accuracy measurement."""
    return any(r.ok and r.accuracy_drop is not None for r in results)


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
