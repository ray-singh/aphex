"""Tests for the recommendation engine."""

import pytest

from infermap.benchmark import BenchmarkResult
from infermap.candidates import DeploymentCandidate
from infermap.recommender import Recommendation, recommend


def _cand(description: str) -> DeploymentCandidate:
    return DeploymentCandidate(
        backend="pytorch_fp32",
        dtype="fp32",
        description=description,
        requires_export=False,
        device="cpu",
    )


def _result(
    description: str,
    latency: float,
    memory: float,
    throughput: float | None = None,
    error: str | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        candidate=_cand(description),
        latency_p50_ms=latency,
        latency_p95_ms=latency * 1.1,
        latency_p99_ms=latency * 1.2,
        throughput_rps=throughput if throughput is not None else 1000.0 / latency,
        memory_mb=memory,
        error=error,
    )


def test_recommend_by_latency_picks_fastest() -> None:
    results = [
        _result("slow", latency=20.0, memory=100.0),
        _result("fast", latency=2.0, memory=100.0),
        _result("medium", latency=10.0, memory=100.0),
    ]
    rec = recommend(results, objective="latency")
    assert rec.result.candidate.description == "fast"


def test_recommend_by_throughput_picks_highest() -> None:
    results = [
        _result("high-tput", latency=5.0, memory=100.0, throughput=500.0),
        _result("low-tput", latency=5.0, memory=100.0, throughput=50.0),
    ]
    rec = recommend(results, objective="throughput")
    assert rec.result.candidate.description == "high-tput"


def test_recommend_by_memory_picks_leanest() -> None:
    results = [
        _result("lean", latency=5.0, memory=50.0),
        _result("heavy", latency=5.0, memory=500.0),
    ]
    rec = recommend(results, objective="memory")
    assert rec.result.candidate.description == "lean"


def test_max_latency_constraint_excludes_slow() -> None:
    results = [
        _result("fast", latency=2.0, memory=100.0),
        _result("slow", latency=50.0, memory=100.0),
    ]
    rec = recommend(results, objective="latency", max_latency_ms=5.0)
    assert rec.result.candidate.description == "fast"


def test_max_memory_constraint_excludes_heavy() -> None:
    results = [
        _result("fast-heavy", latency=2.0, memory=1000.0),
        _result("slow-lean", latency=10.0, memory=100.0),
    ]
    rec = recommend(results, objective="latency", max_memory_mb=200.0)
    assert rec.result.candidate.description == "slow-lean"


def test_min_throughput_constraint_excludes_low_tput() -> None:
    results = [
        _result("fast-low-tput", latency=2.0, memory=100.0, throughput=10.0),
        _result("slower-high-tput", latency=10.0, memory=100.0, throughput=500.0),
    ]
    rec = recommend(results, objective="latency", min_throughput_rps=100.0)
    assert rec.result.candidate.description == "slower-high-tput"


def test_impossible_constraints_fall_back_to_all_passing() -> None:
    results = [_result("only-option", latency=20.0, memory=500.0)]
    # Constraint that nothing can satisfy
    rec = recommend(results, objective="latency", max_latency_ms=0.001)
    # Falls back to all passing results instead of crashing
    assert rec.result.candidate.description == "only-option"


def test_failed_results_are_excluded() -> None:
    results = [
        _result("passing", latency=5.0, memory=100.0),
        _result("failing", latency=1.0, memory=10.0, error="CUDA OOM"),
    ]
    rec = recommend(results, objective="latency")
    assert rec.result.ok
    assert rec.result.candidate.description == "passing"


def test_pareto_frontier_excludes_dominated_results() -> None:
    results = [
        _result("A", latency=2.0, memory=100.0),   # non-dominated
        _result("B", latency=3.0, memory=50.0),    # non-dominated
        _result("C", latency=10.0, memory=200.0),  # dominated by A
    ]
    rec = recommend(results, objective="latency")
    frontier_names = {r.candidate.description for r in rec.pareto_frontier}
    assert "C" not in frontier_names
    assert len(rec.pareto_frontier) == 2


def test_all_results_attached_to_recommendation() -> None:
    results = [
        _result("A", latency=2.0, memory=100.0),
        _result("B", latency=10.0, memory=50.0),
    ]
    rec = recommend(results, objective="latency")
    assert len(rec.all_results) == 2


def test_rationale_is_non_empty() -> None:
    results = [_result("only", latency=5.0, memory=100.0)]
    rec = recommend(results)
    assert len(rec.rationale) > 0


def test_rationale_mentions_candidate_name() -> None:
    results = [_result("my-backend", latency=5.0, memory=100.0)]
    rec = recommend(results)
    assert "my-backend" in rec.rationale


def test_rationale_includes_constraints_when_given() -> None:
    results = [_result("fast", latency=2.0, memory=100.0)]
    rec = recommend(results, objective="latency", max_latency_ms=10.0, max_memory_mb=200.0)
    assert "10.0" in rec.rationale or "200" in rec.rationale
