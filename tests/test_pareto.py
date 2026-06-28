"""Tests for Pareto frontier builder."""


from aphex.benchmark import BenchmarkResult
from aphex.candidates import DeploymentCandidate
from aphex.pareto import accuracy_in_frontier, build_pareto_frontier, rank_by_objective


def _result(
    backend: str,
    latency: float,
    memory: float,
    error: str | None = None,
    accuracy_drop: float | None = None,
) -> BenchmarkResult:
    cand = DeploymentCandidate(
        backend="pytorch_fp32",  # type: ignore[arg-type]
        dtype="fp32",
        description=backend,
        requires_export=False,
        device="cpu",
    )
    return BenchmarkResult(
        candidate=cand,
        latency_p50_ms=latency,
        latency_p95_ms=latency * 1.1,
        latency_p99_ms=latency * 1.2,
        throughput_rps=1000.0 / latency,
        memory_mb=memory,
        error=error,
        accuracy_drop=accuracy_drop,
    )


def test_pareto_basic() -> None:
    # A dominates C (lower latency AND lower memory)
    # B is non-dominated (better latency than C but worse memory than A)
    a = _result("A", latency=2.0, memory=100.0)
    b = _result("B", latency=3.0, memory=50.0)
    c = _result("C", latency=5.0, memory=200.0)

    frontier = build_pareto_frontier([a, b, c])
    desc = {r.candidate.description for r in frontier}
    assert "A" in desc
    assert "B" in desc
    assert "C" not in desc


def test_pareto_excludes_errors() -> None:
    good = _result("good", latency=5.0, memory=100.0)
    bad = _result("bad", latency=1.0, memory=10.0, error="CUDA OOM")
    frontier = build_pareto_frontier([good, bad])
    assert all(r.ok for r in frontier)


def test_pareto_single_result() -> None:
    a = _result("A", latency=10.0, memory=200.0)
    frontier = build_pareto_frontier([a])
    assert len(frontier) == 1


def test_rank_by_latency() -> None:
    results = [
        _result("A", latency=10.0, memory=100.0),
        _result("B", latency=2.0, memory=300.0),
        _result("C", latency=5.0, memory=200.0),
    ]
    ranked = rank_by_objective(results, "latency")
    assert ranked[0].candidate.description == "B"


def test_rank_by_throughput() -> None:
    results = [
        _result("slow", latency=100.0, memory=100.0),
        _result("fast", latency=5.0, memory=100.0),
    ]
    ranked = rank_by_objective(results, "throughput")
    assert ranked[0].candidate.description == "fast"


# ── 3-axis accuracy tests ─────────────────────────────────────────────────────


def test_3d_accuracy_dominance() -> None:
    # A and B identical latency/memory; A has less accuracy degradation — A dominates B.
    a = _result("A", latency=5.0, memory=100.0, accuracy_drop=0.01)
    b = _result("B", latency=5.0, memory=100.0, accuracy_drop=0.05)
    frontier = build_pareto_frontier([a, b])
    descs = {r.candidate.description for r in frontier}
    assert "A" in descs
    assert "B" not in descs


def test_3d_accuracy_tradeoff_both_non_dominated() -> None:
    # A is faster but less accurate; B is slower but more accurate — tradeoff, neither dominates.
    a = _result("A", latency=3.0, memory=100.0, accuracy_drop=0.05)
    b = _result("B", latency=5.0, memory=100.0, accuracy_drop=0.01)
    frontier = build_pareto_frontier([a, b])
    descs = {r.candidate.description for r in frontier}
    assert "A" in descs
    assert "B" in descs


def test_2d_fallback_when_accuracy_missing() -> None:
    # Neither candidate has accuracy — falls back to 2D; A better latency, B better memory.
    a = _result("A", latency=3.0, memory=200.0)
    b = _result("B", latency=5.0, memory=50.0)
    frontier = build_pareto_frontier([a, b])
    assert len(frontier) == 2


def test_mixed_accuracy_no_dominance() -> None:
    # A has accuracy measured, B does not — can't compare on accuracy, no dominance.
    a = _result("A", latency=5.0, memory=100.0, accuracy_drop=0.01)
    b = _result("B", latency=5.0, memory=100.0)
    frontier = build_pareto_frontier([a, b])
    assert len(frontier) == 2


def test_accuracy_in_frontier_helper() -> None:
    assert accuracy_in_frontier([_result("A", 5.0, 100.0, accuracy_drop=0.01)]) is True
    assert accuracy_in_frontier([_result("A", 5.0, 100.0)]) is False
    assert accuracy_in_frontier([_result("A", 5.0, 100.0, error="OOM", accuracy_drop=0.01)]) is False
