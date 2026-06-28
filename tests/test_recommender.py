"""Tests for the recommendation engine."""


from aphex.benchmark import BenchmarkResult
from aphex.candidates import DeploymentCandidate
from aphex.recommender import recommend


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


# ── measurement-variance reporting ───────────────────────────────────────────


def _result_with_std(description: str, latency: float, std: float) -> BenchmarkResult:
    return BenchmarkResult(
        candidate=_cand(description),
        latency_p50_ms=latency,
        latency_p95_ms=latency * 1.1,
        latency_p99_ms=latency * 1.2,
        throughput_rps=(1000.0 / latency) if latency > 0 else 0.0,
        memory_mb=100.0,
        latency_std_ms=std,
    )


def test_stability_property_buckets() -> None:
    assert _result_with_std("a", 10.0, 0.1).stability == "stable"    # cv 1%
    assert _result_with_std("b", 10.0, 1.0).stability == "noisy"     # cv 10%
    assert _result_with_std("c", 10.0, 3.0).stability == "unstable"  # cv 30%


def test_latency_cv_zero_when_no_latency() -> None:
    assert _result_with_std("x", 0.0, 5.0).latency_cv == 0.0


def test_tie_note_when_top2_within_noise() -> None:
    # 10.0 vs 10.2 ms, but each jitters ±0.5 ms → indistinguishable.
    results = [
        _result_with_std("A", 10.0, 0.5),
        _result_with_std("B", 10.2, 0.5),
    ]
    rec = recommend(results, objective="latency")
    assert rec.notes
    assert any("not statistically significant" in n for n in rec.notes)


def test_no_tie_note_when_clearly_separated() -> None:
    results = [
        _result_with_std("fast", 2.0, 0.05),
        _result_with_std("slow", 20.0, 0.05),
    ]
    rec = recommend(results, objective="latency")
    assert not any("statistically significant" in n for n in (rec.notes or []))


def test_noisy_winner_gets_rerun_note() -> None:
    results = [_result_with_std("only", 10.0, 3.0)]  # cv 30% → unstable
    rec = recommend(results, objective="latency")
    assert any("re-run" in n for n in (rec.notes or []))


# ── 3-axis accuracy Pareto ────────────────────────────────────────────────────


def _result_with_acc(description: str, latency: float, memory: float, accuracy_drop: float | None) -> BenchmarkResult:
    r = _result(description, latency, memory)
    r.accuracy_drop = accuracy_drop
    return r


def test_3axis_rationale_when_accuracy_present() -> None:
    results = [
        _result_with_acc("fp32", latency=5.0, memory=100.0, accuracy_drop=0.0),
        _result_with_acc("int8", latency=3.0, memory=80.0, accuracy_drop=0.02),
    ]
    rec = recommend(results, objective="latency")
    assert "3-axis" in rec.rationale


def test_partial_accuracy_note_fires() -> None:
    # One result has accuracy measured, one does not.
    results = [
        _result_with_acc("fp32", latency=5.0, memory=100.0, accuracy_drop=None),
        _result_with_acc("int8", latency=3.0, memory=80.0, accuracy_drop=0.02),
    ]
    rec = recommend(results, objective="latency")
    assert any("1/2" in n for n in (rec.notes or []))
