"""Tests for infermap.checker — regression check logic."""
from __future__ import annotations

from infermap.checker import _check, _pct_delta, run_check

# ── Baseline fixture ──────────────────────────────────────────────────────────

_PERF = {
    "latency_p50_ms": 10.0,
    "latency_p95_ms": 12.0,
    "throughput_rps": 100.0,
    "memory_mb": 500.0,
}


def _results(
    p50=10.0, p95=12.0, tput=100.0, mem=500.0,
    max_lat=10.0, min_tput=None, max_mem=None,
    baseline_eval=None, obs_eval=None, max_acc_drop=None,
):
    return run_check(
        _PERF, p50, p95, tput, mem,
        max_latency_delta_pct=max_lat,
        min_throughput_delta_pct=min_tput,
        max_memory_delta_pct=max_mem,
        baseline_eval_score=baseline_eval,
        observed_eval_score=obs_eval,
        max_accuracy_drop=max_acc_drop,
    )


def _get(results, metric):
    return next(r for r in results if r.metric == metric)


# ── _pct_delta ────────────────────────────────────────────────────────────────


def test_pct_delta_increase() -> None:
    assert abs(_pct_delta(10.0, 11.0) - 10.0) < 1e-9


def test_pct_delta_decrease() -> None:
    assert abs(_pct_delta(10.0, 9.0) - (-10.0)) < 1e-9


def test_pct_delta_no_change() -> None:
    assert _pct_delta(10.0, 10.0) == 0.0


def test_pct_delta_zero_baseline() -> None:
    assert _pct_delta(0.0, 5.0) == 0.0


def test_pct_delta_uses_abs_baseline() -> None:
    # negative baseline shouldn't flip the sign of a positive delta
    assert _pct_delta(-10.0, -9.0) > 0


# ── _check ────────────────────────────────────────────────────────────────────


def test_check_lower_is_better_passes_within_threshold() -> None:
    r = _check("lat", "latency", 10.0, 10.5, threshold_pct=10.0, lower_is_better=True)
    assert r.passed


def test_check_lower_is_better_fails_above_threshold() -> None:
    r = _check("lat", "latency", 10.0, 12.0, threshold_pct=10.0, lower_is_better=True)
    assert not r.passed


def test_check_lower_is_better_passes_decrease() -> None:
    r = _check("lat", "latency", 10.0, 8.0, threshold_pct=10.0, lower_is_better=True)
    assert r.passed


def test_check_higher_is_better_passes_increase() -> None:
    r = _check("tput", "throughput", 100.0, 110.0, threshold_pct=10.0, lower_is_better=False)
    assert r.passed


def test_check_higher_is_better_fails_large_drop() -> None:
    r = _check("tput", "throughput", 100.0, 85.0, threshold_pct=10.0, lower_is_better=False)
    assert not r.passed


def test_check_higher_is_better_passes_small_drop() -> None:
    r = _check("tput", "throughput", 100.0, 95.0, threshold_pct=10.0, lower_is_better=False)
    assert r.passed


def test_check_no_threshold_always_passes() -> None:
    r = _check("mem", "memory", 100.0, 999.0, threshold_pct=None, lower_is_better=True)
    assert r.passed
    assert r.threshold_pct is None


def test_check_zero_baseline_always_passes() -> None:
    r = _check("lat", "latency", 0.0, 100.0, threshold_pct=5.0, lower_is_better=True)
    assert r.passed


def test_check_delta_pct_stored_correctly() -> None:
    r = _check("lat", "latency", 10.0, 11.0, threshold_pct=10.0, lower_is_better=True)
    assert abs(r.delta_pct - 10.0) < 1e-9


def test_check_stores_baseline_and_observed() -> None:
    r = _check("lat", "latency", 10.0, 11.0, threshold_pct=10.0, lower_is_better=True)
    assert r.baseline == 10.0
    assert r.observed == 11.0


# ── run_check: latency ────────────────────────────────────────────────────────


def test_latency_passes_within_threshold() -> None:
    assert _get(_results(p50=10.5, p95=12.3), "latency_p50_ms").passed


def test_latency_fails_above_threshold() -> None:
    assert not _get(_results(p50=15.0), "latency_p50_ms").passed


def test_latency_p95_checked_independently() -> None:
    r = _results(p50=10.0, p95=20.0)  # p95 explodes, p50 fine
    assert _get(r, "latency_p50_ms").passed
    assert not _get(r, "latency_p95_ms").passed


def test_latency_threshold_none_always_passes() -> None:
    r = _results(p50=999.0, max_lat=None)
    assert _get(r, "latency_p50_ms").passed


def test_latency_exactly_at_threshold_passes() -> None:
    r = _results(p50=11.0)  # exactly +10%
    assert _get(r, "latency_p50_ms").passed


# ── run_check: throughput ─────────────────────────────────────────────────────


def test_throughput_unchecked_by_default() -> None:
    r = _results(tput=1.0)  # massive drop but no threshold
    assert _get(r, "throughput_rps").passed
    assert _get(r, "throughput_rps").threshold_pct is None


def test_throughput_passes_within_threshold() -> None:
    r = _results(tput=95.0, min_tput=10.0)
    assert _get(r, "throughput_rps").passed


def test_throughput_fails_large_drop() -> None:
    r = _results(tput=80.0, min_tput=10.0)
    assert not _get(r, "throughput_rps").passed


def test_throughput_passes_increase() -> None:
    r = _results(tput=120.0, min_tput=10.0)
    assert _get(r, "throughput_rps").passed


# ── run_check: memory ─────────────────────────────────────────────────────────


def test_memory_unchecked_by_default() -> None:
    r = _results(mem=9999.0)
    assert _get(r, "memory_mb").passed


def test_memory_passes_within_threshold() -> None:
    r = _results(mem=510.0, max_mem=5.0)
    assert _get(r, "memory_mb").passed


def test_memory_fails_above_threshold() -> None:
    r = _results(mem=600.0, max_mem=5.0)
    assert not _get(r, "memory_mb").passed


# ── run_check: always returns 4 base results ──────────────────────────────────


def test_always_returns_four_base_results() -> None:
    r = _results()
    assert len(r) == 4


def test_accuracy_result_appended_when_scores_provided() -> None:
    r = _results(baseline_eval=0.95, obs_eval=0.93)
    assert len(r) == 5
    assert _get(r, "eval_score").metric == "eval_score"


# ── run_check: accuracy ───────────────────────────────────────────────────────


def test_accuracy_informational_when_no_threshold() -> None:
    r = _results(baseline_eval=0.95, obs_eval=0.90)
    acc = _get(r, "eval_score")
    assert acc.passed
    assert acc.threshold_pct is None


def test_accuracy_passes_within_drop_threshold() -> None:
    r = _results(baseline_eval=0.95, obs_eval=0.94, max_acc_drop=0.02)
    assert _get(r, "eval_score").passed


def test_accuracy_fails_exceeding_drop_threshold() -> None:
    r = _results(baseline_eval=0.95, obs_eval=0.92, max_acc_drop=0.02)
    assert not _get(r, "eval_score").passed


def test_accuracy_passes_when_score_improves() -> None:
    r = _results(baseline_eval=0.95, obs_eval=0.97, max_acc_drop=0.02)
    assert _get(r, "eval_score").passed


def test_accuracy_not_added_when_scores_absent() -> None:
    r = _results()
    assert all(r_.metric != "eval_score" for r_ in r)


def test_accuracy_delta_pct_negative_on_drop() -> None:
    r = _results(baseline_eval=0.95, obs_eval=0.90, max_acc_drop=0.10)
    assert _get(r, "eval_score").delta_pct < 0


# ── run_check: missing / partial baseline values ──────────────────────────────


def test_missing_baseline_value_treated_as_zero() -> None:
    perf = {}  # no keys at all
    results = run_check(perf, 5.0, 6.0, 50.0, 200.0)
    # zero baseline → always pass (no div by zero)
    assert all(r.passed for r in results)


def test_null_baseline_value_treated_as_zero() -> None:
    perf = {"latency_p50_ms": None, "latency_p95_ms": None,
            "throughput_rps": None, "memory_mb": None}
    results = run_check(perf, 5.0, 6.0, 50.0, 200.0)
    assert all(r.passed for r in results)
