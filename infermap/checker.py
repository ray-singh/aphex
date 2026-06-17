"""Regression checking — compares a benchmark run against a deployment.yaml baseline."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CheckResult:
    metric: str            # machine-readable key, e.g. "latency_p50_ms"
    label: str             # display label, e.g. "latency p50"
    baseline: float
    observed: float
    delta_pct: float       # (observed - baseline) / |baseline| * 100; positive = increase
    threshold_pct: float | None  # allowed delta in %; None = informational only
    passed: bool


def _pct_delta(baseline: float, observed: float) -> float:
    if baseline == 0:
        return 0.0
    return (observed - baseline) / abs(baseline) * 100.0


def _check(
    metric: str,
    label: str,
    baseline: float,
    observed: float,
    threshold_pct: float | None,
    lower_is_better: bool,
) -> CheckResult:
    delta = _pct_delta(baseline, observed)
    if threshold_pct is None or baseline == 0:
        passed = True
    elif lower_is_better:
        passed = delta <= threshold_pct
    else:
        passed = delta >= -threshold_pct
    return CheckResult(
        metric=metric,
        label=label,
        baseline=baseline,
        observed=observed,
        delta_pct=delta,
        threshold_pct=threshold_pct,
        passed=passed,
    )


def run_check(
    baseline_perf: dict,
    observed_p50_ms: float,
    observed_p95_ms: float,
    observed_throughput_rps: float,
    observed_memory_mb: float,
    max_latency_delta_pct: float | None = 10.0,
    min_throughput_delta_pct: float | None = None,
    max_memory_delta_pct: float | None = None,
    baseline_eval_score: float | None = None,
    observed_eval_score: float | None = None,
    max_accuracy_drop: float | None = None,
) -> list[CheckResult]:
    """Compare observed metrics against the baseline from a deployment.yaml.

    Returns one CheckResult per metric. Results with threshold_pct=None are
    informational (always pass but still shown in the output).
    """
    b_p50 = float(baseline_perf.get("latency_p50_ms") or 0)
    b_p95 = float(baseline_perf.get("latency_p95_ms") or 0)
    b_tput = float(baseline_perf.get("throughput_rps") or 0)
    b_mem = float(baseline_perf.get("memory_mb") or 0)

    results = [
        _check("latency_p50_ms", "latency p50",
               b_p50, observed_p50_ms, max_latency_delta_pct, lower_is_better=True),
        _check("latency_p95_ms", "latency p95",
               b_p95, observed_p95_ms, max_latency_delta_pct, lower_is_better=True),
        _check("throughput_rps", "throughput",
               b_tput, observed_throughput_rps, min_throughput_delta_pct, lower_is_better=False),
        _check("memory_mb", "memory",
               b_mem, observed_memory_mb, max_memory_delta_pct, lower_is_better=True),
    ]

    if baseline_eval_score is not None and observed_eval_score is not None:
        drop = baseline_eval_score - observed_eval_score
        if max_accuracy_drop is not None:
            passed = drop <= max_accuracy_drop
            threshold_pct = (
                max_accuracy_drop / abs(baseline_eval_score) * 100
                if baseline_eval_score != 0 else None
            )
        else:
            passed = True
            threshold_pct = None
        results.append(CheckResult(
            metric="eval_score",
            label="accuracy",
            baseline=baseline_eval_score,
            observed=observed_eval_score,
            delta_pct=_pct_delta(baseline_eval_score, observed_eval_score),
            threshold_pct=threshold_pct,
            passed=passed,
        ))

    return results
