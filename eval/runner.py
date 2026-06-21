"""Runs a BenchmarkCase through the aphex pipeline and returns an EvalResult.

Calls aphex internals directly (no subprocess) so intermediate counts — number
of candidates generated, number kept after cost-model pruning — are available
for the metrics report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from infermap.benchmark import BenchmarkResult
from infermap.candidates import DeploymentCandidate
from infermap.cost_model import CostEstimate, prune_candidates
from infermap.evaluator import EvalDataset, fill_accuracy_drop
from infermap.profiler import HardwareProfile, profile_hardware
from infermap.recommender import Recommendation, recommend
from infermap.registry import get_plugin

from eval.cases import BenchmarkCase


@dataclass
class EvalResult:
    case: BenchmarkCase
    hardware: HardwareProfile
    # Candidate counts
    n_generated: int
    n_after_cost_model: int
    cost_estimates: list[CostEstimate]
    # Full benchmark output
    all_results: list[BenchmarkResult]
    recommendation: Recommendation
    # Baseline: pytorch_fp32 at smallest batch size (reference point for speedup)
    baseline: BenchmarkResult | None
    elapsed_s: float
    errors: list[str] = field(default_factory=list)

    @property
    def search_reduction_pct(self) -> float:
        """Fraction of candidates eliminated by the cost model (0–1)."""
        if self.n_generated == 0:
            return 0.0
        return (self.n_generated - self.n_after_cost_model) / self.n_generated

    @property
    def speedup(self) -> float | None:
        """Latency speedup of recommendation vs FP32 baseline at same batch size."""
        if self.baseline is None or not self.recommendation.result.ok:
            return None
        base_ms = self.baseline.latency_p50_ms
        win_ms = self.recommendation.result.latency_p50_ms
        if win_ms <= 0:
            return None
        return base_ms / win_ms

    @property
    def memory_reduction_pct(self) -> float | None:
        """Fraction of memory saved vs FP32 baseline (0–1; negative = more memory used)."""
        if self.baseline is None or not self.recommendation.result.ok:
            return None
        base_mb = self.baseline.memory_mb
        win_mb = self.recommendation.result.memory_mb
        if base_mb <= 0:
            return None
        return (base_mb - win_mb) / base_mb

    @property
    def accuracy_delta(self) -> float | None:
        """Accuracy drop of the recommended candidate (None if not measured)."""
        return self.recommendation.result.accuracy_drop


def run_case(
    case: BenchmarkCase,
    hardware: HardwareProfile | None = None,
    warmup_iters: int = 5,
    measure_iters: int = 20,
    timeout_s: float | None = 60.0,
    objective: str = "latency",
    max_quality_loss: float | None = None,
) -> EvalResult:
    """Run one BenchmarkCase and return an EvalResult."""
    t0 = time.perf_counter()
    errors: list[str] = []

    if hardware is None:
        hardware = profile_hardware()

    # Load model
    model = case.load_model()

    # Inspect
    plugin = get_plugin(model)
    info = plugin.inspect(model, input_shape=case.input_shape)

    # Generate candidates
    all_candidates: list[DeploymentCandidate] = plugin.generate_candidates(info, hardware)
    n_generated = len(all_candidates)

    # Cost-model pruning
    kept, cost_estimates = prune_candidates(
        all_candidates, info, hardware, batch_size=case.batch_sizes[0]
    )
    n_after_cost_model = len(kept)

    # Load eval dataset if available
    eval_dataset: EvalDataset | None = None
    if case.load_eval is not None:
        try:
            inputs, labels = case.load_eval()
            eval_dataset = EvalDataset(
                inputs=inputs,
                labels=labels,
                metric="accuracy",
                task="classification",
            )
        except Exception as exc:
            errors.append(f"eval load failed: {exc}")

    # Benchmark each (candidate, batch_size) pair
    all_results: list[BenchmarkResult] = []
    for cand in kept:
        for bs in case.batch_sizes:
            result = plugin.benchmark(
                cand, model, info,
                input_shape=case.input_shape,
                batch_size=bs,
                warmup_iters=warmup_iters,
                measure_iters=measure_iters,
                timeout_s=timeout_s,
                calibration_inputs=None,
            )
            all_results.append(result)
            if not result.ok:
                errors.append(f"{cand.backend} bs={bs}: {result.error}")

    # Fill accuracy drop for all results that support it
    if eval_dataset is not None:
        try:
            fill_accuracy_drop(all_results, model, info, eval_dataset)
        except Exception as exc:
            errors.append(f"accuracy eval failed: {exc}")

    # Find the FP32 baseline at the smallest batch size
    baseline: BenchmarkResult | None = None
    for r in all_results:
        if r.candidate.backend == "pytorch_fp32" and r.batch_size == case.batch_sizes[0] and r.ok:
            baseline = r
            break

    # Recommend
    recommendation = recommend(
        all_results,
        objective=objective,
        max_quality_loss=max_quality_loss,
    )

    elapsed = time.perf_counter() - t0
    return EvalResult(
        case=case,
        hardware=hardware,
        n_generated=n_generated,
        n_after_cost_model=n_after_cost_model,
        cost_estimates=cost_estimates,
        all_results=all_results,
        recommendation=recommendation,
        baseline=baseline,
        elapsed_s=elapsed,
        errors=errors,
    )
