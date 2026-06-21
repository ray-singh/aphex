"""APHEX Evaluation Suite — entry point.

Usage
-----
# Smoke test (tiny synthetic models, always works, no optional deps needed):
python eval/run.py --smoke

# Full suite (requires torchvision, transformers, lightgbm):
python eval/run.py

# Specific cases:
python eval/run.py --cases ResNet-50 LightGBM-200

# Write JSON results:
python eval/run.py --smoke --out results.json

# Constrain quality loss (filters candidates that degrade accuracy >2%):
python eval/run.py --max-quality-loss 0.02

# Run with more measurement iterations for stable numbers:
python eval/run.py --smoke --iters 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="APHEX Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run the smoke suite (tiny synthetic models, no optional deps required)",
    )
    parser.add_argument(
        "--cases", nargs="+", metavar="NAME",
        help="Run only these named cases from the full suite",
    )
    parser.add_argument(
        "--out", type=Path, default=None, metavar="PATH",
        help="Write full results as JSON to this path",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Warmup iterations per candidate (default: 5)",
    )
    parser.add_argument(
        "--iters", type=int, default=20,
        help="Measurement iterations per candidate (default: 20)",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Per-candidate timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--objective", choices=["latency", "throughput", "memory"], default="latency",
        help="Optimization objective for recommendation (default: latency)",
    )
    parser.add_argument(
        "--max-quality-loss", type=float, default=None, metavar="FLOAT",
        help="Max relative accuracy drop vs baseline (0.0–1.0); filters candidates",
    )
    args = parser.parse_args()

    from eval.cases import SMOKE_SUITE, SUITE
    from eval.report import print_summary, write_json
    from eval.runner import run_case

    if args.smoke:
        cases = SMOKE_SUITE
    elif args.cases:
        by_name = {c.name: c for c in SUITE}
        missing = [n for n in args.cases if n not in by_name]
        if missing:
            print(f"Unknown cases: {missing}")
            print(f"Available: {list(by_name)}")
            sys.exit(1)
        cases = [by_name[n] for n in args.cases]
    else:
        cases = SUITE

    results = []
    for case in cases:
        print(f"  running {case.name} ...", flush=True)
        try:
            result = run_case(
                case,
                warmup_iters=args.warmup,
                measure_iters=args.iters,
                timeout_s=args.timeout if args.timeout > 0 else None,
                objective=args.objective,
                max_quality_loss=args.max_quality_loss,
            )
            results.append(result)
        except Exception as exc:
            print(f"  [SKIP] {case.name} failed to run: {exc}")

    if not results:
        print("No results — all cases failed or were skipped.")
        sys.exit(1)

    print_summary(results)

    if args.out:
        write_json(results, args.out)


if __name__ == "__main__":
    main()
