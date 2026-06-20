# Contributing to aphex

## Setup

aphex uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/<your-fork>/aphex
cd aphex
uv sync --all-extras --dev
```

This installs all optional extras (torch, sklearn, onnx, aws, gcp) plus the dev tools (pytest, ruff, mypy).

## Running tests

```bash
# Fast suite — no network, no real models (~7 s)
uv run pytest tests/

# Include CUDA-specific tests (requires a GPU)
uv run pytest tests/ -m requires_cuda

# Integration tests — download real model weights
uv run pytest tests/ -m integration
```

Integration tests are excluded by default because they pull weights from the internet. They are the right regression test to run before opening a PR that touches the benchmark or plugin layer.

## Code style

```bash
uv run ruff check infermap/ tests/   # lint
uv run ruff format infermap/ tests/  # auto-format
uv run mypy infermap/                # type check
```

CI runs all three. A PR that fails lint or type-check won't be merged.

Line length is 100. Target is Python 3.12+.

## Project layout

```
infermap/
  cli.py            # Typer CLI — entry point for all commands
  benchmark.py      # timing loop, BenchmarkResult
  candidates.py     # DeploymentCandidate generation
  evaluator.py      # accuracy/F1/MAE/RMSE measurement, cloud eval data download
  recommender.py    # Pareto filtering + objective ranking → Recommendation
  pareto.py         # build_pareto_frontier(), rank_by_objective()
  profiler.py       # HardwareProfile — CPU, CUDA, MPS, CoreML
  preflight.py      # feasibility check before benchmarking
  inspector.py      # ModelInfo — parameter count, memory, family
  converter.py      # export to .pt / .onnx / .engine / .xml
  selector.py       # candidate fingerprint-based pre-selection
  cost_model.py     # prune candidates predicted >10× slower than best
  checker.py        # regression check against a saved deployment.yaml
  deployment.py     # deployment.yaml read/write
  report.py         # HTML benchmark report
  serving/          # serving config generators (Triton, TorchServe, BentoML, FastAPI)
  cloud/
    remote.py       # SSH-based remote execution
    storage.py      # S3 + GCS backends
    registry.py     # push/pull versioned model artifacts
    instances.py    # cloud instance profiles for --target
  plugins/          # per-framework ModelPlugin implementations
    pytorch.py
    sklearn.py
    tensorflow.py
    llm.py
  registry.py       # plugin auto-detection from file extension / magic bytes
tests/
  integration/      # real-model tests; excluded from default run
```

## Adding a backend

1. Add a `DeploymentCandidate` entry in `candidates.py` for the appropriate hardware path (`_cuda_candidates`, `_mps_candidates`, `_cpu_candidates`).
2. Add a benchmark branch in `benchmark.py` (`_benchmark_candidate`) that runs the backend and returns latency/throughput numbers.
3. Add the backend key to `ALL_CONVERTIBLE` in `converter.py` and implement the export branch if the backend produces a file artifact.
4. Add accuracy evaluation support in `evaluator.py` (`_PYTORCH_EVAL_BACKENDS` or the sklearn equivalent) if the backend can degrade quality.
5. Cover all four steps with tests.

## Adding a model family (plugin)

Implement the `ModelPlugin` abstract class in `infermap/plugin.py`:

- `can_handle(path)` — return True if this plugin owns the file
- `inspect(path)` → `ModelInfo`
- `load(path)` → model object
- `generate_candidates(info, hw)` → `list[DeploymentCandidate]`
- `benchmark(candidate, model, info, shape, batch_size, ...)` → `BenchmarkResult`

Register the plugin in `infermap/registry.py`.

## Pull requests

- Keep PRs focused - one feature or fix per PR.
- New behaviour needs a test. Bug fixes should include a regression test.
- Update `README.md` if you change the CLI interface or add a supported backend.
- CI must pass (lint + type check + tests) before review.
