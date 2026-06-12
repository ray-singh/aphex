<p align="center">
  <img src="docs/logo/lockup-light.svg#gh-light-mode-only" alt="aphex" height="80"/>
  <img src="docs/logo/lockup-dark.svg#gh-dark-mode-only" alt="aphex" height="80"/>
</p>

A hardware-aware ML optimization and recommendation framework.

aphex profiles your hardware, inspects your PyTorch model, benchmarks every viable deployment strategy, and recommends the fastest option that fits your constraints -- all from a single CLI command.

## Features

- **Hardware profiling**: detects CPU cores, RAM, CUDA GPUs, Apple MPS, and CoreML availability
- **Model inspection**: parameter count, memory footprint (FP32/FP16), architecture family
- **Pre-flight checks**: fast feasibility check before committing to a full benchmark run
- **Multi-backend benchmarking**: PyTorch (FP32/FP16/BF16), ONNX Runtime (CPU/CUDA/CoreML), `torch.compile`
- **Pareto-optimal recommendation**: picks the best strategy for your objective (latency, throughput, or memory)

## Installation

```bash
pip install aphex
```

For CUDA support:

```bash
pip install "aphex[cuda]"
```

## Quickstart

```bash
# Inspect your hardware and model
aphex analyze model.pt

# Run a feasibility check before benchmarking
aphex preflight model.pt --dtype fp16

# Benchmark all deployment strategies
aphex benchmark model.pt --input-shape 3,224,224

# Get an optimized recommendation
aphex optimize model.pt --input-shape 3,224,224 --objective latency
```

## Example output

```
Running 5 candidates...

  ✓ PyTorch FP32 CPU                    p50=  17.55 ms        57 req/s
  ✓ PyTorch FP32 MPS                    p50=   4.95 ms       202 req/s
  ✓ PyTorch FP16 MPS                    p50=   4.64 ms       215 req/s
  ✓ ONNX Runtime + CoreML               p50=   0.92 ms      1085 req/s
  ✓ PyTorch FP32 + torch.compile CPU    p50=   8.10 ms       123 req/s
```

## CLI reference

| Command | Description |
|---------|-------------|
| `aphex analyze <model>` | Hardware profile + model inspection |
| `aphex preflight <model>` | Feasibility check (fast, no benchmarking) |
| `aphex benchmark <model>` | Full benchmark across all backends |
| `aphex optimize <model>` | Benchmark + Pareto-optimal recommendation |

### Common options

```
--input-shape 3,224,224   Input tensor shape (no batch dim)
--batch-size 1            Batch size for benchmarking
--warmup 10               Warm-up iterations before timing
--iters 100               Measurement iterations
--objective latency       Optimization goal: latency | throughput | memory
--max-latency-ms 5.0      Hard constraint on p50 latency
--max-memory-mb 512       Hard constraint on peak memory
--min-throughput-rps 200  Hard constraint on throughput
```

## Pipeline

```
model.pt + hardware
       |
       v
  inspect_model()    --> parameters, memory, family
  profile_hardware() --> CPU, RAM, GPU/MPS/CoreML
       |
       v
  run_preflight()    --> feasibility: ok / tight / unlikely / impossible
       |
       v
  generate_candidates() --> list of (backend, dtype, device) combos
       |
       v
  benchmark_candidate() x N --> p50/p95/p99 latency, throughput, memory
       |
       v
  recommend() --> Pareto frontier -> best candidate for objective
```

## Supported backends

| Backend | Device | Dtype |
|---------|--------|-------|
| PyTorch eager | CPU | FP32 |
| PyTorch eager | MPS (Apple Silicon) | FP32, FP16 |
| PyTorch eager | CUDA | FP32, FP16, BF16 |
| torch.compile | CPU / CUDA | FP32 |
| ONNX Runtime | CPU | FP32 |
| ONNX Runtime + CoreML | Apple Silicon | FP32 |
| ONNX Runtime | CUDA | FP32, FP16 |

## Requirements

- Python 3.12+
- PyTorch 2.2+
- onnxruntime 1.17+

## License

MIT
