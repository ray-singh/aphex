<p align="center">
  <img src="docs/logo/lockup-light.svg#gh-light-mode-only" alt="aphex" height="120"/>
  <img src="docs/logo/lockup-dark.svg#gh-dark-mode-only" alt="aphex" height="120"/>
</p>

A hardware-aware ML optimization and recommendation framework.

aphex profiles your hardware, inspects your PyTorch model, benchmarks every viable deployment strategy, and recommends the fastest option that fits your constraints -- all from a single CLI command.

## Features

- **Hardware profiling**: detects CPU cores, RAM, CUDA GPUs, Apple MPS, and CoreML availability
- **Model inspection**: parameter count, memory footprint (FP32/FP16), architecture family
- **Pre-flight checks**: fast feasibility check before committing to a full benchmark run
- **Multi-backend benchmarking**: PyTorch (FP32/FP16/BF16), ONNX Runtime (CPU/CUDA/CoreML), `torch.compile`, INT8 quantization
- **INT8 quantization**: dynamic quantization via PyTorch and ONNX Runtime, with optional accuracy-drop measurement against calibration data
- **Batch size sweep**: benchmarks every backend across multiple batch sizes in one run; recommends the best `(backend, batch_size)` pair
- **JSON output**: `--format json` emits machine-readable results for CI/CD pipelines
- **Pareto-optimal recommendation**: picks the best strategy for your objective (latency, throughput, or memory)

## Installation

```bash
pip install aphex
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
racing 7 backends × 4 batch sizes

  ✓ PyTorch FP32 CPU           bs=1     17.55 ms      57 req/s
  ✓ PyTorch FP32 CPU           bs=8      2.44 ms     410 req/s
  ✓ ONNX Runtime + CoreML      bs=1      0.92 ms    1085 req/s
  ✓ ONNX Runtime + CoreML      bs=8      0.31 ms    3226 req/s
  ✓ ONNX Runtime INT8 (CPU)    bs=1      0.01 ms    9200 req/s
  ✓ ONNX Runtime INT8 (CPU)    bs=8      0.04 ms   24800 req/s
  ...

  #1  ONNX Runtime INT8 (CPU)   bs=8   0.04 ms   24800 req/s  ████████████████░░░░
  #2  ONNX Runtime INT8 (CPU)   bs=4   0.03 ms   16600 req/s  █████████████░░░░░░░
  #3  ONNX Runtime + CoreML     bs=8   0.31 ms    3226 req/s  ██░░░░░░░░░░░░░░░░░░
  ...
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
--batch-sizes 1,2,4,8     Batch sizes to sweep (comma-separated)
--warmup 10               Warm-up iterations before timing
--iters 100               Measurement iterations
--objective latency       Optimization goal: latency | throughput | memory
--max-latency-ms 5.0      Hard constraint on p50 latency
--max-memory-mb 512       Hard constraint on peak memory
--min-throughput-rps 200  Hard constraint on throughput
--calibration-data PATH   .pt file or image directory for INT8 accuracy measurement
--format table|json       Output format (json suppresses all Rich output)
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
  benchmark_candidate() x (backends × batch_sizes) --> p50/p95/p99, throughput, memory
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
| PyTorch INT8 dynamic | CPU | INT8 |
| ONNX Runtime INT8 | CPU | INT8 |

## Requirements

- Python 3.12+
- PyTorch 2.2+
- onnxruntime 1.17+

## License

MIT
