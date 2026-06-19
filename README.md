<p align="center">
  <img src="docs/logo/lockup-light.svg#gh-light-mode-only" alt="aphex" height="120"/>
  <img src="docs/logo/lockup-dark.svg#gh-dark-mode-only" alt="aphex" height="120"/>
</p>

A hardware-aware deployment planner that profiles arbitrary ML models, searches the deployment space, and produces a recommended serving configuration with predicted latency/throughput tradeoffs — locally or on a remote cloud machine.

## Features

- **Hardware profiling**: detects CPU cores, RAM, CUDA GPUs, Apple MPS, and CoreML availability
- **Model inspection**: parameter count, memory footprint (FP32/FP16/BF16), architecture family
- **Pre-flight checks**: fast feasibility check before committing to a full benchmark run
- **Multi-backend benchmarking**: PyTorch (FP32/FP16/BF16), ONNX Runtime (CPU/CUDA/CoreML), `torch.compile`, INT8 quantization, TensorRT, OpenVINO
- **Batch size sweep**: benchmarks every backend across multiple batch sizes in one run
- **Pareto-optimal recommendation**: picks the best `(backend, batch_size)` pair for your objective (latency, throughput, or memory)
- **Artifact export**: converts the recommended model to its deployment format (`.pt`, `.onnx`, `.engine`, `.xml`)
- **HTML report**: interactive latency-vs-throughput chart with full candidate table
- **Remote execution**: runs the full benchmark pipeline on an EC2 instance (or any SSH host) and pulls results back locally
- **Cloud registry**: push/pull versioned model artifacts to S3
- **sklearn / XGBoost / LightGBM / CatBoost support**: ONNX export for traditional ML models

## Installation

Install the core CLI (no ML frameworks):

```bash
pip install aphex
```

Add the extras you need:

```bash
pip install 'aphex[torch]'      # PyTorch benchmarking (~2 GB)
pip install 'aphex[sklearn]'    # scikit-learn / tree model support
pip install 'aphex[onnx]'       # ONNX export + ONNX Runtime
pip install 'aphex[tensorflow]' # TensorFlow models
pip install 'aphex[aws]'        # S3 registry + EC2 remote execution
pip install 'aphex[full]'       # everything above
```

## Quickstart

```bash
# Inspect hardware and model
aphex analyze model.pt

# Feasibility check before benchmarking
aphex preflight model.pt --dtype fp16

# Benchmark all deployment strategies
aphex benchmark model.pt --input-shape 3,224,224

# Get an optimized recommendation + deployment artifact
aphex optimize model.pt --input-shape 3,224,224 --objective latency

# Save an HTML report and metrics JSON
aphex optimize model.pt --input-shape 3,224,224 --report report.html --metrics metrics.json
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
```

## CLI reference

| Command | Description |
|---|---|
| `aphex analyze <model>` | Hardware profile + model inspection |
| `aphex preflight <model>` | Feasibility check (fast, no benchmarking) |
| `aphex benchmark <model>` | Full benchmark across all backends |
| `aphex optimize <model>` | Benchmark + Pareto-optimal recommendation + artifact export |
| `aphex convert <model>` | Convert a model to a specific backend format |
| `aphex check <host>` | Verify aphex is installed on a remote SSH host |
| `aphex targets` | List available hardware targets |
| `aphex push <deployment.yaml> <artifact>` | Push a versioned model to S3 |
| `aphex pull <name>` | Pull a model artifact from S3 |
| `aphex ls` | List models and versions in the S3 registry |

### Common options

```
--input-shape 3,224,224     Input tensor shape (no batch dim)
--batch-sizes 1,2,4,8       Batch sizes to sweep (comma-separated)
--objective latency          Optimization goal: latency | throughput | memory
--max-latency-ms 5.0        Hard latency constraint (p50)
--max-memory-mb 512         Hard memory constraint
--min-throughput-rps 200    Hard throughput constraint
--calibration-data PATH     .pt file or image dir for INT8 accuracy measurement
--format table|json         Output format (json suppresses Rich output)
--report PATH               Write an HTML benchmark report
--metrics PATH              Write benchmark metrics as JSON
--remote HOST               Run benchmark on a remote SSH host
--output PATH               Where to write the deployment artifact
```

## AWS integration

### Remote benchmarking on EC2

Run the full benchmark pipeline on a remote machine — useful when you want results for a GPU instance without setting up a local GPU environment.

```bash
# Benchmark on an EC2 instance and pull results back locally
aphex optimize model.pt \
  --input-shape 3,224,224 \
  --remote ec2-user@<instance-ip> \
  --output deployment.yaml \
  --report report.html \
  --metrics metrics.json
```

aphex uploads the model, runs the full benchmark on the remote host, streams output to your terminal, then downloads `deployment.yaml`, the HTML report, and the metrics JSON. The remote temp directory is cleaned up automatically.

**Setup**

1. Add the instance to `~/.ssh/config`:

```
Host <instance-ip>
    IdentityFile ~/.ssh/your-key.pem
    User ec2-user
    StrictHostKeyChecking no
```

2. Install aphex on the instance:

```bash
ssh ec2-user@<instance-ip> "pip install 'aphex[torch,onnx]'"
```

3. Verify the connection:

```bash
aphex check ec2-user@<instance-ip>
```

**Recommended instance type for cost-effective benchmarking:** `t3a.large` (8 GB RAM, ~$0.02/hr as a spot instance) covers most CPU/ONNX workloads. Use a `g4dn.xlarge` for GPU benchmarking.

### S3 model registry

Push versioned model artifacts to S3 and pull them from any machine.

```bash
# Configure storage (one-time)
export APHEX_BUCKET=my-models-bucket
export AWS_REGION=us-east-1

# Push a deployment artifact
aphex push deployment.yaml model.onnx --name resnet50 --version v1

# Pull on another machine
aphex pull resnet50             # latest version
aphex pull resnet50@v1          # specific version
aphex pull resnet50 --out ./models/

# List what's in the registry
aphex ls                        # all models
aphex ls resnet50               # versions of a specific model
```

Credentials are picked up from the standard AWS chain (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, `~/.aws/credentials`, or an IAM instance role).

## Pipeline

```
model.pt + hardware
       |
       v
  inspect_model()     → parameters, memory, family
  profile_hardware()  → CPU, RAM, GPU / MPS / CoreML
       |
       v
  run_preflight()     → feasibility: ok / tight / unlikely / impossible
       |
       v
  generate_candidates() → (backend, dtype, device, batch_size) combos
       |
       v
  benchmark_candidate() × (backends × batch sizes) → p50 / p95 / p99, throughput, memory
       |
       v
  recommend()         → Pareto frontier → best candidate for objective
       |
       v
  convert()           → deployment artifact (.pt / .onnx / .engine / .xml)
```

## Supported backends

| Backend | Device | Dtype |
|---|---|---|
| PyTorch eager | CPU | FP32 |
| PyTorch eager | MPS (Apple Silicon) | FP32, FP16 |
| PyTorch eager | CUDA | FP32, FP16, BF16 |
| torch.compile | CPU / CUDA | FP32 |
| ONNX Runtime | CPU | FP32 |
| ONNX Runtime + CoreML | Apple Silicon | FP32 |
| ONNX Runtime | CUDA | FP32 |
| PyTorch INT8 dynamic | CPU | INT8 |
| ONNX Runtime INT8 | CPU | INT8 |
| TensorRT | CUDA | FP32, FP16, INT8 |
| OpenVINO | CPU | FP32, INT8 |

## Requirements

- Python 3.12+
- At least one framework extra (`aphex[torch]`, `aphex[sklearn]`, etc.)
- For remote execution: `ssh` and `scp` on the local machine, `aphex` installed on the remote

## License

MIT
