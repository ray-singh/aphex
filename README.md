<p align="center">
  <img src="docs/logo/lockup-light.svg#gh-light-mode-only" alt="aphex" height="120"/>
  <img src="docs/logo/lockup-dark.svg#gh-dark-mode-only" alt="aphex" height="120"/>
</p>

A hardware-aware deployment planner that profiles arbitrary ML models, searches the deployment space, and produces a recommended serving configuration with predicted latency/throughput tradeoffs — locally or on a remote cloud machine.

---

- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Example output](#example-output)
- [CLI reference](#cli-reference)
- [Pruning](#pruning)
- [Distillation](#distillation)
- [Multi-GPU benchmarking](#multi-gpu-benchmarking)
- [AWS integration](#aws-integration)
- [Pipeline](#pipeline)
- [Supported backends](#supported-backends)
- [Out of scope](#out-of-scope)
- [Requirements](#requirements)

---

## Features

- **Hardware profiling**: detects CPU cores, RAM, CUDA GPUs, Apple MPS, and CoreML availability
- **Model inspection**: parameter count, memory footprint (FP32/FP16/BF16), architecture family
- **Pre-flight checks**: fast feasibility check before committing to a full benchmark run
- **Multi-backend benchmarking**: PyTorch (FP32/FP16/BF16), ONNX Runtime (CPU/CUDA/CoreML), `torch.compile`, INT8 quantization, TensorRT, OpenVINO
- **Batch size sweep**: benchmarks every backend across multiple batch sizes in one run
- **Quality-constrained recommendation**: requires a labelled test dataset; measures accuracy/F1/MAE/RMSE drop from quantization and filters candidates that exceed your tolerance before ranking
- **Magnitude pruning**: 30 / 50 / 70 % unstructured + 2:4 structured (Ampere+ sparse Tensor Cores) appear as first-class benchmark candidates with measured accuracy drop
- **Knowledge distillation**: `aphex distill` trains a user-supplied student model from a teacher using soft-label KD (Hinton KL on temperature-scaled logits + optional CE)
- **Multi-GPU data-parallel sweep**: when ≥2 CUDA devices are detected, aphex automatically benchmarks `nn.DataParallel` variants (`pytorch_dp{2,4,8}_{fp32,fp16,bf16}`) so throughput scaling on multi-GPU boxes is measured, not assumed
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

# Get an optimized recommendation (eval data + inference callable required)
aphex optimize model.pt --input-shape 3,224,224 \
  --eval val.pt --infer-fn infer.py:predict \
  --max-accuracy-loss 0.02

# Regression model: constrain by MAE instead
aphex optimize model.pt --input-shape 16 --eval val.pt --max-mae-loss 0.05 --objective latency

# Save an HTML report and metrics JSON
aphex optimize model.pt --input-shape 3,224,224 --eval val.pt --max-accuracy-loss 0.02 \
  --report report.html --metrics metrics.json

# Distill a teacher into a smaller student (training-based; requires labels)
aphex distill teacher.pt --student make_student.py:tiny_mlp \
  --eval val.pt --epochs 5 --output student.pt
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
| `aphex distill <teacher>` | Knowledge-distill a teacher into a smaller student model |
| `aphex check <model> --from-config deployment.yaml` | Regression-check a model against a saved deployment baseline (CI-friendly; exits 1 on threshold breach) |
| `aphex targets` | List available hardware targets |
| `aphex push <deployment.yaml> <artifact>` | Push a versioned model to S3 |
| `aphex pull <name>` | Pull a model artifact from S3 |
| `aphex ls` | List models and versions in the S3 registry |

### Common options

```
--input-shape 3,224,224     Input tensor shape (no batch dim)
--batch-sizes 1,2,4,8       Batch sizes to sweep (comma-separated)
--objective latency          Optimization goal: latency | throughput | memory
--eval PATH                 Labelled test dataset (.pt, .csv, .parquet, image dir, or s3://, gs:// URI). Required for optimize.
--infer-fn module.py:fn     Inference callable for true accuracy measurement. Required to score `--eval` against the user's full pipeline.
--max-accuracy-loss 0.02    Max relative accuracy drop vs original model baseline (classification)
--max-f1-loss 0.02          Max relative macro-F1 drop vs original model baseline (classification)
--max-mae-loss 0.05         Max relative MAE increase vs original model baseline (regression)
--max-rmse-loss 0.05        Max relative RMSE increase vs original model baseline (regression)
--max-latency-ms 5.0        Hard latency constraint (p50)
--max-memory-mb 512         Hard memory constraint
--min-throughput-rps 200    Hard throughput constraint
--calibration-data PATH     .pt file or image dir for INT8 quantization calibration
--format table|json         Output format (json suppresses Rich output)
--report PATH               Write an HTML benchmark report
--metrics PATH              Write benchmark metrics as JSON
--remote HOST               Run benchmark on a remote SSH host
--output PATH               Where to write the deployment artifact
--jobs N, -j N              Run N (candidate, batch_size) benchmarks in parallel (default 1; keep at 1 for accurate latency/memory)
```

### Loading user models safely

Some models can only be loaded with PyTorch's pickle-based loader, which executes arbitrary code on load. aphex defaults to the safe `weights_only=True` path; if it fails with a pickle-related error you get a friendly message listing both options
(re-save as a `state_dict`, or opt in for trusted files):

```bash
APHEX_TRUST_PICKLE=1 aphex optimize model.pt --input-shape 3,224,224 --eval val.pt ...
```

Do not set this for models from untrusted sources.

## Pruning

Magnitude pruning is wired in as four additional benchmark candidates on every device path, so `aphex benchmark` / `aphex optimize` automatically score them alongside FP16, INT8, etc.:

| Backend | Sparsity | Notes |
|---|---|---|
| `pytorch_prune_unstructured_30` | 30 % | L1-magnitude, every Linear/Conv weight |
| `pytorch_prune_unstructured_50` | 50 % | same, more aggressive |
| `pytorch_prune_unstructured_70` | 70 % | accuracy cost is usually visible past 50 % |
| `pytorch_prune_2_4` | 50 % structured | 2-of-4 pattern for NVIDIA Ampere+ sparse Tensor Cores |

aphex measures both **latency** and **accuracy drop** for pruned candidates through the same pipeline as quantized backends. Latency improvement on dense CPUs is usually modest; the value is the storage / accuracy tradeoff and, on Ampere+ GPUs, the 2:4 sparse-kernel speedup. Use the existing `--max-accuracy-loss` / `--max-f1-loss` / etc. flags to filter out pruned variants that exceed your quality budget before ranking.

aphex's pruning is **post-training**: no labels, no gradient updates. For recovery training, distill into a smaller dense student instead (below).

## Distillation

`aphex distill` is the only command that performs gradient updates. It trains a **student** model to imitate a **teacher** using soft-label knowledge 

Distillation:
```
L = α · KL( softmax(student / T) || softmax(teacher / T) ) · T²
  + (1 - α) · CE(student, hard_label)        # classification
L = MSE(student, teacher)                    # regression
```

The student architecture is yours; provide a zero-argument factory function and aphex handles the training loop, scoring, and report:

```bash
# Write a tiny factory file
cat > make_student.py <<'PY'
import torch.nn as nn
def tiny_mlp():
    return nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 3))
PY

aphex distill teacher.pt \
  --student make_student.py:tiny_mlp \
  --eval val.pt \
  --epochs 8 --batch-size 16 --lr 1e-2 \
  --temperature 3.0 --alpha 0.7 \
  --task classification --device cpu \
  --output student.pt --report distill_report.json
```

Output (excerpt):

```
  teacher params  387
  student params  51
  epochs=8  batch=16  lr=0.01  temp=3.0  alpha=0.7  task=classification  device=cpu

  compression  7.6×
  final loss   1.93  (first epoch 3.99)
  accuracy     teacher 1.0000  →  student 0.7350
  ✓  student state_dict → student.pt
```

Common flags:

```
--student module.py:fn      Student factory: zero-arg callable returning an nn.Module
--eval PATH                 Labelled dataset for distillation
--task classification|regression
--temperature 4.0           KD softmax temperature (higher = softer teacher distribution)
--alpha 0.7                 Weight on KD loss; (1 - alpha) on hard-label CE (classification only)
--epochs 3 --batch-size 32 --lr 1e-3
--device cpu|cuda|mps
--output student.pt         Destination for the trained student state_dict
--report report.json        Optional JSON: per-epoch losses, param counts, teacher/student scores
```

The teacher is frozen during training. With `labels=None` aphex falls back to
pure KD (`alpha=1.0`). The output is a `state_dict` — reconstruct your student
with the same factory + `load_state_dict()` to deploy or feed back into
`aphex optimize` for a deployment-format search.

## Multi-GPU benchmarking

When `profile_hardware()` detects more than one CUDA device, the candidate generator emits single-process `nn.DataParallel` variants alongside the regular PyTorch backends:

| Backend | Devices | Dtype |
|---|---|---|
| `pytorch_dp2_{fp32,fp16,bf16}` | 2 × GPU | matches dtype suffix |
| `pytorch_dp4_{fp32,fp16,bf16}` | 4 × GPU (host must have ≥4) | — |
| `pytorch_dp8_{fp32,fp16,bf16}` | 8 × GPU (host must have ≥8) | — |

BF16 variants are only emitted on Ampere+ (sm_80+). DP shards the batch dimension across replicas, so it's a **throughput** win at large batch and a **latency** no-op (or slight loss) at `batch=1`. The runner enforces `--batch-size >= N` and surfaces a clear error otherwise — feed a multi-GPU sweep a sensible batch list:

```bash
aphex benchmark model.pt --input-shape 3,224,224 --batch-sizes 8,16,32
```

DP candidates do not run the cosine-similarity accuracy proxy: replication doesn't alter weights, so accuracy is identical to the underlying-dtype candidate (e.g. `pytorch_dp4_fp16` shares the same accuracy signal as `pytorch_fp16`).

For real distributed training / inference (DDP, tensor parallelism, pipeline parallelism, multi-node), see **Out of scope** below.

## AWS integration

### Remote benchmarking on EC2

Run the full benchmark pipeline on a remote machine — useful when you want results for a GPU instance without setting up a local GPU environment.

```bash
# Benchmark on an EC2 instance and pull results back locally
aphex optimize model.pt \
  --input-shape 3,224,224 \
  --eval val.pt \
  --max-accuracy-loss 0.02 \
  --remote ec2-user@<instance-ip> \
  --output deployment.yaml \
  --report report.html \
  --metrics metrics.json
```

aphex uploads the model and eval dataset, runs the full benchmark on the remote host, streams output to your terminal, then downloads `deployment.yaml`, the HTML report, and the metrics JSON. The remote temp directory is cleaned up automatically.

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
ssh ec2-user@<instance-ip> "aphex --help"
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
                          incl. quantized, pruned (30/50/70/2:4), and torch.compile variants
       |
       v
  benchmark_candidate() × (backends × batch sizes) → p50 / p95 / p99, throughput, memory
       |
       v
  evaluate_quality()  → accuracy/F1/MAE/RMSE drop vs original model baseline (--eval dataset)
       |
       v
  recommend()         → Pareto frontier → filter by quality constraint → best candidate for objective
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
| PyTorch + magnitude prune | CPU / MPS / CUDA | FP32 @ 30 / 50 / 70 % sparsity |
| PyTorch + 2:4 structured prune | CPU / MPS / CUDA (Ampere+ for speedup) | FP32 @ 50 % sparsity |
| PyTorch + `nn.DataParallel` | CUDA × {2, 4, 8} GPUs | FP32, FP16, BF16 (throughput-oriented) |

## Out of scope (for current version)

- **Pruning recovery training**: aphex's pruning is post-training only. If your model degrades past tolerance at the sparsity you want, distill into a smaller dense student instead.
- **Quantization-aware training (QAT)**: only post-training quantization is supported.
- **LLM-specific quality metrics**: cosine-similarity proxies are skipped for generative families (`llm`, `transformer_decoder`, `seq2seq`); score those models with a custom `--infer-fn` (perplexity, task benchmarks).
- **Distributed multi-GPU (DDP / tensor parallelism / pipeline parallelism)**: aphex sweeps single-process `nn.DataParallel` candidates (`pytorch_dp{2,4,8}_{fp32,fp16,bf16}`) when ≥2 CUDA devices are detected, but real DDP / `torchrun` orchestration and tensor- or pipeline-parallel sharding are out of scope. DP variants are throughput-oriented and require `--batch-size >= N`.

## Requirements

- Python 3.12+
- At least one framework extra (`aphex[torch]`, `aphex[sklearn]`, etc.)
- For remote execution: `ssh` and `scp` on the local machine, `aphex` installed on the remote

## License

MIT
