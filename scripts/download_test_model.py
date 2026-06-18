#!/usr/bin/env python3
"""Download EfficientNet-B0 + ImageNet validation samples for aphex optimization testing.

ImageNet samples are streamed from Hugging Face (ILSVRC/imagenet-1k) — only the 288
samples needed for calibration and eval are downloaded, not the full 150 GB dataset.

Prerequisites
-------------
  1. Accept dataset terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
  2. Authenticate:  huggingface-cli login   (or set HF_TOKEN env var)
  3. pip install datasets

Produces
--------
  model.pt        — pretrained EfficientNet-B0 (whole model, torch.save)
  calib_data.pt   — 32 calibration tensors for INT8 backends
  eval_data.pt    — 256 labelled samples for --eval accuracy measurement
  infer.py        — inference callable for --infer-fn infer.py:predict

Usage
-----
  python scripts/download_test_model.py
  aphex optimize model.pt --input-shape 3,224,224 --calibration-data calib_data.pt
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

OUT = Path(".")
N_CALIB = 32
N_EVAL = 256
N_TOTAL = N_CALIB + N_EVAL  # stream only what we need

# ── model ─────────────────────────────────────────────────────────────────────

print("Downloading EfficientNet-B0 weights...")
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights).eval()
torch.save(model, OUT / "model.pt")
n_params = sum(p.numel() for p in model.parameters())
print(f"  saved model.pt  ({n_params:,} params, {n_params * 4 / 1e6:.1f} MB fp32)")

# ── ImageNet samples via Hugging Face streaming ────────────────────────────────

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit(
        "\nMissing dependency: pip install datasets\n"
        "Then re-run this script."
    )

print(f"\nStreaming {N_TOTAL} ImageNet validation samples from Hugging Face...")
print("  (requires HF login and accepted terms at huggingface.co/datasets/ILSVRC/imagenet-1k)\n")

try:
    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )
except Exception as exc:
    raise SystemExit(
        f"\nCould not load ImageNet dataset: {exc}\n\n"
        "Steps to fix:\n"
        "  1. Accept terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
        "  2. Run:  huggingface-cli login\n"
        "  3. Re-run this script."
    )

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tensors: list[torch.Tensor] = []
labels: list[int] = []

for i, sample in enumerate(ds):
    if i >= N_TOTAL:
        break
    img = sample["image"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    tensors.append(transform(img))
    labels.append(int(sample["label"]))
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{N_TOTAL} samples")

print(f"  {N_TOTAL}/{N_TOTAL} samples")

if len(tensors) < N_TOTAL:
    raise SystemExit(
        f"\nOnly got {len(tensors)} samples (expected {N_TOTAL}). "
        "The validation split may have fewer samples than expected."
    )

# ── calibration data ──────────────────────────────────────────────────────────

calib = [tensors[i].unsqueeze(0) for i in range(N_CALIB)]
torch.save(calib, OUT / "calib_data.pt")
print(f"\n  saved calib_data.pt  ({N_CALIB} samples, shape {list(calib[0].shape)})")

# ── eval data ─────────────────────────────────────────────────────────────────

inputs = torch.stack(tensors[N_CALIB:])
eval_labels = torch.tensor(labels[N_CALIB:])
torch.save({"inputs": inputs, "labels": eval_labels}, OUT / "eval_data.pt")
print(f"  saved eval_data.pt   ({N_EVAL} samples, shape {list(inputs.shape)})")
print(f"  label range: {eval_labels.min().item()}–{eval_labels.max().item()} (ImageNet classes 0–999)")

# ── inference function ────────────────────────────────────────────────────────

INFER_SRC = '''\
"""Inference function for aphex --infer-fn infer.py:predict.

Receives a list of tensors (one per sample, each [1, 3, 224, 224]) and returns
a numpy array of class indices of shape [N].
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

_model = None

def predict(inputs: list) -> np.ndarray:
    global _model
    if _model is None:
        _model = torch.load(Path(__file__).parent / "model.pt", weights_only=False).eval()
    batch = torch.cat([
        x if isinstance(x, torch.Tensor) else torch.tensor(x)
        for x in inputs
    ], dim=0).float()
    with torch.no_grad():
        logits = _model(batch)
    return logits.argmax(dim=1).numpy()
'''

(OUT / "infer.py").write_text(INFER_SRC)
print("  saved infer.py")

# ── instructions ──────────────────────────────────────────────────────────────

print("""
Done. Run aphex:

  # basic optimization
  aphex optimize model.pt --input-shape 3,224,224

  # with INT8 calibration
  aphex optimize model.pt --input-shape 3,224,224 \\
    --calibration-data calib_data.pt

  # with accuracy drop tracking (ImageNet top-1)
  aphex optimize model.pt --input-shape 3,224,224 \\
    --calibration-data calib_data.pt \\
    --eval eval_data.pt \\
    --infer-fn infer.py:predict \\
    --max-quality-loss 0.02 \\
    --output deployment.yaml
""")
