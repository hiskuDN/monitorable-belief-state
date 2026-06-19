"""
Shared Modal scaffolding for nextlat-safety experiments.

Each experiment family (gridworld/, toolworld/, ...) defines its OWN modal.App
and Volume but reuses this image + run helpers so the dependency set and mount
layout stay identical across families.

Layout assumed by the mount: the whole `experiments/` tree is mounted at
`/root/experiments`, and the cloned upstream repo at `/root/NextLat`. So a
family's remote scripts are invoked by their subpath, e.g.
`python /root/experiments/gridworld/probe_exp0.py` (cwd=/root/NextLat).
"""

import os
import subprocess

import modal

# --- repo layout (host side) ---
COMMON_DIR = os.path.dirname(os.path.abspath(__file__))   # experiments/common
EXPERIMENTS_DIR = os.path.dirname(COMMON_DIR)             # experiments
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)           # repo root
REPO = os.path.join(PROJECT_ROOT, "NextLat")              # cloned upstream model+harness
RESULTS_BASE = os.path.join(PROJECT_ROOT, "docs", "results")

# --- container paths (remote side) ---
REMOTE_REPO = "/root/NextLat"
REMOTE_EXPERIMENTS = "/root/experiments"
OUT = "/outputs"

GPU = "A100-80GB"  # NextLat's mtp_horizon=8 latent unroll needs ~22GB at micro-batch
# 256 at the v2b scale (L4/d256); the v3 scale sweep (L8/d512) ~4x's activation memory,
# so we use the 80GB card and keep grad_accum=1 (a clean, untouched train path —
# grad_accum>1 + drop_last=False val batches hits an empty-micro-batch crash in the
# upstream rope pos.max()). The GPT arm fits comfortably; same GPU for both.

PIP = [
    "torch>=2.6.0",
    "numpy",
    "lightning",
    "omegaconf>=2.3.0",
    "transformers",
    "datasets==4.6.1",
    "huggingface_hub",
    "scikit-learn",
    "wandb",
    "tqdm",
    "pyyaml",
]


def build_image():
    """Debian + ML deps, with the upstream repo and the whole experiments/ tree
    mounted at runtime (copy=False -> local edits picked up without an image rebuild)."""
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(*PIP)
        .add_local_dir(REPO, REMOTE_REPO, copy=False)
        .add_local_dir(EXPERIMENTS_DIR, REMOTE_EXPERIMENTS, copy=False)
    )


def run(cmd, cwd):
    """Echo + run a subprocess, raising on non-zero exit."""
    print(f"[modal] $ {' '.join(cmd)}  (cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)
