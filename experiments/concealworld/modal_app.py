"""
Modal harness for Concealworld (2a) — the deception rung of the monitorable-belief-
state program.

Trains the matched GPT vs predict-ahead (NextLat/MTP/JTP) concealworld arms and
probes them on Modal GPUs, persisting checkpoints/outputs to a Modal Volume. Shared
scaffolding (image, run helper, paths) lives in experiments/common/modal_base.py.

Usage (from the project root, NOT inside NextLat/):
    # quick end-to-end smoke test (tiny, ~minutes):
    python3 -m modal run experiments/concealworld/modal_app.py::smoke

    # train one arm:
    python3 -m modal run experiments/concealworld/modal_app.py::train --arm gpt --seed 1234

    # full sweep (5 arms x seeds) + probes:
    python3 -m modal run experiments/concealworld/modal_app.py::main --seeds 1234,1235,1236

The NextLat repo is mounted at /root/NextLat and the whole experiments/ tree at
/root/experiments (copy=False -> local edits picked up without an image rebuild).
Training runs `python train.py` directly (single GPU, no fabric launcher needed).
"""

import datetime
import glob
import json
import os
import sys

# make `common` importable both locally (run via `modal run .../modal_app.py`) and inside
# the Modal container — modal 1.5 flattens this entrypoint to /root/modal_app.py, so the
# dirname trick misses; build_image mounts the experiments/ tree at /root/experiments.
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_here), "/root/experiments"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import modal  # noqa: E402

from common.modal_base import (  # noqa: E402
    GPU,
    OUT,
    REMOTE_REPO,
    RESULTS_BASE,
    build_image,
    run as _run,
)

app = modal.App("nextlat-conceal", image=build_image())
vol = modal.Volume.from_name("nextlat-conceal", create_if_missing=True)


# arm label -> (config file, extra CLI overrides). nextlat_h1 = ablated control.
CFG_MAP = {
    "gpt": ("gpt_concealworld.yaml", []),
    "nextlat": ("nextlat_concealworld.yaml", []),
    "nextlat_h1": ("nextlat_concealworld.yaml", ["model.mtp_horizon=1"]),
    "mtp": ("mtp_concealworld.yaml", []),
    "jtp": ("jtp_concealworld.yaml", []),
}


def _run_dir(arm, seed, tag):
    return f"{OUT}/{arm}{('_' + tag) if tag else ''}_seed{seed}"


@app.function(gpu=GPU, volumes={OUT: vol}, timeout=60 * 60 * 6, retries=2)
def train_arm(arm: str, seed: int, overrides: list[str] | None = None, tag: str = ""):
    """Train one arm at a given seed. Returns the out_dir + latest checkpoint path."""
    cfg_file, arm_ov = CFG_MAP[arm]
    cfg = f"config/concealworld/{cfg_file}"
    out_dir = _run_dir(arm, seed, tag)
    cmd = [
        "python",
        "train.py",
        "--config",
        cfg,
        "--no_pbar",
        f"seed={seed}",
        f"trainer.out_dir={out_dir}",
    ] + arm_ov + (overrides or [])
    _run(cmd, cwd=REMOTE_REPO)
    vol.commit()
    ptr = os.path.join(out_dir, "latest_ckpt")
    ckpt = open(ptr).read().strip() if os.path.isfile(ptr) else None
    print(f"[modal] arm={arm} seed={seed} tag={tag} latest_ckpt={ckpt}", flush=True)
    return {"arm": arm, "seed": seed, "out_dir": out_dir, "ckpt": ckpt}


@app.function(gpu=GPU, volumes={OUT: vol}, timeout=60 * 60 * 2, retries=2)
def run_probe(arm: str, seed: int, n_eval: int = 6000, tag: str = ""):
    """Probe a trained arm's frozen checkpoint. Returns the parsed results dict."""
    vol.reload()
    out_dir = _run_dir(arm, seed, tag)
    ckpt = open(os.path.join(out_dir, "latest_ckpt")).read().strip()
    cfgs = glob.glob(os.path.join(out_dir, "*", "materialized_config.yaml"))
    assert cfgs, f"no materialized_config under {out_dir}"
    cmd = [
        "python", "/root/experiments/concealworld/probe.py",
        "--config", cfgs[0],
        "--ckpt", ckpt,
        "--arm", arm,
        "--seed", str(seed),
        "--out", out_dir,
        "--repo", REMOTE_REPO,
        "--n-eval", str(n_eval),
    ]
    _run(cmd, cwd=REMOTE_REPO)
    vol.commit()
    res_path = os.path.join(out_dir, f"probe_{arm}_seed{seed}.json")
    with open(res_path) as f:
        result = json.load(f)
    result["tag"] = tag
    return result


def _save_local(result, tag=""):
    variant = tag or "2a"
    day = datetime.date.today().isoformat()
    d = os.path.join(RESULTS_BASE, day, variant)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"probe_{result['arm']}_seed{result['seed']}.json")
    with open(p, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[local] saved {p}")


@app.local_entrypoint()
def probe(arm: str = "gpt", seed: int = 1234, n_eval: int = 6000, tag: str = ""):
    _save_local(run_probe.remote(arm, seed, n_eval, tag), tag)


@app.local_entrypoint()
def probe_all(
    seeds: str = "1234,1235,1236",
    arms: str = "gpt,nextlat,mtp,jtp,nextlat_h1",
    n_eval: int = 6000,
    tag: str = "2a",
):
    """Re-probe already-trained checkpoints (no retraining) — e.g. after a probe edit."""
    seed_list = [int(s) for s in seeds.split(",")]
    jobs = [(a, s, n_eval, tag) for a in arms.split(",") for s in seed_list]
    for res in run_probe.starmap(jobs):
        _save_local(res, tag)
        print(f"[reprobe] {res['arm']} seed{res['seed']} eff_rank={res['effective_rank_final']:.1f}")


@app.local_entrypoint()
def smoke(arm: str = "gpt"):
    """Tiny end-to-end run to validate env+datamodule+model+train-loop+probe wiring."""
    overrides = [
        "trainer.train_batches=300",
        "trainer.val_interval=150",
        "trainer.val_batches=10",
        "trainer.log_interval=25",
        "data.n_train=20000",
        "data.n_val=6000",
    ]
    res = train_arm.remote(arm, 1234, overrides, "smoke")
    print("SMOKE RESULT:", res)
    pres = run_probe.remote(arm, 1234, 3000, "smoke")
    _save_local(pres, "smoke")
    print("SMOKE PROBE eff_rank:", pres.get("effective_rank_final"))


@app.local_entrypoint()
def train(arm: str = "gpt", seed: int = 1234, tag: str = ""):
    print(train_arm.remote(arm, seed, None, tag))


@app.local_entrypoint()
def main(
    seeds: str = "1234,1235,1236",
    arms: str = "gpt,nextlat,mtp,jtp,nextlat_h1",
    n_states: int = 8,
    wander_len: int = 48,
    train_batches: int = 12000,
    n_eval: int = 6000,
    tag: str = "2a",
):
    """Full concealworld (2a) sweep: train all (arm x seed) with a UNIFORM step budget
    (matched across arms), then probe all, save locally to docs/results/<date>/<tag>/.

    n_states (K) / wander_len (L) are the HEADROOM knobs: if GPT saturates on the running
    secret in the wandering window (no gap), raise n_states and/or wander_len and re-run
    under a new tag."""
    seed_list = [int(s) for s in seeds.split(",")]
    arm_list = arms.split(",")
    ov = [
        f"trainer.train_batches={train_batches}",
        f"data.n_states={n_states}",
        f"data.wander_len={wander_len}",
    ]
    train_jobs = [(a, s, ov, tag) for a in arm_list for s in seed_list]
    print(f"[conceal:{tag}] training {len(train_jobs)} arms x seeds "
          f"(K={n_states}, L={wander_len}, steps={train_batches})")
    for r in train_arm.starmap(train_jobs):
        print("[conceal] trained:", r)
    probe_jobs = [(a, s, n_eval, tag) for a in arm_list for s in seed_list]
    print(f"[conceal:{tag}] probing {len(probe_jobs)} checkpoints")
    for res in run_probe.starmap(probe_jobs):
        _save_local(res, tag)
        print(f"[conceal:{tag}] probed {res['arm']} seed{res['seed']} "
              f"eff_rank={res['effective_rank_final']:.1f}")
