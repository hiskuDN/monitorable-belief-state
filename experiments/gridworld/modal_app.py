"""
Modal harness for NextLat-for-safety Experiment 0 (the gridworld belief-state study).

Runs the matched GPT vs NextLat gridworld training arms (and probes) on Modal
GPUs, persisting checkpoints/outputs to a Modal Volume. Shared scaffolding
(image, run helper, paths) lives in experiments/common/modal_base.py.

Usage (from the project root, NOT inside NextLat/):
    # quick end-to-end smoke test (tiny, ~minutes):
    python3 -m modal run experiments/gridworld/modal_app.py::smoke

    # train one arm:
    python3 -m modal run experiments/gridworld/modal_app.py::train --arm gpt --seed 1234

    # full v1 fan-out (both arms x seeds) + probes:
    python3 -m modal run experiments/gridworld/modal_app.py::main --seeds 1234,1235,1236

The NextLat repo is mounted at /root/NextLat and the whole experiments/ tree at
/root/experiments (copy=False -> local edits picked up without an image rebuild).
Training runs `python train.py` directly (single GPU, no fabric launcher needed).
"""

import datetime
import glob
import json
import os
import sys

# make `common` importable when run via `modal run experiments/gridworld/modal_app.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import modal  # noqa: E402

from common.modal_base import (  # noqa: E402
    GPU,
    OUT,
    PROJECT_ROOT,
    REMOTE_REPO,
    RESULTS_BASE,
    build_image,
    run as _run,
)

app = modal.App("nextlat-exp0", image=build_image())
vol = modal.Volume.from_name("nextlat-exp0", create_if_missing=True)


# arm label -> (config file, extra CLI overrides). nextlat_h1 = ablated control.
CFG_MAP = {
    "gpt": ("gpt_gridworld.yaml", []),
    "nextlat": ("nextlat_gridworld.yaml", []),
    "nextlat_h1": ("nextlat_gridworld.yaml", ["model.mtp_horizon=1"]),
    "mtp": ("mtp_gridworld.yaml", []),
    "jtp": ("jtp_gridworld.yaml", []),
}


def _run_dir(arm, seed, tag):
    return f"{OUT}/{arm}{('_' + tag) if tag else ''}_seed{seed}"


@app.function(gpu=GPU, volumes={OUT: vol}, timeout=60 * 60 * 6, retries=2)
def train_arm(arm: str, seed: int, overrides: list[str] | None = None, tag: str = ""):
    """Train one arm (gpt|nextlat|nextlat_h1) at a given seed. Returns the out_dir."""
    cfg_file, arm_ov = CFG_MAP[arm]
    cfg = f"config/gridworld/{cfg_file}"
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
        "python", "/root/experiments/gridworld/probe_exp0.py",
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
    variant = tag or "v1"
    day = datetime.date.today().isoformat()
    d = os.path.join(RESULTS_BASE, day, variant)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"probe_{result['arm']}_seed{result['seed']}.json")
    with open(p, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[local] saved {p}")


@app.local_entrypoint()
def probe(arm: str = "gpt", seed: int = 1234, n_eval: int = 6000):
    _save_local(run_probe.remote(arm, seed, n_eval))


@app.local_entrypoint()
def probe_all(
    seeds: str = "1234,1235,1236",
    arms: str = "gpt,nextlat",
    n_eval: int = 6000,
    tag: str = "",
):
    """Re-probe already-trained checkpoints (no retraining) — e.g. after adding a
    new probe target or strengthening the probe. `tag` selects the run namespace."""
    seed_list = [int(s) for s in seeds.split(",")]
    jobs = [(a, s, n_eval, tag) for a in arms.split(",") for s in seed_list]
    for res in run_probe.starmap(jobs):
        _save_local(res, tag)
        print(f"[reprobe] {res['arm']} seed{res['seed']} eff_rank={res['effective_rank_final']:.1f}")


@app.function(gpu=GPU, volumes={OUT: vol}, timeout=60 * 60, retries=1)
def export_demo_remote(arms: str, tag: str, seed: int, n_fit: int, n_demo: int):
    """Run experiments/gridworld/export_demo.py against trained checkpoints on the
    volume and return the demo.json text (the web app ships only this, not checkpoints)."""
    vol.reload()
    out_path = "/tmp/demo.json"
    cmd = [
        "python", "/root/experiments/gridworld/export_demo.py",
        "--arms", arms, "--repo", REMOTE_REPO, "--resolve-dir", OUT,
        "--tag", tag, "--seed", str(seed), "--out", out_path,
        "--n-fit", str(n_fit), "--n-demo", str(n_demo),
    ]
    _run(cmd, cwd=REMOTE_REPO)
    with open(out_path) as f:
        return f.read()


@app.local_entrypoint()
def export_demo(
    arms: str = "gpt,nextlat",
    tag: str = "v2b",
    seed: int = 1234,
    n_fit: int = 4000,
    n_demo: int = 24,
):
    """Build web/public/data/demo.json (belief-decode + design-space panel) from
    trained checkpoints. e.g.: modal run experiments/gridworld/modal_app.py::export_demo"""
    data = export_demo_remote.remote(arms, tag, seed, n_fit, n_demo)
    dest = os.path.join(PROJECT_ROOT, "web", "public", "data", "demo.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        f.write(data)
    print(f"[demo] saved {dest} ({len(data) / 1e6:.2f} MB)")


@app.function(gpu=GPU, volumes={OUT: vol}, timeout=60 * 60 * 2, retries=2)
def run_step2(arm: str, seed: int = 1234, tag: str = "v3_L8d512", n_eval: int = 6000):
    """Step 2 (does compactness buy a monitor anything): PCA/sample-eff/shift on a
    trained checkpoint. Returns the parsed results dict."""
    vol.reload()
    out_dir = _run_dir(arm, seed, tag)
    ckpt = open(os.path.join(out_dir, "latest_ckpt")).read().strip()
    cfgs = glob.glob(os.path.join(out_dir, "*", "materialized_config.yaml"))
    assert cfgs, f"no materialized_config under {out_dir}"
    cmd = [
        "python", "/root/experiments/gridworld/step2_compactness.py",
        "--config", cfgs[0], "--ckpt", ckpt, "--arm", arm, "--seed", str(seed),
        "--out", out_dir, "--repo", REMOTE_REPO, "--n-eval", str(n_eval),
    ]
    _run(cmd, cwd=REMOTE_REPO)
    vol.commit()
    with open(os.path.join(out_dir, f"step2_{arm}_seed{seed}.json")) as f:
        return json.load(f)


@app.local_entrypoint()
def step2(
    arms: str = "gpt,nextlat,mtp,jtp,nextlat_h1",
    seed: int = 1234,
    tag: str = "v3_L8d512",
    n_eval: int = 6000,
):
    """Exp 0 STEP 2 — does NextLat's compactness buy a monitor anything? Runs the
    PCA (low-dim monitor) / sample-efficiency / distribution-shift battery across the
    arm panel on the v3 scale checkpoints. Saves to docs/results/<date>/<tag>_step2/."""
    jobs = [(a, seed, tag, n_eval) for a in arms.split(",")]
    day = datetime.date.today().isoformat()
    d = os.path.join(RESULTS_BASE, day, f"{tag}_step2")
    os.makedirs(d, exist_ok=True)
    for res in run_step2.starmap(jobs):
        with open(os.path.join(d, f"step2_{res['arm']}_seed{res['seed']}.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"[step2] {res['arm']}: full={res['full_acc']} rank={res['eff_rank']} "
              f"k95={res['k95']} shift_drop={res['shift']['drop']}")


@app.local_entrypoint()
def smoke(arm: str = "gpt"):
    """Tiny end-to-end run to validate env+datamodule+model+train-loop wiring."""
    overrides = [
        "trainer.train_batches=300",
        "trainer.val_interval=150",
        "trainer.val_batches=10",
        "trainer.log_interval=25",
        "data.n_train=20000",
        "data.n_val=6000",
    ]
    res = train_arm.remote(arm, 1234, overrides)
    print("SMOKE RESULT:", res)
    # immediately probe the smoke checkpoint to validate the probe path too
    pres = run_probe.remote(arm, 1234, 3000)
    _save_local(pres)
    print("SMOKE PROBE eff_rank:", pres.get("effective_rank_final"))


@app.local_entrypoint()
def train(arm: str = "gpt", seed: int = 1234):
    print(train_arm.remote(arm, seed))


@app.local_entrypoint()
def main(
    seeds: str = "1234,1235,1236",
    arms: str = "gpt,nextlat",
    train_batches: int = 10000,
    n_eval: int = 6000,
):
    """Full Experiment 0 sweep: train all (arm x seed), then probe all, save locally."""
    seed_list = [int(s) for s in seeds.split(",")]
    arm_list = arms.split(",")
    ov = [f"trainer.train_batches={train_batches}"]
    train_jobs = [(a, s, ov) for a in arm_list for s in seed_list]
    print(f"[exp0] training {len(train_jobs)} arms x seeds @ {train_batches} steps")
    for r in train_arm.starmap(train_jobs):
        print("[exp0] trained:", r)
    probe_jobs = [(a, s, n_eval) for a in arm_list for s in seed_list]
    print(f"[exp0] probing {len(probe_jobs)} checkpoints")
    for res in run_probe.starmap(probe_jobs):
        _save_local(res)
        print(f"[exp0] probed {res['arm']} seed{res['seed']} eff_rank={res['effective_rank_final']:.1f}")


@app.local_entrypoint()
def main_v2b(
    seeds: str = "1234,1235,1236",
    arms: str = "gpt,nextlat",
    n_layer: int = 4,
    n_embd: int = 256,
    trap_frac: float = 0.12,
    train_batches: int = 8000,
    n_eval: int = 6000,
    tag: str = "v2b",
):
    """Partial-observability Exp 0 — the decisive test. No absolute start anchor;
    interleaved wall-pattern observations, so position must be integrated over the
    trajectory (a genuine belief-state task that should not saturate). Denser traps
    => more informative observations (localization achievable, not a floor)."""
    seed_list = [int(s) for s in seeds.split(",")]
    arm_list = arms.split(",")
    ov = [
        f"trainer.train_batches={train_batches}",
        f"model.n_layer={n_layer}",
        f"model.n_embd={n_embd}",
        "data.partial_obs=true",
        f"data.trap_frac={trap_frac}",
    ]
    train_jobs = [(a, s, ov, tag) for a in arm_list for s in seed_list]
    print(f"[exp0:{tag}] partial-obs training {len(train_jobs)} runs (L={n_layer}, d={n_embd}, traps={trap_frac})")
    for r in train_arm.starmap(train_jobs):
        print("[exp0] trained:", r)
    probe_jobs = [(a, s, n_eval, tag) for a in arm_list for s in seed_list]
    print(f"[exp0:{tag}] probing {len(probe_jobs)} checkpoints")
    for res in run_probe.starmap(probe_jobs):
        _save_local(res, tag)
        print(f"[exp0:{tag}] probed {res['arm']} seed{res['seed']} eff_rank={res['effective_rank_final']:.1f}")


@app.local_entrypoint()
def main_scale(
    seeds: str = "1234,1235,1236",
    arms: str = "gpt,nextlat,mtp,jtp,nextlat_h1",
    n_layer: int = 8,
    n_embd: int = 512,
    trap_frac: float = 0.12,
    train_batches: int = 12000,
    n_eval: int = 6000,
    tag: str = "v3_L8d512",
):
    """Exp 0 STEP 1 — scale-robustness check. Re-run the v2b partial-obs design-
    space sweep at a LARGER backbone (default 8-layer/d=512, ~4x v2b's L4/d256) so
    the GPT baseline is no longer capacity-starved. The task is held IDENTICAL to
    v2b (same partial_obs, trap_frac, min/max steps) and so is the training recipe
    (grad_accum=1, effective_batch_size=256) — ONLY model size + training length
    scale up, so any change in the gap is attributable to capacity, not the task or
    optimizer. Runs on A100-80GB (the wider d ~4x's the unroll activation memory).
    Question: does the monitorability gap (GPT ~0.71 vs aux ~0.95 on 81-way
    position) survive when the baseline has room to integrate position itself?
    train_batches bumped to 12k (~10 epochs) so GPT is genuinely converged — an
    undertrained baseline would masquerade as a capacity-starved one. Verify task
    sanity (val loss + path validity) from the logs before trusting the probe."""
    seed_list = [int(s) for s in seeds.split(",")]
    arm_list = arms.split(",")
    ov = [
        f"trainer.train_batches={train_batches}",
        f"model.n_layer={n_layer}",
        f"model.n_embd={n_embd}",
        "data.partial_obs=true",
        f"data.trap_frac={trap_frac}",
    ]
    train_jobs = [(a, s, ov, tag) for a in arm_list for s in seed_list]
    print(f"[exp0:{tag}] SCALE training {len(train_jobs)} runs "
          f"(L={n_layer}, d={n_embd}, grad_accum=1, steps={train_batches})")
    # Plain starmap consumption (the proven path). Resilience comes from retries=2
    # on the function itself, which self-heals the transient container deaths that
    # bit the first attempt. (Avoid starmap(return_exceptions=True) here: it hits a
    # Modal 1.5 / py3.14 async-generator bug that fails to dispatch the calls.)
    for r in train_arm.starmap(train_jobs):
        print("[exp0] trained:", r)
    probe_jobs = [(a, s, n_eval, tag) for a in arm_list for s in seed_list]
    print(f"[exp0:{tag}] probing {len(probe_jobs)} checkpoints")
    for res in run_probe.starmap(probe_jobs):
        _save_local(res, tag)
        print(f"[exp0:{tag}] probed {res['arm']} seed{res['seed']} eff_rank={res['effective_rank_final']:.1f}")


@app.local_entrypoint()
def main_v2(
    seeds: str = "1234,1235,1236",
    arms: str = "gpt,nextlat",
    n_layer: int = 2,
    n_embd: int = 128,
    train_batches: int = 6000,
    n_eval: int = 6000,
    tag: str = "v2",
):
    """Capacity-limited Exp 0 (headroom variant). Smaller model so the baseline
    does not saturate; namespaced by `tag` so v1 results are preserved.
    Add arms='gpt,nextlat,nextlat_h1' to include the mtp_horizon=1 control."""
    seed_list = [int(s) for s in seeds.split(",")]
    arm_list = arms.split(",")
    ov = [
        f"trainer.train_batches={train_batches}",
        f"model.n_layer={n_layer}",
        f"model.n_embd={n_embd}",
    ]
    train_jobs = [(a, s, ov, tag) for a in arm_list for s in seed_list]
    print(f"[exp0:{tag}] training {len(train_jobs)} runs (n_layer={n_layer}, n_embd={n_embd})")
    for r in train_arm.starmap(train_jobs):
        print("[exp0] trained:", r)
    probe_jobs = [(a, s, n_eval, tag) for a in arm_list for s in seed_list]
    print(f"[exp0:{tag}] probing {len(probe_jobs)} checkpoints")
    for res in run_probe.starmap(probe_jobs):
        _save_local(res, tag)
        print(f"[exp0:{tag}] probed {res['arm']} seed{res['seed']} eff_rank={res['effective_rank_final']:.1f}")
