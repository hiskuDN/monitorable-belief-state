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

app = modal.App("nextlat-story", image=build_image())
vol = modal.Volume.from_name("nextlat-story", create_if_missing=True)


# arm label -> (config file, extra CLI overrides). nextlat_h1 = ablated control.
CFG_MAP = {
    "gpt": ("gpt_storyworld.yaml", []),
    "nextlat": ("nextlat_storyworld.yaml", []),
    "nextlat_h1": ("nextlat_storyworld.yaml", ["model.mtp_horizon=1"]),
    "mtp": ("mtp_storyworld.yaml", []),
    "jtp": ("jtp_storyworld.yaml", []),
}


def _run_dir(arm, seed, tag):
    return f"{OUT}/{arm}{('_' + tag) if tag else ''}_seed{seed}"


@app.function(gpu=GPU, volumes={OUT: vol}, timeout=60 * 60 * 6, retries=2)
def train_arm(arm: str, seed: int, overrides: list[str] | None = None, tag: str = ""):
    """Train one arm at a given seed. Returns the out_dir + latest checkpoint path."""
    cfg_file, arm_ov = CFG_MAP[arm]
    cfg = f"config/storyworld/{cfg_file}"
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
def run_probe(arm: str, seed: int, n_eval: int = 6000, tag: str = "", ckpt_override: str = ""):
    """Probe a trained arm's frozen checkpoint. Returns the parsed results dict.

    ckpt_override: probe a SPECIFIC checkpoint instead of latest_ckpt (e.g. a best-val
    checkpoint for a seed whose training diverged) — diagnostic, used by `probe_at`."""
    vol.reload()
    out_dir = _run_dir(arm, seed, tag)
    ckpt = ckpt_override.strip() or open(os.path.join(out_dir, "latest_ckpt")).read().strip()
    cfgs = glob.glob(os.path.join(out_dir, "*", "materialized_config.yaml"))
    assert cfgs, f"no materialized_config under {out_dir}"
    cmd = [
        "python", "/root/experiments/storyworld/probe.py",
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


@app.function(gpu=GPU, volumes={OUT: vol}, timeout=60 * 60 * 2, retries=2)
def run_examples(arm: str, seed: int, tag: str = "", n_examples: int = 4):
    """Dump per-position decoded probabilities for a few example sequences (probe --mode
    examples). Lightweight (linear probes only). Returns the parsed examples dict."""
    vol.reload()
    out_dir = _run_dir(arm, seed, tag)
    ckpt = open(os.path.join(out_dir, "latest_ckpt")).read().strip()
    cfgs = glob.glob(os.path.join(out_dir, "*", "materialized_config.yaml"))
    assert cfgs, f"no materialized_config under {out_dir}"
    cmd = [
        "python", "/root/experiments/storyworld/probe.py",
        "--config", cfgs[0], "--ckpt", ckpt, "--arm", arm, "--seed", str(seed),
        "--out", out_dir, "--repo", REMOTE_REPO,
        "--mode", "examples", "--n-examples", str(n_examples),
    ]
    _run(cmd, cwd=REMOTE_REPO)
    vol.commit()
    with open(os.path.join(out_dir, f"examples_{arm}_seed{seed}.json")) as f:
        return json.load(f)


@app.local_entrypoint()
def dump_examples(
    seeds: str = "1234,1235,1236",
    arms: str = "gpt,nextlat,mtp,jtp,nextlat_h1",
    tag: str = "2a_full",
    n_examples: int = 4,
):
    """Fan out the examples dump over arms x seeds; save examples_<arm>_seed<seed>.json
    locally under docs/results/<date>/<tag>/ for the web export to merge."""
    seed_list = [int(s) for s in seeds.split(",")]
    jobs = [(a, s, tag, n_examples) for a in arms.split(",") for s in seed_list]
    day = datetime.date.today().isoformat()
    d = os.path.join(RESULTS_BASE, day, tag)
    os.makedirs(d, exist_ok=True)
    for res in run_examples.starmap(jobs):
        p = os.path.join(d, f"examples_{res['arm']}_seed{res['seed']}.json")
        with open(p, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[examples] {res['arm']} seed{res['seed']}: {len(res['examples'])} trajectories")


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


@app.function(gpu=GPU, volumes={OUT: vol}, timeout=60 * 60, retries=2)
def run_features(arm: str, seed: int, tag: str = "", n_eval: int = 8000):
    """Dump frozen last-layer wandering features + S_t labels (probe --mode features) and
    return the .npz bytes for the offline price-of-concealment study."""
    vol.reload()
    out_dir = _run_dir(arm, seed, tag)
    ckpt = open(os.path.join(out_dir, "latest_ckpt")).read().strip()
    cfgs = glob.glob(os.path.join(out_dir, "*", "materialized_config.yaml"))
    assert cfgs, f"no materialized_config under {out_dir}"
    cmd = [
        "python", "/root/experiments/storyworld/probe.py",
        "--config", cfgs[0], "--ckpt", ckpt, "--arm", arm, "--seed", str(seed),
        "--out", out_dir, "--repo", REMOTE_REPO, "--n-eval", str(n_eval), "--mode", "features",
    ]
    _run(cmd, cwd=REMOTE_REPO)
    with open(os.path.join(out_dir, f"features_{arm}_seed{seed}.npz"), "rb") as f:
        return f.read()


@app.local_entrypoint()
def dump_features(arms: str = "gpt,nextlat_h1", seeds: str = "1234", tag: str = "2b_l0.0", n_eval: int = 8000):
    """Pull frozen-feature .npz files (one per arm x seed) locally for the offline study."""
    day = datetime.date.today().isoformat()
    d = os.path.join(RESULTS_BASE, day, f"{tag}_features")
    os.makedirs(d, exist_ok=True)
    jobs = [(a, int(s), tag, n_eval) for a in arms.split(",") for s in seeds.split(",")]
    for (a, s, _, _), data in zip(jobs, run_features.starmap(jobs)):
        p = os.path.join(d, f"features_{a}_seed{s}.npz")
        with open(p, "wb") as f:
            f.write(data)
        print(f"[features] wrote {p} ({len(data)//1024} KB)")


@app.local_entrypoint()
def probe_at(arm: str, seed: int, ckpt: str, tag: str = "2a_full",
             label: str = "bestckpt", n_eval: int = 6000):
    """Diagnostic: probe a SPECIFIC checkpoint (not latest_ckpt) and save to a labelled
    file so it doesn't clobber the canonical probe_<arm>_seed<seed>.json. Used to test
    whether a diverged-final seed carries at its best-val checkpoint."""
    res = run_probe.remote(arm, seed, n_eval, tag, ckpt)
    day = datetime.date.today().isoformat()
    d = os.path.join(RESULTS_BASE, day, tag)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"probe_{arm}_seed{seed}_{label}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=2)
    g = res["results"]["gather_vs_carry"]
    ret = res["results"]["S_run_wait"]["retention"]
    late = [(v["mlp_acc"], v["chance"]) for o, v in ret.items() if int(o) >= 16]
    lift = (sum(m for m, _ in late) - sum(c for _, c in late)) / max(1, len(late))
    print(f"[local] saved {p}")
    print(f"[probe_at] {arm} seed{seed} ckpt={ckpt}\n"
          f"  gather-vs-carry: last={g['last_wander']:.3f} fork={g['fork']:.3f} jump={g['jump']:+.3f}\n"
          f"  late-window (off>=16) MLP lift over chance = {lift:+.2f}")


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
    arms: str = "gpt,nextlat,nextlat_h1,mtp,jtp",
    train_batches: int = 20000,
    n_states: int = 4,
    n_distractors: int = 1,
    n_move: int = 6,
    max_delta: int = 1,
    coref_prob: float = 0.0,
    target_density: float = 0.5,
    n_eval: int = 6000,
    tag: str = "story_easy",
):
    """BLOCKING storyworld sweep: train all (arm x seed) then probe all, saving to
    docs/results/<date>/<tag>/. Runs in one tracked job (auto-notified on completion); keep
    the client connected (starmap is cancelled on disconnect — use ::launch + --detach for
    fire-and-forget). Reports answer_acc (capability gate) per checkpoint."""
    seed_list = [int(s) for s in seeds.split(",")]
    arm_list = arms.split(",")
    ov = [
        f"trainer.train_batches={train_batches}",
        f"data.n_states={n_states}", f"data.n_distractors={n_distractors}",
        f"data.n_move={n_move}", f"data.max_delta={max_delta}",
        f"data.coref_prob={coref_prob}", f"data.target_density={target_density}",
    ]
    train_jobs = [(a, s, ov, tag) for a in arm_list for s in seed_list]
    print(f"[story:{tag}] training {len(train_jobs)} arms x seeds "
          f"(K={n_states}, n_move={n_move}, n_distractors={n_distractors}, coref={coref_prob}, steps={train_batches})")
    for r in train_arm.starmap(train_jobs):
        print("[story] trained:", r)
    probe_jobs = [(a, s, n_eval, tag) for a in arm_list for s in seed_list]
    print(f"[story:{tag}] probing {len(probe_jobs)} checkpoints")
    for res in run_probe.starmap(probe_jobs):
        _save_local(res, tag)
        print(f"[story:{tag}] probed {res['arm']} seed{res['seed']} "
              f"answer_acc={res.get('answer_acc', float('nan')):.2f} eff_rank={res['effective_rank_final']:.1f}")


@app.local_entrypoint()
def launch(
    seeds: str = "1234,1235,1236",
    arms: str = "gpt,nextlat,nextlat_h1,mtp,jtp",
    train_batches: int = 32000,
    n_states: int = 0,         # 0 => config default (6); else override K (headroom knob)
    n_distractors: int = -1,   # -1 => config default; else override (headroom knob)
    n_move: int = 0,           # 0 => config default; else override (headroom knob)
    max_delta: int = 0,        # 0 => config default; else override (easing knob)
    coref_prob: float = -1.0,  # <0 => config default; else override (easing knob)
    target_density: float = -1.0,  # <0 => config default; else override
    tag: str = "story_v1",
):
    """Fire-and-forget TRAINING via .spawn() — survives client disconnect with `modal run
    --detach`. Each train_arm commits its checkpoint to the volume. Probe on reconnect:
        modal run experiments/storyworld/modal_app.py::probe_all --arms <...> --seeds <...> --tag <tag>
    Storyworld env params come from the configs; only override knobs if a value is set."""
    seed_list = [int(s) for s in seeds.split(",")]
    ov = [f"trainer.train_batches={train_batches}"]
    if n_states:
        ov.append(f"data.n_states={n_states}")
    if n_distractors >= 0:
        ov.append(f"data.n_distractors={n_distractors}")
    if n_move:
        ov.append(f"data.n_move={n_move}")
    if max_delta:
        ov.append(f"data.max_delta={max_delta}")
    if coref_prob >= 0:
        ov.append(f"data.coref_prob={coref_prob}")
    if target_density >= 0:
        ov.append(f"data.target_density={target_density}")
    spawned = []
    for a in arms.split(","):
        for s in seed_list:
            fc = train_arm.spawn(a, s, ov, tag)
            spawned.append((a, s, fc.object_id))
            print(f"[launch:{tag}] spawned train_arm({a}, seed={s}) -> {fc.object_id}")
    print(f"[launch:{tag}] {len(spawned)} detached training jobs (steps={train_batches}). "
          f"When done, probe with:\n"
          f"  modal run experiments/storyworld/modal_app.py::probe_all "
          f"--arms {arms} --seeds {seeds} --tag {tag}")


@app.local_entrypoint()
def pilot(
    arm: str = "gpt", seed: int = 1234, train_batches: int = 20000,
    n_states: int = 4, n_distractors: int = 1, n_move: int = 6,
    max_delta: int = 1, coref_prob: float = 0.0, target_density: float = 0.5,
    tag: str = "story_pilot", n_eval: int = 6000,
):
    """Learnability/headroom gate (build-order step 3): train ONE arm at an eased config
    (blocking) then probe, and report answer accuracy at the query. Ease until a GPT answers
    well, then add complexity back before the full arm comparison."""
    ov = [
        f"trainer.train_batches={train_batches}",
        f"data.n_states={n_states}", f"data.n_distractors={n_distractors}",
        f"data.n_move={n_move}", f"data.max_delta={max_delta}",
        f"data.coref_prob={coref_prob}", f"data.target_density={target_density}",
    ]
    print(f"[pilot:{tag}] training {arm} seed{seed}: K={n_states} n_move={n_move} "
          f"n_distractors={n_distractors} max_delta={max_delta} coref={coref_prob} steps={train_batches}")
    print(train_arm.remote(arm, seed, ov, tag))
    res = run_probe.remote(arm, seed, n_eval, tag)
    _save_local(res, tag)
    print(f"[pilot:{tag}] {arm} seed{seed} ANSWER_ACC={res.get('answer_acc'):.3f} "
          f"(chance={1.0/n_states:.3f}) eff_rank={res['effective_rank_final']:.1f}")


@app.local_entrypoint()
def smoke_adv(arm: str = "nextlat_h1", lam: float = 0.5):
    """Tiny GPU run validating the adversarial-probe wiring end-to-end (loss assembles,
    adv_loss logged, no crash), then a probe. Not detached."""
    ov = [
        "trainer.train_batches=400", "trainer.val_interval=200", "trainer.val_batches=10",
        "trainer.log_interval=50", "data.n_train=20000", "data.n_val=6000",
        "data.n_states=4", "data.wander_len=24", "data.update_density=0.3",
        f"model.lambda_adv={lam}", "model.adv_n_states=4",
    ]
    res = train_arm.remote(arm, 1234, ov, "smoke_adv")
    print("SMOKE_ADV TRAIN:", res)
    pres = run_probe.remote(arm, 1234, 3000, "smoke_adv")
    _save_local(pres, "smoke_adv")
    print("SMOKE_ADV PROBE eff_rank:", pres.get("effective_rank_final"))


@app.local_entrypoint()
def launch_adv(
    seeds: str = "1234,1235,1236",
    lambdas: str = "0.0,0.3,1.0",
    arms: str = "gpt,nextlat_h1",
    train_batches: int = 32000,
    n_states: int = 4,
    wander_len: int = 24,
    update_density: float = 0.3,
    tag_prefix: str = "2b",
):
    """Concealworld 2b: adversarial probe-evasion pressure sweep. For each (arm, lambda_adv)
    spawn all seeds DETACHED (run with `modal run --detach`). lambda_adv=0 is the control
    (no adversarial head; identical to 2a). Each (arm, lambda) gets its own tag so run dirs
    don't collide; probe each tag separately on reconnect:
        probe_all --arms gpt,nextlat_h1 --seeds <seeds> --tag 2b_l<lambda>
    """
    seed_list = [int(s) for s in seeds.split(",")]
    spawned = []
    for a in arms.split(","):
        for lam in lambdas.split(","):
            tag = f"{tag_prefix}_l{lam}"
            ov = [
                f"trainer.train_batches={train_batches}",
                f"data.n_states={n_states}",
                f"data.wander_len={wander_len}",
                f"data.update_density={update_density}",
                f"model.lambda_adv={lam}",
                f"model.adv_n_states={n_states}",
            ]
            for s in seed_list:
                fc = train_arm.spawn(a, s, ov, tag)
                spawned.append((a, lam, s, fc.object_id))
                print(f"[launch_adv:{tag}] {a} seed{s} lambda_adv={lam} -> {fc.object_id}")
    tags = sorted({f"{tag_prefix}_l{lam}" for lam in lambdas.split(',')})
    print(f"[launch_adv] {len(spawned)} detached jobs. Probe each tag on reconnect:")
    for t in tags:
        print(f"  modal run experiments/concealworld/modal_app.py::probe_all "
              f"--arms {arms} --seeds {seeds} --tag {t}")
