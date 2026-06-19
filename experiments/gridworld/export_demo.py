"""
Export data for the static web demo (docs/research-overview.md showpiece):
decode the belief state (81-way grid position) per step from frozen models on
partial-observability gridworld trajectories, so the site can render side-by-side
"belief heatmaps" (GPT muddy vs NextLat sharp) plus the design-space panel
(monitorability vs compactness across the arm panel).

Runs inside the NextLat repo on Modal (cwd=/root/NextLat), like probe_exp0.py.
The web app ships ONLY the emitted demo.json — never the checkpoints.

    python /root/experiments/gridworld/export_demo.py \
        --arms gpt,nextlat --repo /root/NextLat \
        --resolve-dir /outputs --tag v2b \
        --out web/public/data/demo.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for probe_exp0
from probe_exp0 import (  # noqa: E402
    _effective_rank,
    extract_hidden_states,
    build_label_positions,
    gather,
)

# Arms we render as belief heatmaps (need per-step decode); others are scatter-only.
HEATMAP_ARMS = {"gpt", "nextlat"}
EXP_NAME = {  # checkpoint subdir name per arm (matches training experiment_name)
    "gpt": "GPT", "nextlat": "NextLat", "mtp": "MTP", "jtp": "JTP",
    "nextlat_h1": "NextLat",
}


def _round(a, nd=4):
    return [round(float(x), nd) for x in a]


def _fit_cell_probe(hs_layer, items):
    """Fit an 81-way LogisticRegression position probe on one layer's states.
    Returns (scaler, clf, classes) — predict_proba is remapped to full 81 later."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    X, y, _ = gather(hs_layer, items)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X), y)
    return scaler, clf


def _acc(scaler, clf, hs_layer, items):
    X, y, _ = gather(hs_layer, items)
    return float((clf.predict(scaler.transform(X)) == y).mean())


def _proba_full(scaler, clf, x_row, n_cells):
    """predict_proba for one state, remapped to a dense length-n_cells vector."""
    p = clf.predict_proba(scaler.transform(x_row[None]))[0]
    full = np.zeros(n_cells, dtype=np.float64)
    full[clf.classes_] = p
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="gpt,nextlat")
    ap.add_argument("--repo", default="/root/NextLat")
    ap.add_argument("--resolve-dir", default="/outputs", help="volume dir holding <arm>_<tag>_seed<seed> run dirs")
    ap.add_argument("--tag", default="v2b")
    ap.add_argument("--seed", type=int, default=1234, help="which trained seed's checkpoint to load")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-fit", type=int, default=4000, help="trajectories to fit the probe on")
    ap.add_argument("--n-demo", type=int, default=24, help="trajectories to visualize")
    ap.add_argument("--fit-seed", type=int, default=99991)
    ap.add_argument("--demo-seed", type=int, default=4242)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import lightning as L
    from omegaconf import OmegaConf
    from core_train import initialize_model
    from data.gridworld import generate_records, params_from_config

    arms = args.arms.split(",")

    def run_dir(arm):
        return os.path.join(args.resolve_dir, f"{arm}_{args.tag}_seed{args.seed}")

    def resolve(arm):
        d = run_dir(arm)
        ckpt = open(os.path.join(d, "latest_ckpt")).read().strip()
        import glob
        cfg = glob.glob(os.path.join(d, "*", "materialized_config.yaml"))[0]
        return cfg, ckpt

    # One shared eval set (same trajectories across arms) — params from the first arm's config.
    cfg0, _ = resolve(arms[0])
    params = params_from_config(OmegaConf.load(cfg0))
    grid_size = int(params["grid_size"])
    n_cells = grid_size * grid_size

    fit_records, traps, tokenizer = generate_records(args.n_fit, args.fit_seed, params)
    demo_records, _, _ = generate_records(args.n_demo, args.demo_seed, params)
    print(f"[demo] grid {grid_size}x{grid_size}, {len(fit_records)} fit / {len(demo_records)} demo trajectories", flush=True)

    # 80/20 split of the fit set for measuring probe accuracy by layer.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(fit_records))
    split = np.zeros(len(fit_records), dtype=bool)
    split[perm[: int(0.8 * len(fit_records))]] = True
    fit_lp = build_label_positions(fit_records, split)
    demo_lp = build_label_positions(demo_records, np.zeros(len(demo_records), dtype=bool))  # all "val"

    def _cap(items, n, seed=1):
        if len(items) <= n:
            return items
        idx = np.random.default_rng(seed).choice(len(items), n, replace=False)
        return [items[i] for i in idx]

    tr_items = _cap(fit_lp["y_cell"]["train"], 40000)
    va_items = _cap(fit_lp["y_cell"]["val"], 20000)

    arms_out = {}
    demo_beliefs = {}  # arm -> {ri -> [{pos,true,argmax,belief(top-k)}]}
    for arm in arms:
        cfg, ckpt = resolve(arm)
        config = OmegaConf.load(cfg)
        config.trainer.compile = False
        fabric = L.Fabric(devices=1)
        model = initialize_model(fabric, config, tokenizer, initialize_optimizer=False, checkpoint_path=ckpt)
        model.eval()
        device = next(model.model.parameters()).device

        hs_fit, n_layers = extract_hidden_states(model, fit_records, tokenizer, device)
        # by-layer y_cell accuracy + best layer
        by_layer, scalers, clfs = [], {}, {}
        for li in range(n_layers):
            sc, clf = _fit_cell_probe(hs_fit[li], tr_items)
            acc = _acc(sc, clf, hs_fit[li], va_items)
            by_layer.append(round(acc, 4))
            scalers[li], clfs[li] = sc, clf
        best = int(np.argmax(by_layer))
        # effective rank on the final layer (in-window positions, like the probe)
        Xer, _, _ = gather(hs_fit[n_layers - 1], _cap(fit_lp["y_dec"]["train"], 6000))
        eff_rank = round(_effective_rank(Xer), 1)
        arms_out[arm] = {
            "by_layer_acc": by_layer, "best_layer": best,
            "best_acc": by_layer[best], "eff_rank": eff_rank, "n_layers": n_layers,
        }
        print(f"[demo] {arm}: best y_cell acc={by_layer[best]:.3f}@{best} eff_rank={eff_rank} layers={n_layers}", flush=True)

        if True:  # decode per-step demo beliefs for every arm (UI lets you pick any two)
            hs_demo, _ = extract_hidden_states(model, demo_records, tokenizer, device)
            sc, clf = scalers[best], clfs[best]
            per_traj = {}
            for (ri, pos, y, off) in demo_lp["y_cell"]["val"]:
                full = _proba_full(sc, clf, hs_demo[best][ri][pos], n_cells)
                # store top-8 cells to keep payload small; UI renders sparse heatmap
                top = np.argsort(full)[::-1][:8]
                per_traj.setdefault(ri, []).append({
                    "pos": int(pos), "true": int(y), "pred": int(full.argmax()),
                    "top": [[int(c), round(float(full[c]), 3)] for c in top],
                })
            demo_beliefs[arm] = per_traj

    # Assemble demo trajectories (tokens + true path + both models' beliefs).
    def trap_cells():
        t = traps
        out = []
        for c in (t if hasattr(t, "__iter__") else []):
            if isinstance(c, (tuple, list)) and len(c) == 2:
                out.append(int(c[0]) * grid_size + int(c[1]))
            else:
                out.append(int(c))
        return sorted(set(out))

    trajectories = []
    for ri, rec in enumerate(demo_records):
        per_arm = {a: {s["pos"]: s for s in demo_beliefs.get(a, {}).get(ri, [])} for a in arms}
        positions = sorted(set().union(*[set(d) for d in per_arm.values()])) if per_arm else []
        steps = []
        for pos in positions:
            entry = {"pos": pos}
            for a in arms:  # true cell from whichever arm has this position
                if pos in per_arm[a]:
                    entry["true"] = per_arm[a][pos]["true"]
                    break
            for a in arms:  # each arm's top-k decoded belief at this step
                entry[a] = per_arm[a].get(pos, {}).get("top")
            steps.append(entry)
        trajectories.append({
            "tokens": rec["tokens"],
            "goal": int(rec.get("goal", -1)) if "goal" in rec else None,
            "steps": steps,
        })

    out = {
        "meta": {
            "grid_size": grid_size, "n_cells": n_cells, "tag": args.tag,
            "trap_cells": trap_cells(),
            "note": "Belief = 81-way position decoded by a linear probe on frozen hidden states. "
                    "Partial observability: position is never given, only local wall-patterns + moves.",
        },
        "arms": arms_out,
        "trajectories": trajectories,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    sz = os.path.getsize(args.out) / 1e6
    print(f"[demo] wrote {args.out} ({sz:.2f} MB, {len(trajectories)} trajectories)", flush=True)


if __name__ == "__main__":
    main()
