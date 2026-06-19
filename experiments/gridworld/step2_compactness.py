"""
Experiment 0 — Step 2: does NextLat's compactness buy a *monitor* anything?

At matched monitorability (all predictive arms decode position ~0.95), compare arms
on the levers a real monitor cares about, at each arm's best layer:

  (A) low-dimensional monitors — decode accuracy vs #PCA components (cheaper monitor)
  (B) sample-efficiency        — decode accuracy vs #labeled examples (cheaper to build)
  (C) robustness under shift   — in-dist probe evaluated on shifted trap density

Hypothesis to falsify: compactness is safety-neutral (NextLat no cheaper / more robust
to monitor than diffuse-but-decodable MTP). Runs on a frozen checkpoint; reuses the
probe_exp0 helpers. Invoked by modal_app.run_step2, or directly like probe_exp0.py.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_exp0 import (  # noqa: E402
    _effective_rank,
    extract_hidden_states,
    build_label_positions,
    gather,
)

K_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]            # PCA dims to try
N_LIST = [50, 100, 250, 500, 1000, 2000, 5000, 15000]  # probe train sizes


def _cap(items, n, seed=1):
    if len(items) <= n:
        return items
    idx = np.random.default_rng(seed).choice(len(items), n, replace=False)
    return [items[i] for i in idx]


def _acc(clf, X, y):
    return float((clf.predict(X) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repo", default="/root/NextLat")
    ap.add_argument("--n-eval", type=int, default=6000)
    ap.add_argument("--probe-seed", type=int, default=99991)
    ap.add_argument("--shift-fracs", default="0.06,0.18")  # model trained at 0.12
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import lightning as L
    from omegaconf import OmegaConf
    from core_train import initialize_model
    from data.gridworld import generate_records, params_from_config
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA

    config = OmegaConf.load(args.config)
    config.trainer.compile = False
    params = params_from_config(config)
    n_cells = int(params["grid_size"]) ** 2

    records, traps, tokenizer = generate_records(args.n_eval, args.probe_seed, params)
    fabric = L.Fabric(devices=1)
    model = initialize_model(fabric, config, tokenizer, initialize_optimizer=False, checkpoint_path=args.ckpt)
    model.eval()
    device = next(model.model.parameters()).device

    hs, n_layers = extract_hidden_states(model, records, tokenizer, device)
    print(f"[step2] {args.arm}: extracted {n_layers} layers", flush=True)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(records))
    split = np.zeros(len(records), dtype=bool)
    split[perm[: int(0.8 * len(records))]] = True
    lp = build_label_positions(records, split)
    tr = _cap(lp["y_cell"]["train"], 40000)
    va = _cap(lp["y_cell"]["val"], 20000)

    # best layer by full-dim linear decode
    layer_accs = []
    for li in range(n_layers):
        Xt, yt, _ = gather(hs[li], tr)
        Xv, yv, _ = gather(hs[li], va)
        sc = StandardScaler().fit(Xt)
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xt), yt)
        layer_accs.append(round(_acc(clf, sc.transform(Xv), yv), 4))
    best = int(np.argmax(layer_accs))
    print(f"[step2] {args.arm}: best layer {best} (acc {layer_accs[best]})", flush=True)

    Xtr, ytr, _ = gather(hs[best], tr)
    Xva, yva, _ = gather(hs[best], va)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
    ref = LogisticRegression(max_iter=2000).fit(Xtr_s, ytr)   # full-dim reference probe
    full_acc = _acc(ref, Xva_s, yva)
    eff_rank = round(_effective_rank(Xtr), 1)

    # (A) low-dimensional monitors: PCA -> linear probe
    pca = PCA(n_components=min(max(K_LIST), Xtr_s.shape[1])).fit(Xtr_s)
    Ztr, Zva = pca.transform(Xtr_s), pca.transform(Xva_s)
    pca_curve = []
    for k in K_LIST:
        if k > Ztr.shape[1]:
            continue
        clf = LogisticRegression(max_iter=1000).fit(Ztr[:, :k], ytr)
        pca_curve.append({"k": k, "acc": round(_acc(clf, Zva[:, :k], yva), 4)})
    # PCs needed to reach 95% of full-dim accuracy
    target = 0.95 * full_acc
    k95 = next((c["k"] for c in pca_curve if c["acc"] >= target), None)

    # (B) sample-efficiency: full-dim probe trained on N examples
    samp_curve = []
    for N in N_LIST:
        n = min(N, Xtr_s.shape[0])
        idx = np.random.default_rng(7).choice(Xtr_s.shape[0], n, replace=False)
        clf = LogisticRegression(max_iter=1000).fit(Xtr_s[idx], ytr[idx])
        samp_curve.append({"n": int(n), "acc": round(_acc(clf, Xva_s, yva), 4)})

    # (C) robustness under a behavioral shift that keeps the MODEL in-distribution
    # (so this isolates the *monitor's* generalization, not model OOD): train the
    # monitor on SHORT trajectories, deploy it on LONG ones. Lower drop = more robust
    # monitor. (A trap-density/map shift instead makes the model fully OOD -> all arms
    # floor, no signal.)
    traj_len = {ri: int(np.sum(np.asarray(rec["y_cell"]) >= 0)) for ri, rec in enumerate(records)}
    med_len = int(np.median(list(traj_len.values())))
    Xs, ys, _ = gather(hs[best], _cap([it for it in tr if traj_len[it[0]] < med_len], 40000))
    sc_s = StandardScaler().fit(Xs)
    clf_s = LogisticRegression(max_iter=1500).fit(sc_s.transform(Xs), ys)
    Xsv, ysv, _ = gather(hs[best], _cap([it for it in va if traj_len[it[0]] < med_len], 20000))
    Xlv, ylv, _ = gather(hs[best], _cap([it for it in va if traj_len[it[0]] >= med_len], 20000))
    acc_in = _acc(clf_s, sc_s.transform(Xsv), ysv)
    acc_shift = _acc(clf_s, sc_s.transform(Xlv), ylv)
    shift = {"split": "train_short_eval_long", "median_len": med_len,
             "acc_in_dist": round(acc_in, 4), "acc_shifted": round(acc_shift, 4),
             "drop": round(acc_in - acc_shift, 4)}

    out = {
        "arm": args.arm, "seed": args.seed, "n_cells": n_cells, "best_layer": best,
        "full_acc": round(full_acc, 4), "eff_rank": eff_rank, "layer_accs": layer_accs,
        "pca_curve": pca_curve, "k95": k95, "sample_curve": samp_curve, "shift": shift,
        "train_trap_frac": float(params["trap_frac"]),
    }
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, f"step2_{args.arm}_seed{args.seed}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[step2] {args.arm}: full {full_acc:.3f} rank {eff_rank} | k95={k95} | "
          f"shift_drop={shift['drop']} -> {p}", flush=True)


if __name__ == "__main__":
    main()
