"""
Experiment 0 probe: extract per-layer frozen hidden states from a trained
gridworld model and fit linear + MLP probes for the expressed and decoupled
safety labels, with strict window discipline and floors.

Runs inside the NextLat repo (cwd=/root/NextLat on Modal) so that `core_train`
and `data.gridworld` import cleanly. Invoked by modal_app.run_probe, or directly:

    cd NextLat && python ../experiments/gridworld/probe_exp0.py \
        --config output/gridworld/gpt_seed1234/materialized_config.yaml \
        --ckpt   <path-to-ckpt.pt> --arm gpt --seed 1234 --out <dir>

Key design points (see docs/experiments/gridworld/exp0.md):
  * y_dec (the secret bit G) is probed ONLY at positions inside the window
    W = {r+d < t < T-d}; we also report accuracy conditioned on (t - r).
  * y_exp_1 / y_exp_k are probed on direction positions where the label is defined.
  * Identical trajectory-level train/val split across arms & seeds (the eval set is
    regenerated deterministically), so arms differ only in their representations.
  * For every (label, layer): chance floor (class prior) + shuffled-label floor.
  * Linear probe is primary; MLP secondary.
"""

import argparse
import json
import os
import sys

import numpy as np


def _import_repo(repo):
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _effective_rank(feats, max_n=4000, seed=0):
    """exp(entropy of normalized singular values) — the paper's compression metric."""
    rng = np.random.default_rng(seed)
    if feats.shape[0] > max_n:
        feats = feats[rng.choice(feats.shape[0], max_n, replace=False)]
    feats = feats - feats.mean(0, keepdims=True)
    s = np.linalg.svd(feats, compute_uv=False)
    s = s[s > 1e-12]
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def extract_hidden_states(model, records, tokenizer, device, batch_size=128):
    """Run the frozen model and return per-layer hidden states aligned to tokens.

    Returns hs[layer] = list (per record) of np.array [seqlen, D].
    Uses one trajectory per row (padded); we only ever read positions < T.
    """
    import torch

    pad_id = tokenizer.pad_token_id
    token_id_lists = [tokenizer.encode(" ".join(r["tokens"])) for r in records]

    n_layers = None
    hs = None
    order = np.argsort([len(t) for t in token_id_lists])  # length-bucket for less padding
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            seqs = [token_id_lists[i] for i in idx]
            maxlen = max(len(s) for s in seqs)
            arr = np.full((len(seqs), maxlen), pad_id, dtype=np.int64)
            for j, s in enumerate(seqs):
                arr[j, : len(s)] = s
            inp = torch.from_numpy(arr).to(device)
            _, all_layers = model.model(inp, return_all_layers=True)
            all_layers = [l.float().cpu().numpy() for l in all_layers]
            if hs is None:
                n_layers = len(all_layers)
                hs = [[None] * len(records) for _ in range(n_layers)]
            for li in range(n_layers):
                for j, i in enumerate(idx):
                    hs[li][int(i)] = all_layers[li][j, : len(seqs[j])]
    return hs, n_layers


def _fit_and_score(Xtr, ytr, Xva, yva, kind, compute_shuffled=True):
    """Fit a probe (linear|mlp), return dict with acc/auroc + shuffled-label floor.
    MLP is the secondary probe; we shrink it and skip its shuffled floor for speed."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
    # MLP = the nonlinear floor ("is the info present but not linear?"). Must be
    # strong enough not to underperform the linear probe (esp. on 81-way y_cell).
    if kind == "mlp" and Xtr_s.shape[0] > 40000:
        sub = np.random.default_rng(5).choice(Xtr_s.shape[0], 40000, replace=False)
        Xtr_mlp, ytr_mlp = Xtr_s[sub], ytr[sub]
    else:
        Xtr_mlp, ytr_mlp = Xtr_s, ytr

    def _make():
        if kind == "linear":
            return LogisticRegression(max_iter=2000, C=1.0)
        return MLPClassifier(hidden_layer_sizes=(256,), max_iter=400, early_stopping=True)

    is_binary = len(np.unique(ytr)) == 2

    def _score(clf, Xt, yt):
        clf.fit(Xt, yt)
        pred = clf.predict(Xva_s)
        acc = float(accuracy_score(yva, pred))
        auroc = float("nan")
        if is_binary:
            try:
                proba = clf.predict_proba(Xva_s)[:, 1]
                auroc = float(roc_auc_score(yva, proba))
            except (ValueError, IndexError):
                auroc = float("nan")
        return acc, auroc

    acc, auroc = _score(_make(), Xtr_mlp if kind == "mlp" else Xtr_s, ytr_mlp if kind == "mlp" else ytr)
    # shuffled-label floor (permute train labels) — linear only (the informative floor)
    if compute_shuffled:
        rng = np.random.default_rng(0)
        ysh = (ytr_mlp if kind == "mlp" else ytr).copy()
        rng.shuffle(ysh)
        try:
            sh_acc, sh_auroc = _score_shuffled(
                _make(), Xtr_mlp if kind == "mlp" else Xtr_s, ysh, Xva_s, yva
            )
        except Exception:
            sh_acc, sh_auroc = float("nan"), float("nan")
    else:
        sh_acc, sh_auroc = float("nan"), float("nan")
    chance = float(np.bincount(yva).max() / len(yva))  # majority-class baseline (multiclass-safe)
    return {
        "acc": acc,
        "auroc": auroc,
        "chance": chance,
        "shuffled_acc": sh_acc,
        "shuffled_auroc": sh_auroc,
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "val_pos_rate": float(np.mean(yva)),
    }


def _score_shuffled(clf, Xtr_s, ysh, Xva_s, yva):
    from sklearn.metrics import accuracy_score, roc_auc_score

    clf.fit(Xtr_s, ysh)
    pred = clf.predict(Xva_s)
    acc = float(accuracy_score(yva, pred))
    try:
        proba = clf.predict_proba(Xva_s)[:, 1]
        auroc = float(roc_auc_score(yva, proba))
    except (ValueError, IndexError):
        auroc = float("nan")
    return acc, auroc


def build_label_positions(records, split):
    """Collect (record_idx, pos) lists per label, tagged train/val by trajectory.

    Returns dict label -> {"train": [(ri,pos,y,off)], "val": [...]}
    off = t - r (distance from secret reveal), used for position conditioning.
    """
    out = {"y_exp_1": {"train": [], "val": []},
           "y_exp_k": {"train": [], "val": []},
           "y_cell": {"train": [], "val": []},
           "y_dec": {"train": [], "val": []}}
    for ri, rec in enumerate(records):
        sp = "train" if split[ri] else "val"
        r = rec["r"]
        for lab in ("y_exp_1", "y_exp_k", "y_cell"):
            arr = rec[lab]
            for pos in range(len(arr)):
                if arr[pos] >= 0:
                    out[lab][sp].append((ri, pos, int(arr[pos]), pos - r))
        win = rec["in_window"]
        for pos in np.nonzero(win)[0]:
            out["y_dec"][sp].append((ri, int(pos), int(rec["y_dec"]), int(pos) - r))
    return out


def gather(hs_layer, items):
    X = np.stack([hs_layer[ri][pos] for (ri, pos, y, off) in items]).astype(np.float32)
    y = np.array([y for (_, _, y, _) in items], dtype=np.int64)
    off = np.array([off for (_, _, _, off) in items], dtype=np.int64)
    return X, y, off


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
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    _import_repo(args.repo)
    import torch
    import lightning as L
    from omegaconf import OmegaConf
    from core_train import initialize_model
    from data.gridworld import (
        generate_records,
        params_from_config,
        leakage_audit,
    )

    config = OmegaConf.load(args.config)
    params = params_from_config(config)

    # Held-out probe eval set (deterministic; independent of training data seed).
    records, traps, tokenizer = generate_records(args.n_eval, args.probe_seed, params)
    audit = leakage_audit(records, raise_on_fail=True)
    print(f"[probe] leakage audit: {audit['failures'] or 'OK'}", flush=True)

    # Frozen model.
    fabric = L.Fabric(devices=1)
    config.trainer.compile = False
    model = initialize_model(
        fabric, config, tokenizer,
        initialize_optimizer=False, checkpoint_path=args.ckpt,
    )
    model.eval()
    device = next(model.model.parameters()).device

    hs, n_layers = extract_hidden_states(
        model, records, tokenizer, device, batch_size=args.batch_size
    )
    print(f"[probe] extracted {n_layers} layers over {len(records)} trajectories", flush=True)

    # Trajectory-level 80/20 split (identical across arms: same records).
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(records))
    n_train = int(0.8 * len(records))
    split = np.zeros(len(records), dtype=bool)
    split[perm[:n_train]] = True

    label_pos = build_label_positions(records, split)

    # Cap positions per label to keep the ~100 probe fits tractable (statistically
    # ample). Position-conditioning below still uses the full val set.
    rng_sub = np.random.default_rng(123)

    def _cap(items, n):
        if len(items) <= n:
            return items
        idx = rng_sub.choice(len(items), n, replace=False)
        return [items[i] for i in idx]

    # effective rank on final layer (in-window positions) — compression sanity
    fin = n_layers - 1
    win_items = label_pos["y_dec"]["train"][:6000]
    Xer, _, _ = gather(hs[fin], win_items) if win_items else (np.zeros((2, 2)), None, None)
    eff_rank = _effective_rank(Xer)

    results = {}
    for lab in ("y_exp_1", "y_exp_k", "y_cell", "y_dec"):
        tr_items = _cap(label_pos[lab]["train"], 40000)
        va_items = _cap(label_pos[lab]["val"], 20000)
        results[lab] = {"linear": [], "mlp": []}
        print(f"[probe] fitting {lab} ({len(tr_items)} train / {len(va_items)} val positions)", flush=True)
        for li in range(n_layers):
            Xtr, ytr, _ = gather(hs[li], tr_items)
            Xva, yva, off_va = gather(hs[li], va_items)
            for kind in ("linear", "mlp"):
                rec = _fit_and_score(Xtr, ytr, Xva, yva, kind, compute_shuffled=(kind == "linear"))
                rec["layer"] = li
                results[lab][kind].append(rec)
            # position-conditioned (decoupled only, linear, one point per offset)
        if lab == "y_dec":
            results[lab]["position_conditioned"] = position_conditioned(
                hs, label_pos["y_dec"], n_layers, fin
            )

    out = {
        "arm": args.arm,
        "seed": args.seed,
        "ckpt": args.ckpt,
        "n_layers": n_layers,
        "n_eval": len(records),
        "params": params,
        "leakage_audit": audit,
        "effective_rank_final": eff_rank,
        "results": results,
    }
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"probe_{args.arm}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[probe] wrote {out_path}", flush=True)
    # brief console summary (linear, primary)
    for lab in ("y_exp_k", "y_cell", "y_dec"):
        accs = [r["acc"] for r in results[lab]["linear"]]
        best = max(accs)
        print(f"[probe] {args.arm} {lab}: best linear acc={best:.3f} eff_rank={eff_rank:.1f}", flush=True)


def position_conditioned(hs, dec_split, n_layers, layer, max_off=64):
    """Decoupled accuracy vs (t - r), linear probe, best layer = final.
    One position per trajectory per offset bin for independence."""
    import numpy as np

    tr = dec_split["train"]
    va = dec_split["val"]
    if len(tr) > 40000:
        sub = np.random.default_rng(7).choice(len(tr), 40000, replace=False)
        tr = [tr[i] for i in sub]
    Xtr = np.stack([hs[layer][ri][pos] for (ri, pos, y, off) in tr]).astype(np.float32)
    ytr = np.array([y for (_, _, y, _) in tr])
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr), ytr)

    # bin val positions by offset
    bins = {}
    for (ri, pos, y, off) in va:
        bins.setdefault(int(off), []).append((ri, pos, y))
    curve = {}
    for off, items in sorted(bins.items()):
        if len(items) < 30:
            continue
        X = scaler.transform(
            np.stack([hs[layer][ri][pos] for (ri, pos, y) in items]).astype(np.float32)
        )
        yv = np.array([y for (_, _, y) in items])
        acc = float((clf.predict(X) == yv).mean())
        curve[int(off)] = {"acc": acc, "n": len(items)}
    return curve


if __name__ == "__main__":
    main()
