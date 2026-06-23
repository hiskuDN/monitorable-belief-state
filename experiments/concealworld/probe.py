"""
Concealworld (2a, v2) probe: extract per-layer frozen hidden states from a trained model and fit
linear + MLP probes for the RUNNING secret S_t at wandering positions, with offset discipline,
per-offset floors, and the gather-vs-carry diagnostic.

Headline read = the RETENTION CURVE: per-offset-trained LINEAR decodability of S_t vs position t
(distance from the start of the running sum). Predict-ahead arms should track S_t throughout
(flat/high); a vanilla GPT can defer the computation to the fork (low mid-sequence, rising near
the act token). We probe primarily at WAIT positions (the current token is uninformative, so any
signal must be a carried register — the purest test), and report all wandering positions too.

Confound controls (per Codex review):
  * per-offset-TRAINED probes for the retention curve (no pooled cross-offset leakage);
  * per-offset chance (1/K) + shuffled-label floors;
  * the running secret is uniform at every offset by construction (leakage_audit), so position
    carries no information about the label;
  * gather-vs-carry diagnostic: decode S_final at the last wandering position vs the fork token —
    a sharp jump = the model assembled the secret at the fork (deferral), not continuous carry.
Linear is primary; MLP is the floor (availability-vs-linearization: if MLP recovers S for GPT but
linear does not, predict-ahead's gain is *linearization* of an otherwise-tangled secret).

Runs inside the NextLat repo (cwd=/root/NextLat on Modal). Invoked by modal_app.run_probe, or:
    cd NextLat && python ../experiments/concealworld/probe.py \
        --config <materialized_config.yaml> --ckpt <ckpt.pt> --arm gpt --seed 1234 --out <dir>
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


def _import_repo(repo):
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _effective_rank(feats, max_n=4000, seed=0):
    rng = np.random.default_rng(seed)
    if feats.shape[0] > max_n:
        feats = feats[rng.choice(feats.shape[0], max_n, replace=False)]
    feats = feats - feats.mean(0, keepdims=True)
    s = np.linalg.svd(feats, compute_uv=False)
    s = s[s > 1e-12]
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def extract_hidden_states(model, records, tokenizer, device, batch_size=128):
    """Run the frozen model; return hs[layer] = list (per record) of np.array [seqlen, D]."""
    import torch

    pad_id = tokenizer.pad_token_id
    token_id_lists = [tokenizer.encode(" ".join(r["tokens"])) for r in records]

    n_layers = None
    hs = None
    order = np.argsort([len(t) for t in token_id_lists])
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


def gather(hs_layer, items):
    X = np.stack([hs_layer[ri][pos] for (ri, pos, y, off) in items]).astype(np.float32)
    y = np.array([y for (_, _, y, _) in items], dtype=np.int64)
    off = np.array([off for (_, _, _, off) in items], dtype=np.int64)
    return X, y, off


def _fit_and_score(Xtr, ytr, Xva, yva, kind, compute_shuffled=True):
    """Fit a probe (linear|mlp); return acc + chance + shuffled-label floor. Multiclass."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
    if kind == "mlp" and Xtr_s.shape[0] > 40000:
        sub = np.random.default_rng(5).choice(Xtr_s.shape[0], 40000, replace=False)
        Xtr_f, ytr_f = Xtr_s[sub], ytr[sub]
    else:
        Xtr_f, ytr_f = Xtr_s, ytr

    def _make():
        if kind == "linear":
            return LogisticRegression(max_iter=2000, C=1.0)
        return MLPClassifier(hidden_layer_sizes=(256,), max_iter=400, early_stopping=True)

    clf = _make().fit(Xtr_f, ytr_f)
    acc = float(accuracy_score(yva, clf.predict(Xva_s)))
    sh_acc = float("nan")
    if compute_shuffled:
        ysh = ytr_f.copy()
        np.random.default_rng(0).shuffle(ysh)
        try:
            sh_acc = float(accuracy_score(yva, _make().fit(Xtr_f, ysh).predict(Xva_s)))
        except Exception:
            sh_acc = float("nan")
    chance = float(np.bincount(yva).max() / len(yva))
    return {"acc": acc, "chance": chance, "shuffled_acc": sh_acc,
            "n_train": int(len(ytr)), "n_val": int(len(yva))}


def build_label_positions(records, split):
    """Per-position running-secret labels (off = position, since r=0).
    S_run: all wandering positions. S_run_wait: wait positions only (purest carry test).
    S_final: S_final at the last wandering position and at the fork (gather-vs-carry)."""
    out = {"S_run": {"train": [], "val": []},
           "S_run_wait": {"train": [], "val": []},
           "S_final": {"train": [], "val": []}}
    for ri, rec in enumerate(records):
        sp = "train" if split[ri] else "val"
        L = rec["L"]
        Sf = int(rec["y_dec"])
        for pos in range(L):
            s = int(rec["S_running"][pos])
            out["S_run"][sp].append((ri, pos, s, pos))
            if rec["is_update_pos"][pos] == 0:
                out["S_run_wait"][sp].append((ri, pos, s, pos))
        out["S_final"][sp].append((ri, L - 1, Sf, L - 1))           # last wandering token
        out["S_final"][sp].append((ri, rec["act_pos"], Sf, rec["act_pos"]))  # fork token
    return out


def retention_curve(hs, items_tr, items_va, layer, min_train=300, min_val=30):
    """Per-offset-TRAINED linear retention: a separate linear probe per position t.
    Returns {off: {acc, shuffled, chance, n}}."""
    tr_by, va_by = defaultdict(list), defaultdict(list)
    for it in items_tr:
        tr_by[it[3]].append(it)
    for it in items_va:
        va_by[it[3]].append(it)
    curve = {}
    for off in sorted(va_by):
        tr, va = tr_by.get(off, []), va_by[off]
        if len(tr) < min_train or len(va) < min_val:
            continue
        Xtr, ytr, _ = gather(hs[layer], tr)
        Xva, yva, _ = gather(hs[layer], va)
        rec = _fit_and_score(Xtr, ytr, Xva, yva, "linear", compute_shuffled=True)
        curve[int(off)] = {"acc": rec["acc"], "shuffled": rec["shuffled_acc"],
                           "chance": rec["chance"], "n": rec["n_val"]}
    return curve


def gather_vs_carry(hs, items_tr, items_va, layer, last_off, fork_off):
    """Decode S_final at the last wandering position vs the fork token (per-position-trained).
    A large positive jump (fork - last) = deferral/gather rather than continuous carry."""
    def _at(off, items):
        return [it for it in items if it[3] == off]
    res = {}
    for name, off in (("last_wander", last_off), ("fork", fork_off)):
        tr, va = _at(off, items_tr), _at(off, items_va)
        if len(tr) < 100 or len(va) < 30:
            res[name] = float("nan")
            continue
        Xtr, ytr, _ = gather(hs[layer], tr)
        Xva, yva, _ = gather(hs[layer], va)
        res[name] = _fit_and_score(Xtr, ytr, Xva, yva, "linear", compute_shuffled=False)["acc"]
    res["jump"] = (res["fork"] - res["last_wander"]
                   if not (np.isnan(res["fork"]) or np.isnan(res["last_wander"])) else float("nan"))
    return res


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
    import lightning as L
    from omegaconf import OmegaConf
    from core_train import initialize_model
    from data.concealworld import generate_records, params_from_config, leakage_audit

    config = OmegaConf.load(args.config)
    params = params_from_config(config)

    records, _, tokenizer = generate_records(args.n_eval, args.probe_seed, params)
    audit = leakage_audit(records, n_states=params["n_states"], raise_on_fail=True)
    print(f"[probe] leakage audit: {audit['failures'] or 'OK'}", flush=True)

    fabric = L.Fabric(devices=1)
    config.trainer.compile = False
    model = initialize_model(fabric, config, tokenizer,
                             initialize_optimizer=False, checkpoint_path=args.ckpt)
    model.eval()
    device = next(model.model.parameters()).device

    hs, n_layers = extract_hidden_states(model, records, tokenizer, device, batch_size=args.batch_size)
    print(f"[probe] extracted {n_layers} layers over {len(records)} trajectories", flush=True)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(records))
    split = np.zeros(len(records), dtype=bool)
    split[perm[: int(0.8 * len(records))]] = True
    label_pos = build_label_positions(records, split)

    rng_sub = np.random.default_rng(123)

    def _cap(items, n):
        if len(items) <= n:
            return items
        return [items[i] for i in rng_sub.choice(len(items), n, replace=False)]

    fin = n_layers - 1
    er_items = label_pos["S_run"]["train"][:6000]
    Xer, _, _ = gather(hs[fin], er_items) if er_items else (np.zeros((2, 2)), None, None)
    eff_rank = _effective_rank(Xer)

    results = {}
    for lab in ("S_run", "S_run_wait"):
        tr = _cap(label_pos[lab]["train"], 40000)
        va = _cap(label_pos[lab]["val"], 20000)
        results[lab] = {"linear": [], "mlp": []}
        if not tr or not va:
            continue
        print(f"[probe] fitting {lab} ({len(tr)} train / {len(va)} val)", flush=True)
        for li in range(n_layers):
            Xtr, ytr, _ = gather(hs[li], tr)
            Xva, yva, _ = gather(hs[li], va)
            for kind in ("linear", "mlp"):
                rec = _fit_and_score(Xtr, ytr, Xva, yva, kind, compute_shuffled=(kind == "linear"))
                rec["layer"] = li
                results[lab][kind].append(rec)
        results[lab]["retention"] = retention_curve(
            hs, label_pos[lab]["train"], label_pos[lab]["val"], fin
        )

    Lw = int(records[0]["L"])
    results["gather_vs_carry"] = gather_vs_carry(
        hs, label_pos["S_final"]["train"], label_pos["S_final"]["val"],
        fin, last_off=Lw - 1, fork_off=int(records[0]["act_pos"]),
    )

    out = {"arm": args.arm, "seed": args.seed, "ckpt": args.ckpt,
           "n_layers": n_layers, "n_eval": len(records), "params": params,
           "leakage_audit": audit, "effective_rank_final": eff_rank, "results": results}
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"probe_{args.arm}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[probe] wrote {out_path}", flush=True)
    for lab in ("S_run", "S_run_wait"):
        if results[lab]["linear"]:
            best = max(r["acc"] for r in results[lab]["linear"])
            print(f"[probe] {args.arm} {lab}: best linear acc={best:.3f}", flush=True)
    gvc = results["gather_vs_carry"]
    print(f"[probe] {args.arm} gather-vs-carry: last={gvc['last_wander']:.3f} "
          f"fork={gvc['fork']:.3f} jump={gvc['jump']:+.3f} eff_rank={eff_rank:.1f}", flush=True)


if __name__ == "__main__":
    main()
