"""Export concealworld 2a probe results -> web/public/data/concealworld.json for the
interactive dashboard (web/, the same React+Vite app as probe.hiskiasdingeto.com).

Arm/seed-agnostic: reads every canonical probe_<arm>_seed<seed>.json under a results dir,
plus any labelled diagnostic files probe_<arm>_seed<seed>_<label>.json (used to trace a
single seed's carry trajectory across training checkpoints).

Usage (from project root, in the venv):
    ./.venv/bin/python experiments/concealworld/export_web.py 2026-06-24/2a_full
"""

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "docs", "results")
OUT = os.path.join(ROOT, "web", "public", "data", "concealworld.json")

LATE_OFF = 16          # late-window threshold (offset >= this)
CARRY_THR = 0.30       # late-window MLP lift over chance above which we call it "carry"
LEARNED_FORK = 0.60    # fork-decode above which the model has learned the commit task


def _late_lift(retention):
    """Mean (MLP acc - chance) over late offsets (>= LATE_OFF)."""
    late = [(v["mlp_acc"], v["chance"]) for o, v in retention.items() if int(o) >= LATE_OFF]
    if not late:
        return 0.0
    return sum(m for m, _ in late) / len(late) - sum(c for _, c in late) / len(late)


def _retention_rows(retention):
    rows = []
    for o in sorted(retention, key=lambda x: int(x)):
        v = retention[o]
        rows.append({"off": int(o), "lin": round(v["acc"], 4),
                     "mlp": round(v["mlp_acc"], 4), "chance": round(v["chance"], 4)})
    return rows


def _mean_retention(seed_rows):
    """Average lin/mlp/chance across seeds, aligned by offset."""
    by_off = {}
    for rows in seed_rows:
        for r in rows:
            d = by_off.setdefault(r["off"], {"lin": [], "mlp": [], "chance": []})
            d["lin"].append(r["lin"]); d["mlp"].append(r["mlp"]); d["chance"].append(r["chance"])
    out = []
    for off in sorted(by_off):
        d = by_off[off]
        out.append({"off": off,
                    "lin": round(sum(d["lin"]) / len(d["lin"]), 4),
                    "mlp": round(sum(d["mlp"]) / len(d["mlp"]), 4),
                    "chance": round(sum(d["chance"]) / len(d["chance"]), 4)})
    return out


def _iter_loss(ckpt_path):
    m = re.search(r"ckpt_iter_(\d+)_([0-9.]+)\.pt", ckpt_path or "")
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: export_web.py <date>/<tag>   e.g. 2026-06-24/2a_full")
    rel = sys.argv[1]
    d = os.path.join(RESULTS, rel)
    canon = sorted(f for f in glob.glob(os.path.join(d, "probe_*_seed*.json"))
                   if not re.search(r"_seed\d+_[a-zA-Z]", os.path.basename(f)))
    if not canon:
        sys.exit(f"no canonical probe_*.json under {d}")

    arms = {}
    cc = []          # competence-vs-carry points (one per arm x seed)
    K = None
    for f in canon:
        rec = json.load(open(f))
        arm, seed = rec["arm"], rec["seed"]
        K = rec["params"]["n_states"]
        ret = rec["results"]["S_run_wait"]["retention"]
        gvc = rec["results"]["gather_vs_carry"]
        rows = _retention_rows(ret)
        lift = _late_lift(ret)
        learned = gvc["fork"] >= LEARNED_FORK
        a = arms.setdefault(arm, {"seeds": {}})
        a["seeds"][str(seed)] = {
            "retention": rows,
            "fork": round(gvc["fork"], 4), "last": round(gvc["last_wander"], 4),
            "jump": round(gvc["jump"], 4), "late_lift": round(lift, 4),
            "learned": learned, "eff_rank": round(rec["effective_rank_final"], 1),
        }
        cc.append({"arm": arm, "seed": seed, "fork": round(gvc["fork"], 4),
                   "late_lift": round(lift, 4), "learned": learned,
                   "carry": bool(learned and lift > CARRY_THR)})

    for arm, a in arms.items():
        a["mean_retention"] = _mean_retention([s["retention"] for s in a["seeds"].values()])
        lifts = [s["late_lift"] for s in a["seeds"].values()]
        learned_n = sum(1 for s in a["seeds"].values() if s["learned"])
        carry_n = sum(1 for s in a["seeds"].values() if s["learned"] and s["late_lift"] > CARRY_THR)
        a["summary"] = {"n_seeds": len(a["seeds"]), "learned": learned_n, "carry": carry_n,
                        "late_lift_mean": round(sum(lifts) / len(lifts), 4)}

    # trajectory: trace one seed across checkpoints (canonical final + labelled diagnostics)
    traj_arm, traj_seed = "nextlat", 1235
    pts = []
    diag = glob.glob(os.path.join(d, f"probe_{traj_arm}_seed{traj_seed}_*.json"))
    canon_f = os.path.join(d, f"probe_{traj_arm}_seed{traj_seed}.json")
    for f in diag + ([canon_f] if os.path.isfile(canon_f) else []):
        rec = json.load(open(f))
        it, loss = _iter_loss(rec.get("ckpt", ""))
        if it is None:
            continue
        gvc = rec["results"]["gather_vs_carry"]
        pts.append({"iter": it, "val_loss": loss,
                    "eff_rank": round(rec["effective_rank_final"], 1),
                    "fork": round(gvc["fork"], 4), "last": round(gvc["last_wander"], 4),
                    "late_lift": round(_late_lift(rec["results"]["S_run_wait"]["retention"]), 4)})
    pts.sort(key=lambda p: p["iter"])
    trajectory = {"arm": traj_arm, "seed": traj_seed, "points": pts} if pts else None

    # per-position example trajectories (K-cell mind window), if dumped
    examples = {}
    for f in sorted(glob.glob(os.path.join(d, "examples_*_seed*.json"))):
        rec = json.load(open(f))
        examples.setdefault(rec["arm"], {})[str(rec["seed"])] = rec["examples"]

    out = {
        "meta": {
            "tag": rel, "K": K, "chance": round(1.0 / K, 4) if K else None,
            "late_off": LATE_OFF, "carry_thr": CARRY_THR, "learned_fork": LEARNED_FORK,
            "note": ("Per-seed read. The task has two task-optimal solutions — carry the "
                     "running secret, or defer-and-gather it at the fork; the predict-ahead "
                     "aux loss tips the balance. Precomputed from frozen checkpoints."),
        },
        "arms": arms,
        "competence_carry": sorted(cc, key=lambda r: (r["arm"], r["seed"])),
        "trajectory": trajectory,
        "examples": examples,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[export] wrote {OUT}")
    print(f"[export] arms={list(arms)}  competence_carry={len(cc)} pts  "
          f"trajectory={'%d pts' % len(pts) if pts else 'none'}  "
          f"examples={sum(len(v) for v in examples.values())} arm-seeds")


if __name__ == "__main__":
    main()
