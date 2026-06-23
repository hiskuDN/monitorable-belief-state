"""
Aggregate Concealworld (2a, v2) probe results into the decision-relevant views:

  * headline: best-layer LINEAR accuracy for the running secret at WAIT positions (S_run_wait,
    the purest carry test) and at all wandering positions (S_run), per arm, vs chance (1/K);
  * RETENTION CURVE: per-offset linear accuracy vs position t, per arm (the headline read);
  * GATHER-vs-CARRY: S_final decodability at the last wandering token vs the fork token, per arm
    (a large jump = the model deferred/assembled at the fork rather than carrying);
  * availability-vs-linearization: linear vs MLP by layer;
  * effective-rank (compression) comparison.

    python3 experiments/concealworld/analyze.py 2026-06-23/2a
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
DOCS_RESULTS = os.path.join(PROJECT_ROOT, "docs", "results")
_arg = sys.argv[1] if len(sys.argv) > 1 else ""
RESULTS = (_arg if (_arg and os.path.isdir(_arg))
           else os.path.join(DOCS_RESULTS, _arg) if _arg else DOCS_RESULTS)
ARMS = ["gpt", "nextlat", "mtp", "jtp", "nextlat_h1"]
PREDICTIVE = ["nextlat", "mtp", "jtp", "nextlat_h1"]
LABELS = ["S_run_wait", "S_run"]


def load():
    runs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(RESULTS, "probe_*.json"))):
        with open(p) as f:
            r = json.load(f)
        runs[r["arm"]].append(r)
    return runs


def _present(runs, arm):
    return arm in runs and bool(runs[arm])


def _has(runs, arm, label):
    return _present(runs, arm) and runs[arm][0]["results"].get(label, {}).get("linear")


def _agg(runs_for_arm, label, kind, field):
    mats = [[row[field] for row in r["results"][label][kind]] for r in runs_for_arm]
    M = np.array(mats, dtype=float)
    return np.nanmean(M, 0), np.nanstd(M, 0)


def best_layer_acc(runs, arm, label, kind="linear"):
    m, s = _agg(runs[arm], label, kind, "acc")
    li = int(np.nanargmax(m))
    return li, m[li], s[li]


def by_layer_table(runs, label, kind="linear"):
    arms = [a for a in ARMS if _has(runs, a, label)]
    if not arms:
        return ""
    lines = [f"\n### {label} — {kind} probe accuracy, mean±std over seeds"]
    lines.append("layer | " + " | ".join(arms) + " | chance | shuffled")
    acc = {a: _agg(runs[a], label, kind, "acc") for a in arms}
    nL = max(len(acc[a][0]) for a in arms)
    ch, _ = _agg(runs[arms[0]], label, kind, "chance")
    sh, _ = _agg(runs[arms[0]], label, kind, "shuffled_acc")
    for li in range(nL):
        cells = [f"{acc[a][0][li]:.3f}±{acc[a][1][li]:.3f}" if li < len(acc[a][0]) else "—" for a in arms]
        c = f"{ch[li]:.3f}" if li < len(ch) else "—"
        s = f"{sh[li]:.3f}" if li < len(sh) and not np.isnan(sh[li]) else "—"
        lines.append(f"{li} | " + " | ".join(cells) + f" | {c} | {s}")
    return "\n".join(lines)


def headline(runs):
    arms = [a for a in ARMS if _present(runs, a)]
    out = ["\n## Headline (linear probe — best-layer accuracy mean±std @layer)"]
    out.append("label | " + " | ".join(arms) + " | chance")
    for label in LABELS:
        cells = []
        ch = "—"
        for a in arms:
            if not _has(runs, a, label):
                cells.append("—"); continue
            li, m, s = best_layer_acc(runs, a, label)
            cells.append(f"{m:.3f}±{s:.3f}@{li}")
            chm, _ = _agg(runs[a], label, "linear", "chance"); ch = f"{np.nanmean(chm):.3f}"
        out.append(f"{label} | " + " | ".join(cells) + f" | {ch}")
    out.append("\n## Effective rank (final layer, wandering positions) — lower = more compressed")
    for a in arms:
        er = np.array([r["effective_rank_final"] for r in runs[a]])
        out.append(f"{a}: {er.mean():.1f}±{er.std():.1f}")
    return "\n".join(out)


def availability_vs_linearization(runs):
    """If GPT's MLP >> GPT's linear, the secret is present-but-tangled for GPT and predict-ahead's
    gain is linearization rather than availability."""
    out = ["\n## Availability vs linearization (best-layer linear vs MLP, S_run_wait)"]
    for a in [x for x in ARMS if _has(runs, x, "S_run_wait")]:
        _, lin, _ = best_layer_acc(runs, a, "S_run_wait", "linear")
        mlp_m, _ = _agg(runs[a], "S_run_wait", "mlp", "acc")
        mlp = float(np.nanmax(mlp_m))
        out.append(f"{a}: linear {lin:.3f} | MLP {mlp:.3f}  (MLP−linear = {mlp - lin:+.3f})")
    return "\n".join(out)


def gather_vs_carry_text(runs):
    out = ["\n## Gather-vs-carry (S_final decodability: last wandering token → fork token)"]
    for a in [x for x in ARMS if _present(runs, x)]:
        vals = [r["results"].get("gather_vs_carry") for r in runs[a] if r["results"].get("gather_vs_carry")]
        if not vals:
            continue
        last = np.nanmean([v["last_wander"] for v in vals])
        fork = np.nanmean([v["fork"] for v in vals])
        jump = np.nanmean([v["jump"] for v in vals])
        out.append(f"{a}: last={last:.3f} fork={fork:.3f} jump={jump:+.3f}"
                   + ("  (defers/gathers)" if jump > 0.1 else "  (carries)"))
    return "\n".join(out)


def retention_text(runs, label="S_run_wait"):
    out = [f"\n## Retention curve — {label} linear acc vs position t [final layer, per-offset trained]"]
    for a in [x for x in ARMS if _has(runs, x, label)]:
        # aggregate per-offset acc across seeds
        per = defaultdict(list)
        for r in runs[a]:
            for off, d in r["results"][label].get("retention", {}).items():
                per[int(off)].append(d["acc"])
        if not per:
            continue
        offs = sorted(per)
        pts = ", ".join(f"{o}:{np.mean(per[o]):.2f}" for o in offs)
        out.append(f"{a}: {pts}")
    return "\n".join(out)


def decision(runs):
    if not _has(runs, "gpt", "S_run_wait"):
        return "\n## Decision\n(need a gpt run with S_run_wait)"
    _, gpt_m, _ = best_layer_acc(runs, "gpt", "S_run_wait")
    msg = ["\n## Decision (auto, from linear S_run_wait accuracy — confirm by eye)"]
    msg.append(f"GPT best-layer linear acc (running secret, wait positions) = {gpt_m:.3f}")
    deltas = []
    for a in PREDICTIVE:
        if not _has(runs, a, "S_run_wait"):
            continue
        _, m, _ = best_layer_acc(runs, a, "S_run_wait")
        deltas.append(m - gpt_m)
        msg.append(f"  {a}: {m:.3f}  (Δ vs GPT = {m - gpt_m:+.3f})")
    if not deltas:
        return "\n".join(msg + ["(no predict-ahead arms yet)"])
    md = float(np.mean(deltas))
    if gpt_m > 0.9:
        msg.append(f"=> NO HEADROOM: GPT already {gpt_m:.3f}. Raise n_states / wander_len and re-run.")
    elif md > 0.05:
        msg.append(f"=> predict-ahead TRACKS the masked secret better (mean Δ={md:+.3f}): "
                   "monitorable, predict-ahead helps. Greenlight 2b.")
    else:
        msg.append(f"=> no clear gap (mean Δ={md:+.3f}) with GPT off-ceiling: "
                   "predict-ahead doesn't visibly help; inspect retention + gather-vs-carry before 2b.")
    return "\n".join(msg)


def make_plots(runs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[analyze] matplotlib unavailable; skipping plots")
        return
    for label in LABELS:
        arms = [a for a in ARMS if _has(runs, a, label)]
        if not arms:
            continue
        plt.figure(figsize=(6, 4))
        for a in arms:
            m, s = _agg(runs[a], label, "linear", "acc")
            xs = np.arange(len(m))
            plt.plot(xs, m, marker="o", label=a)
            plt.fill_between(xs, m - s, m + s, alpha=0.15)
        plt.xlabel("layer"); plt.ylabel("linear accuracy"); plt.title(f"{label} by layer")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, f"bylayer_{label}.png"), dpi=120); plt.close()
    # retention curves
    for label in LABELS:
        arms = [a for a in ARMS if _has(runs, a, label)]
        if not arms:
            continue
        plt.figure(figsize=(6, 4)); plotted = False
        for a in arms:
            per = defaultdict(list)
            for r in runs[a]:
                for off, d in r["results"][label].get("retention", {}).items():
                    per[int(off)].append(d["acc"])
            if not per:
                continue
            offs = sorted(per)
            plt.plot(offs, [np.mean(per[o]) for o in offs], marker=".", label=a); plotted = True
        if plotted:
            ch = 1.0 / runs[arms[0]][0]["params"]["n_states"]
            plt.axhline(ch, ls="--", c="gray", lw=1, label="chance")
            plt.xlabel("position t"); plt.ylabel("linear acc of running secret")
            plt.title(f"Retention — {label}"); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(RESULTS, f"retention_{label}.png"), dpi=120)
        plt.close()
    print(f"[analyze] wrote plots to {RESULTS}")


def main():
    runs = load()
    if not runs:
        print(f"No results in {RESULTS}. Run the Modal probes first.")
        return
    print(f"Loaded arms: { {a: len(v) for a, v in runs.items()} }")
    parts = [
        "# Concealworld (2a) results\n",
        f"arms/seeds: { {a: [r['seed'] for r in v] for a, v in runs.items()} }",
        headline(runs),
        decision(runs),
        retention_text(runs, "S_run_wait"),
        retention_text(runs, "S_run"),
        gather_vs_carry_text(runs),
        availability_vs_linearization(runs),
    ]
    for label in LABELS:
        parts.append(by_layer_table(runs, label, "linear"))
    parts.append(by_layer_table(runs, "S_run_wait", "mlp"))
    text = "\n".join(p for p in parts if p)
    print(text)
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write(text + "\n")
    make_plots(runs)
    print(f"\n[analyze] wrote {os.path.join(RESULTS, 'summary.md')}")


if __name__ == "__main__":
    main()
