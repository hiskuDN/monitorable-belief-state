"""
Aggregate Experiment 0 probe results (experiments/results/probe_*.json) into the
decision-relevant views:

  * by-layer table of probe accuracy/AUROC, GPT vs NextLat, linear (primary) and
    MLP (secondary), aggregated mean +/- std across seeds, with chance + shuffled
    floors;
  * headline: best-layer linear AUROC for y_exp_k (expressed) and y_dec-in-W
    (decoupled), per arm;
  * effective-rank (compression) comparison;
  * position-conditioned y_dec accuracy vs (t - r).

Writes <results-dir>/summary.md and, if matplotlib is available, PNG plots.
Runs locally (numpy + stdlib; matplotlib optional).

    python3 experiments/gridworld/analyze.py 2026-06-19/v3_L8d512
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))  # experiments/gridworld
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root
DOCS_RESULTS = os.path.join(PROJECT_ROOT, "docs", "results")
# CLI arg: a path, or a subpath under docs/results (e.g. "2026-06-17/v2b").
_arg = sys.argv[1] if len(sys.argv) > 1 else ""
RESULTS = (
    _arg if (_arg and os.path.isdir(_arg))
    else os.path.join(DOCS_RESULTS, _arg) if _arg
    else DOCS_RESULTS
)
ARMS = ["gpt", "nextlat", "mtp", "jtp", "nextlat_h1"]  # design-space sweep + ablated control


def load():
    runs = defaultdict(list)  # arm -> [result dicts over seeds]
    for p in sorted(glob.glob(os.path.join(RESULTS, "probe_*.json"))):
        with open(p) as f:
            r = json.load(f)
        runs[r["arm"]].append(r)
    return runs


def _agg(runs_for_arm, label, kind, field):
    """Return per-layer mean,std arrays over seeds for results[label][kind][layer][field]."""
    mats = []
    for r in runs_for_arm:
        series = r["results"][label][kind]
        mats.append([row[field] for row in series])
    M = np.array(mats, dtype=float)  # [seeds, layers]
    return np.nanmean(M, 0), np.nanstd(M, 0)


def fmt(m, s):
    return f"{m:.3f}±{s:.3f}"


def by_layer_table(runs, label, kind="linear"):
    lines = [f"\n### {label} — {kind} probe (acc | AUROC), mean±std over seeds"]
    header = "layer | " + " | ".join(
        f"{a} acc | {a} auroc" for a in ARMS if a in runs
    ) + " | chance | shuffled_auroc"
    lines.append(header)
    arms_present = [a for a in ARMS if a in runs]
    acc = {a: _agg(runs[a], label, kind, "acc") for a in arms_present}
    auc = {a: _agg(runs[a], label, kind, "auroc") for a in arms_present}
    # Arms can have different layer counts (e.g. MTP's shared trunk = n_layer-1
    # blocks). Render ragged: iterate to the max, blank cells past an arm's depth.
    n_layers = max(len(acc[a][0]) for a in arms_present)
    ch_m, _ = _agg(runs[arms_present[0]], label, kind, "chance")
    sh_m, _ = _agg(runs[arms_present[0]], label, kind, "shuffled_auroc")
    for li in range(n_layers):
        cells = []
        for a in arms_present:
            if li < len(acc[a][0]):
                cells.append(f"{fmt(acc[a][0][li], acc[a][1][li])} | {fmt(auc[a][0][li], auc[a][1][li])}")
            else:
                cells.append("— | —")
        ch = f"{ch_m[li]:.3f}" if li < len(ch_m) else "—"
        sh = f"{sh_m[li]:.3f}" if li < len(sh_m) else "—"
        lines.append(f"{li} | " + " | ".join(cells) + f" | {ch} | {sh}")
    return "\n".join(lines)


def best_layer_auroc(runs, arm, label, kind="linear"):
    m, s = _agg(runs[arm], label, kind, "auroc")
    li = int(np.nanargmax(m))
    return li, m[li], s[li]


def best_layer_acc(runs, arm, label, kind="linear"):
    m, s = _agg(runs[arm], label, kind, "acc")
    li = int(np.nanargmax(m))
    return li, m[li], s[li]


def headline(runs):
    arms_present = [a for a in ARMS if a in runs]
    out = ["\n## Headline (linear probe — primary)"]
    out.append("label | " + " | ".join(arms_present) + "  (best-layer AUROC mean±std @layer)")
    for label in ("y_exp_1", "y_exp_k", "y_dec"):
        cells = []
        for a in arms_present:
            li, m, s = best_layer_auroc(runs, a, label)
            cells.append(f"{m:.3f}±{s:.3f}@{li}")
        out.append(f"{label} | " + " | ".join(cells))
    # y_cell is multiclass (81-way exact position) — report ACCURACY, the headroom metric
    out.append("\n## y_cell (exact position, 81-way) — best-layer ACCURACY mean±std @layer")
    if any("y_cell" in runs[a][0]["results"] for a in arms_present):
        out.append("arm | best linear acc | chance | per-layer linear acc")
        for a in arms_present:
            if "y_cell" not in runs[a][0]["results"]:
                continue
            li, m, s = best_layer_acc(runs, a, "y_cell")
            ch, _ = _agg(runs[a], "y_cell", "linear", "chance")
            per_layer, _ = _agg(runs[a], "y_cell", "linear", "acc")
            series = " ".join(f"{v:.2f}" for v in per_layer)
            out.append(f"{a} | {m:.3f}±{s:.3f}@{li} | {ch[0]:.3f} | {series}")
    # effective rank
    out.append("\n## Effective rank (final layer) — lower = more compressed")
    for a in arms_present:
        er = np.array([r["effective_rank_final"] for r in runs[a]])
        out.append(f"{a}: {er.mean():.1f}±{er.std():.1f}")
    return "\n".join(out)


def decision(runs):
    """Apply the agenda's go/no-go criteria to the linear-probe headline."""
    if "gpt" not in runs or "nextlat" not in runs:
        return "\n## Decision\n(need both gpt and nextlat runs)"
    def bestauc(a, lab):
        return best_layer_auroc(runs, a, lab)[1]
    exp_g, exp_n = bestauc("gpt", "y_exp_k"), bestauc("nextlat", "y_exp_k")
    dec_g, dec_n = bestauc("gpt", "y_dec"), bestauc("nextlat", "y_dec")
    msg = ["\n## Decision (auto, from linear AUROC — confirm by eye)"]
    msg.append(f"expressed y_exp_k: NextLat {exp_n:.3f} vs GPT {exp_g:.3f}  (Δ={exp_n-exp_g:+.3f})")
    msg.append(f"decoupled y_dec-in-W: NextLat {dec_n:.3f} vs GPT {dec_g:.3f}  (Δ={dec_n-dec_g:+.3f})")
    if exp_n <= exp_g + 0.01:
        msg.append("=> NextLat NOT better on expressed: core premise looks FALSE (reclassify as capability).")
    elif dec_n <= dec_g + 0.02:
        msg.append("=> framing HOLDS: NextLat better on expressed, ties/loses on decoupled. Greenlight WS2.")
    else:
        msg.append("=> NextLat better on BOTH: compression-deletes-decoupled worry FALSIFIED; widen scope.")
    return "\n".join(msg)


def position_curve_text(runs):
    out = ["\n## Decoupled y_dec accuracy vs (t - r) [final layer, linear]"]
    arms_present = [a for a in ARMS if a in runs]
    for a in arms_present:
        pc = runs[a][0]["results"]["y_dec"].get("position_conditioned", {})
        if not pc:
            continue
        offs = sorted(int(o) for o in pc.keys())
        pts = ", ".join(f"{o}:{pc[str(o)]['acc']:.2f}" for o in offs)
        out.append(f"{a}: {pts}")
    return "\n".join(out)


def make_plots(runs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[analyze] matplotlib unavailable; skipping plots")
        return
    arms_present = [a for a in ARMS if a in runs]
    # by-layer AUROC for the two headline labels
    for label in ("y_exp_k", "y_dec"):
        plt.figure(figsize=(6, 4))
        for a in arms_present:
            m, s = _agg(runs[a], label, "linear", "auroc")
            xs = np.arange(len(m))
            plt.plot(xs, m, marker="o", label=a)
            plt.fill_between(xs, m - s, m + s, alpha=0.15)
        plt.axhline(0.5, ls="--", c="gray", lw=1, label="chance")
        plt.xlabel("layer"); plt.ylabel("linear AUROC"); plt.title(f"{label} by layer")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, f"bylayer_{label}.png"), dpi=120)
        plt.close()
    # position-conditioned y_dec
    plt.figure(figsize=(6, 4))
    for a in arms_present:
        pc = runs[a][0]["results"]["y_dec"].get("position_conditioned", {})
        if not pc:
            continue
        offs = sorted(int(o) for o in pc.keys())
        plt.plot(offs, [pc[str(o)]["acc"] for o in offs], marker=".", label=a)
    plt.axhline(0.5, ls="--", c="gray", lw=1)
    plt.xlabel("t - r (distance from secret reveal)"); plt.ylabel("y_dec acc")
    plt.title("Decoupled retention across window W"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "position_conditioned_ydec.png"), dpi=120)
    plt.close()
    print(f"[analyze] wrote plots to {RESULTS}")


def main():
    runs = load()
    if not runs:
        print(f"No results in {RESULTS}. Run the Modal probes first.")
        return
    print(f"Loaded arms: { {a: len(v) for a, v in runs.items()} }")
    parts = [
        "# Experiment 0 results\n",
        f"arms/seeds: { {a: [r['seed'] for r in v] for a, v in runs.items()} }",
        headline(runs),
        decision(runs),
        position_curve_text(runs),
    ]
    for label in ("y_exp_1", "y_exp_k", "y_cell", "y_dec"):
        parts.append(by_layer_table(runs, label, "linear"))
    for label in ("y_exp_k", "y_dec"):
        parts.append(by_layer_table(runs, label, "mlp"))
    text = "\n".join(parts)
    print(text)
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write(text + "\n")
    make_plots(runs)
    print(f"\n[analyze] wrote {os.path.join(RESULTS, 'summary.md')}")


if __name__ == "__main__":
    main()
