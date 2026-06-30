"""Export storyworld mind-reader examples -> web/public/data/storyworld.json for Demo 3.

Merges the GPT and NextLat-1step example dumps (same held-out narratives) into per-token
records carrying both models' linear-probe belief over the K shelves, so the web player can
sweep the narrative and show, side by side, what a monitor reads off each model's mind.

Usage: ./.venv/bin/python experiments/storyworld/export_web.py docs/results/<date>/story_easy
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "web", "public", "data", "storyworld.json")


def load(arm, d):
    fs = glob.glob(f"{d}/examples_{arm}_seed*.json")
    return json.load(open(fs[0]))["examples"] if fs else None


def argmax(p):
    return max(range(len(p)), key=lambda i: p[i])


def main():
    d = sys.argv[1]
    G, N = load("gpt", d), load("nextlat_h1", d)
    assert G and N, f"need examples_gpt/nextlat_h1 under {d}"
    K = len(G[0]["steps"][0]["probs"])

    examples = []
    for ge, ne in zip(G, N):
        gby = {s["pos"]: s for s in ge["steps"]}
        nby = {s["pos"]: s for s in ne["steps"]}
        common = sorted(set(gby) & set(nby))
        steps = [{
            "pos": p, "tok": gby[p]["tok"], "is_update": gby[p]["is_update"],
            "true_t": gby[p]["true_t"], "gpt": gby[p]["probs"], "nl": nby[p]["probs"],
        } for p in common]
        g_acc = sum(argmax(s["gpt"]) == s["true_t"] for s in steps) / len(steps)
        n_acc = sum(argmax(s["nl"]) == s["true_t"] for s in steps) / len(steps)
        examples.append({
            "tokens": ge["tokens"], "true_final": ge["true_final"], "act_pos": ge["act_pos"],
            "steps": steps,
            "fork": {"true": ge["fork"]["true"], "gpt": ge["fork"]["probs"], "nl": ne["fork"]["probs"]},
            "gpt_acc": round(g_acc, 3), "nl_acc": round(n_acc, 3),
        })
    # sharpest contrast first (NextLat carries, GPT defers)
    examples.sort(key=lambda e: e["nl_acc"] - e["gpt_acc"], reverse=True)

    out = {
        "meta": {
            "K": K, "chance": round(1.0 / K, 3),
            "note": ("Linear probe decoding the target object's current shelf from each frozen "
                     "model's hidden state, per token, on held-out narratives. Same task; both "
                     "models answer the query correctly. Precomputed; no model runs in your browser."),
        },
        "examples": examples,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"[export] wrote {OUT}: {len(examples)} narratives, K={K}")
    print("  gpt_acc / nl_acc per narrative:", [(e["gpt_acc"], e["nl_acc"]) for e in examples])


if __name__ == "__main__":
    main()
