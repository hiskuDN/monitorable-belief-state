"""Render reproducible 'read the model's mind' examples for storyworld: on the SAME held-out
narratives, show what a linear probe decodes as the object's current shelf from GPT vs
NextLat-1step at each step. Safety framing: you are an interpretability monitor who can only
read the hidden state ("where does it think the ball is right now?"). NextLat stays legible
throughout; GPT goes blind mid-story and only reconstructs the answer at the query (too late).

Usage: ./.venv/bin/python experiments/storyworld/render_examples.py docs/results/<date>/story_easy
"""
import glob
import json
import sys


def load(arm, d):
    f = glob.glob(f"{d}/examples_{arm}_seed1234.json")
    return json.load(open(f[0]))["examples"] if f else None


def argmax(p):
    return max(range(len(p)), key=lambda i: p[i])


def render( d):
    G = load("gpt", d)
    N = load("nextlat_h1", d)
    assert G and N, "need examples_gpt_seed1234.json and examples_nextlat_h1_seed1234.json"

    scored = []
    for i, (ge, ne) in enumerate(zip(G, N)):
        gsteps = {s["pos"]: s for s in ge["steps"]}
        nsteps = {s["pos"]: s for s in ne["steps"]}
        common = sorted(set(gsteps) & set(nsteps))
        g_corr = sum(argmax(gsteps[p]["probs"]) == gsteps[p]["true_t"] for p in common) / len(common)
        n_corr = sum(argmax(nsteps[p]["probs"]) == nsteps[p]["true_t"] for p in common) / len(common)
        last = common[-1]
        g_last_ok = argmax(gsteps[last]["probs"]) == gsteps[last]["true_t"]
        n_last_ok = argmax(nsteps[last]["probs"]) == nsteps[last]["true_t"]
        scored.append((n_corr - g_corr, n_last_ok and not g_last_ok, i, ge, ne, gsteps, nsteps, common))
    # surface the sharpest contrasts first
    scored.sort(key=lambda x: (x[1], x[0]), reverse=True)

    out = ["# Storyworld — reading the model's mind (GPT vs NextLat-1step)\n",
           "You are an interpretability monitor. You can't see the model's output — you only get to",
           "read its hidden state with a linear probe and ask **\"which shelf does it think the ball is",
           "on, right now?\"** Below, the same narratives are read off both models. NextLat keeps the",
           "belief legible the whole way; GPT goes blind mid-story and only reconstructs the answer at",
           "the question — by which point a monitor watching it has already lost the thread.\n"]

    def fmt(step):
        s = argmax(step["probs"]); c = step["probs"][s]
        ok = "OK " if s == step["true_t"] else "XX "
        return f"shelf{s} {c*100:>3.0f}% {ok}"

    for rank, (_, sharp, i, ge, ne, gsteps, nsteps, common) in enumerate(scored[:4]):
        out.append(f"\n## Example {rank+1}  (true final shelf = {ge['true_final']})\n")
        out.append("> " + " ".join(ge["tokens"]) + "\n")
        out.append(f"{'pos · token':>22} | {'TRUE':>6} | {'GPT probe':>14} | {'NextLat-1 probe':>16}")
        out.append("-" * 70)
        for p in common:
            tok = gsteps[p]["tok"]
            mark = " (target move)" if gsteps[p]["is_update"] else ""
            out.append(f"{p:>4} · {tok:<14}{'':>0} | shelf{gsteps[p]['true_t']:<1} | "
                       f"{fmt(gsteps[p]):>14} | {fmt(nsteps[p]):>16}{mark}")
        # the query: what each model has ready when asked
        g_f, n_f = ge["fork"], ne["fork"]
        out.append(f"{'QUERY → answer':>22} | shelf{g_f['true']:<1} | "
                   f"{('shelf'+str(argmax(g_f['probs']))+' '+str(round(max(g_f['probs'])*100))+'%'):>14} | "
                   f"{('shelf'+str(argmax(n_f['probs']))+' '+str(round(max(n_f['probs'])*100))+'%'):>16}")
        gc = sum(argmax(gsteps[p]['probs']) == gsteps[p]['true_t'] for p in common) / len(common)
        nc = sum(argmax(nsteps[p]['probs']) == nsteps[p]['true_t'] for p in common) / len(common)
        out.append(f"\nmonitor read-accuracy across the story:  GPT {gc*100:.0f}%   NextLat-1step {nc*100:.0f}%")

    text = "\n".join(out)
    path = f"{d}/mind_reader_examples.md"
    open(path, "w").write(text)
    print(text)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    render(sys.argv[1])
