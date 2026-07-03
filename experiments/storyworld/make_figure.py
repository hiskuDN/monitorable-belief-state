"""Render the storyworld blog figure (SVG) from the real example dumps: a single frozen
mid-story frame showing what a linear probe reads off GPT vs NextLat-1step at the same token.
NextLat is sharp on the true shelf (green, gold-outlined); GPT is confidently on a wrong
shelf (red). Reproducible and self-contained (no browser), matching web/public/figures style.

Usage: ./.venv/bin/python experiments/storyworld/make_figure.py docs/results/<date>/story_easy
Writes: web/public/figures/storyworld-demo.svg
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "web", "public", "figures", "storyworld-demo.svg")

GREEN = "#4fd1c5"
RED = "#f2754f"
GOLD = "#ffd166"
DIM = "#2b3350"
INK = "#0a0d14"


def load(arm, d):
    fs = glob.glob(f"{d}/examples_{arm}_seed*.json")
    return json.load(open(fs[0]))["examples"] if fs else None


def argmax(p):
    return max(range(len(p)), key=lambda i: p[i])


def pick(G, N):
    """Choose the narrative + token where GPT is most confidently WRONG while NextLat is
    sharply RIGHT, away from the reveal head and the query."""
    best = None
    for ei, (ge, ne) in enumerate(zip(G, N)):
        gby = {s["pos"]: s for s in ge["steps"]}
        nby = {s["pos"]: s for s in ne["steps"]}
        for p in sorted(set(gby) & set(nby)):
            g, n = gby[p], nby[p]
            if p < 12:  # skip the opening reveal, before any target move
                continue
            t = g["true_t"]
            if argmax(n["probs"]) != t or argmax(g["probs"]) == t:
                continue
            score = n["probs"][t] * g["probs"][argmax(g["probs"])]
            if best is None or score > best[0]:
                best = (score, ei, p, ge, ne, g, n)
    return best


def shelf_panel(x0, y0, probs, true_t, accent, cell_w=46, cell_h=90, gap=9):
    """A row of K vertical bars; height/opacity ~ prob; true shelf gold-outlined; argmax ringed."""
    am = argmax(probs)
    parts = []
    for k, p in enumerate(probs):
        cx = x0 + k * (cell_w + gap)
        is_true = k == true_t
        fill = GREEN if is_true else (RED if k == am else DIM)
        # cell frame
        parts.append(
            f'<rect x="{cx}" y="{y0}" width="{cell_w}" height="{cell_h}" rx="7" '
            f'fill="{INK}" stroke="{"#2b3350" if not is_true else GOLD}" '
            f'stroke-width="{1 if not is_true else 2}"/>'
        )
        # fill from the bottom
        fh = max(4, p * cell_h)
        op = 0.22 + 0.6 * p
        parts.append(
            f'<rect x="{cx}" y="{y0 + cell_h - fh}" width="{cell_w}" height="{fh:.1f}" '
            f'rx="6" fill="{fill}" opacity="{op:.2f}"/>'
        )
        # ring the argmax (the probe's actual read)
        if k == am:
            parts.append(
                f'<rect x="{cx - 2}" y="{y0 - 2}" width="{cell_w + 4}" height="{cell_h + 4}" '
                f'rx="8" fill="none" stroke="{accent}" stroke-width="2.2"/>'
            )
        # shelf index + pct
        parts.append(
            f'<text x="{cx + cell_w/2:.0f}" y="{y0 + cell_h/2 + 6:.0f}" text-anchor="middle" '
            f'font-size="18" font-weight="700" fill="#e7ecff">{k}</text>'
        )
        parts.append(
            f'<text x="{cx + cell_w - 5:.0f}" y="{y0 + cell_h - 6:.0f}" text-anchor="end" '
            f'font-size="10" fill="#8b93ad">{p*100:.0f}</text>'
        )
    return "".join(parts)


def main():
    d = sys.argv[1]
    G, N = load("gpt", d), load("nextlat_h1", d)
    assert G and N, f"need examples_gpt/nextlat_h1 under {d}"
    _, ei, pos, ge, ne, gstep, nstep = pick(G, N)
    K = len(gstep["probs"])
    tokens = ge["tokens"]
    true_t = gstep["true_t"]

    # narrative excerpt centered on the current token (compact single line)
    lo = max(0, pos - 7)
    hi = min(len(tokens), pos + 2)
    frag = tokens[lo:hi]

    def pretty(tok):
        return tok.replace("shelf", "shelf ")

    # lay out the excerpt as tspans, highlighting the current token
    W = 880
    strip_y = 96
    xs = 40
    strip = []
    if lo > 0:
        strip.append(f'<tspan fill="#5b6480">... </tspan>')
    for j, tok in enumerate(frag):
        cur = (lo + j) == pos
        col = GOLD if cur else "#aeb6cc"  # playhead token in gold, visible on the dark strip
        strip.append(
            (f'<tspan fill="{col}" font-weight="{700 if cur else 400}">'
             f'{pretty(tok)}</tspan><tspan> </tspan>')
        )
    if hi < len(tokens):
        strip.append(f'<tspan fill="#5b6480">...</tspan>')
    strip_txt = "".join(strip)

    gam, nam = argmax(gstep["probs"]), argmax(nstep["probs"])

    # panels
    panel_w = K * 46 + (K - 1) * 9
    lx = 40 + (W / 2 - 40 - panel_w) / 2          # centered in left half
    rx = W / 2 + (W / 2 - 40 - panel_w) / 2
    py = 214

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 430" width="{W}" height="430" fill="none" role="img" aria-labelledby="sw-title">
  <title id="sw-title">A linear probe reading the tracked object's current shelf off two models at the same mid-story token. NextLat-1step reads the true shelf sharply (green, gold-outlined); the vanilla GPT model is confidently on the wrong shelf (red).</title>
  <style>text {{ font-family: SFMono-Regular, ui-monospace, "SF Mono", Menlo, monospace; }}</style>
  <rect x="2" y="2" width="{W-4}" height="426" rx="16" fill="#0d1117" stroke="#1c2333"/>

  <text x="{W/2:.0f}" y="42" text-anchor="middle" font-size="11" letter-spacing="2.2" fill="#7c9cff" font-weight="700">READING THE BELIEF WHILE THE MODEL READS A STORY</text>
  <text x="{W/2:.0f}" y="64" text-anchor="middle" font-size="12" fill="#97a0b8">same story, same probe, only the training objective differs</text>

  <!-- narrative strip with the playhead -->
  <rect x="30" y="78" width="{W-60}" height="34" rx="9" fill="#0a0d14" stroke="#222b42"/>
  <text x="{xs}" y="{strip_y+4}" font-size="13">{strip_txt}</text>

  <text x="{lx + panel_w/2:.0f}" y="188" text-anchor="middle" font-size="13" font-weight="700" fill="{RED}">GPT (vanilla)</text>
  <text x="{rx + panel_w/2:.0f}" y="188" text-anchor="middle" font-size="13" font-weight="700" fill="{GREEN}">NextLat (1 step ahead)</text>
  <text x="{lx + panel_w/2:.0f}" y="205" text-anchor="middle" font-size="10" fill="#6b748c">probe &#8595; which shelf is the ball on?</text>
  <text x="{rx + panel_w/2:.0f}" y="205" text-anchor="middle" font-size="10" fill="#6b748c">probe &#8595; which shelf is the ball on?</text>

  {shelf_panel(lx, py, gstep["probs"], true_t, RED)}
  {shelf_panel(rx, py, nstep["probs"], true_t, GREEN)}

  <text x="{lx + panel_w/2:.0f}" y="{py+126:.0f}" text-anchor="middle" font-size="12" fill="{RED}" font-weight="700">reads shelf {gam}: confidently wrong</text>
  <text x="{rx + panel_w/2:.0f}" y="{py+126:.0f}" text-anchor="middle" font-size="12" fill="{GREEN}" font-weight="700">reads shelf {nam}: the true shelf</text>
  <text x="{lx + panel_w/2:.0f}" y="{py+146:.0f}" text-anchor="middle" font-size="10.5" fill="#7c8398">true shelf is <tspan fill="{GOLD}">{true_t}</tspan> (outlined)</text>
  <text x="{rx + panel_w/2:.0f}" y="{py+146:.0f}" text-anchor="middle" font-size="10.5" fill="#7c8398">true shelf is <tspan fill="{GOLD}">{true_t}</tspan> (outlined)</text>

  <text x="{W/2:.0f}" y="410" text-anchor="middle" font-size="13" fill="#c7cde0">the predictive objective keeps the tracked belief <tspan fill="#fff" font-weight="700">linearly legible</tspan> mid-story</text>
</svg>
'''
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(svg)
    print(f"[figure] example {ei}, token pos {pos} ('{gstep['tok']}'), true shelf {true_t}")
    print(f"[figure] GPT reads shelf {gam} ({gstep['probs'][gam]*100:.0f}%), "
          f"NextLat reads shelf {nam} ({nstep['probs'][nam]*100:.0f}%)")
    print(f"[figure] wrote {OUT}")


if __name__ == "__main__":
    main()
