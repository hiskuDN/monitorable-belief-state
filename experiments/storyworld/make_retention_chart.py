"""Render the storyworld retention chart (SVG) from the real probe dumps: monitor read-accuracy
(linear probe decoding the ball's current shelf) as a function of how far into the story we are,
GPT vs NextLat-1step, averaged over seeds. NextLat stays pinned near 100% the whole way (carry);
GPT decays toward chance as the ball moves and only snaps back at the question (defer). Self-
contained SVG, hand-built (no charting lib), matching web/public/figures style.

Usage: ./.venv/bin/python experiments/storyworld/make_retention_chart.py docs/results/2026-06-27/story_easy
Writes: web/public/figures/storyworld-retention.svg
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "web", "public", "figures", "storyworld-retention.svg")

ORANGE = "#f2754f"
TEAL = "#4fd1c5"

W, H = 880, 430
PL, PR = 74, 736          # plot left / right edge of the move region
PT, PB = 96, 344          # plot top (acc=1) / bottom (acc=0)
QX = 806                  # x of the "question" column
PMIN, PMAX = 4, 53        # story positions to plot (from the shelf reveal to the last move)


def curve(d, arm):
    accs = {}
    for f in sorted(glob.glob(f"{d}/probe_{arm}_seed*.json")):
        r = json.load(open(f))["results"]["S_run"]["retention"]
        for k, v in r.items():
            accs.setdefault(int(k), []).append(v["acc"])
    return {k: sum(v) / len(v) for k, v in accs.items()}


def fork(d, arm):
    vals = [json.load(open(f))["results"]["gather_vs_carry"]["fork"]
            for f in sorted(glob.glob(f"{d}/probe_{arm}_seed*.json"))]
    return sum(vals) / len(vals)


def X(p):
    return PL + (p - PMIN) / (PMAX - PMIN) * (PR - PL)


def Y(a):
    return PB - a * (PB - PT)


def polyline(d, arm, color):
    c = curve(d, arm)
    pts = [(X(p), Y(c[p])) for p in range(PMIN, PMAX + 1) if p in c]
    pts.append((QX, Y(fork(d, arm))))  # final "question" column
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    end = pts[-1]
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="{color}" opacity="0.5"/>'
                   for x, y in pts[:-1])
    return (
        f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.6" '
        f'stroke-linejoin="round" stroke-linecap="round"/>{dots}'
        f'<circle cx="{end[0]:.1f}" cy="{end[1]:.1f}" r="4.5" fill="{color}"/>'
    )


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs/results/2026-06-27/story_easy")
    chance = 0.25
    move_start = 19  # first position where the plotted lines visibly diverge (ball moving)

    # y gridlines
    grid = []
    for a in (0.25, 0.5, 0.75, 1.0):
        y = Y(a)
        dash = ' stroke-dasharray="4 4"' if a == chance else ""
        grid.append(f'<line x1="{PL}" y1="{y:.1f}" x2="{QX}" y2="{y:.1f}" stroke="#222b42" stroke-width="1"{dash}/>')
        lab = "chance" if a == chance else f"{a:.2f}"
        col = "#6b748c" if a == chance else "#7c8398"
        grid.append(f'<text x="{PL-8}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="{col}">{lab}</text>')

    # x guide: where the ball starts moving, and the question column
    mx = X(move_start)
    guides = (
        f'<line x1="{mx:.1f}" y1="{PT-6}" x2="{mx:.1f}" y2="{PB}" stroke="#2b3350" stroke-width="1" stroke-dasharray="3 4"/>'
        f'<text x="{mx:.1f}" y="{PT-12}" text-anchor="middle" font-size="10" fill="#8b93ad">the ball starts moving</text>'
        f'<line x1="{QX}" y1="{PT-6}" x2="{QX}" y2="{PB}" stroke="#2b3350" stroke-width="1" stroke-dasharray="3 4"/>'
        f'<text x="{QX}" y="{PT-12}" text-anchor="middle" font-size="10" fill="#8b93ad">the question</text>'
    )

    gpt_line = polyline(d, "gpt", ORANGE)
    nl_line = polyline(d, "nextlat_h1", TEAL)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" fill="none" role="img" aria-labelledby="ret-title">
  <title id="ret-title">Monitor read-accuracy versus story position. A linear probe reading NextLat-1step stays near 100 percent across the whole story; reading vanilla GPT it decays toward chance as the ball moves, then snaps back to 100 percent only at the question.</title>
  <style>text {{ font-family: SFMono-Regular, ui-monospace, "SF Mono", Menlo, monospace; }}</style>
  <rect x="2" y="2" width="{W-4}" height="{H-4}" rx="16" fill="#0d1117" stroke="#1c2333"/>

  <text x="{W/2:.0f}" y="40" text-anchor="middle" font-size="11" letter-spacing="2.2" fill="#7c9cff" font-weight="700">CAN A MONITOR STILL READ THE BALL'S LOCATION?</text>
  <text x="{W/2:.0f}" y="62" text-anchor="middle" font-size="12" fill="#97a0b8">linear-probe accuracy for the ball's current shelf, across the story (3-seed mean)</text>

  {"".join(grid)}
  {guides}
  <line x1="{PL}" y1="{PB}" x2="{QX}" y2="{PB}" stroke="#2b3350" stroke-width="1"/>

  {gpt_line}
  {nl_line}

  <!-- legend, in the empty lower-left where both lines still sit high -->
  <g transform="translate({PL+14},{PT+150})">
    <line x1="0" y1="0" x2="22" y2="0" stroke="{TEAL}" stroke-width="2.6"/>
    <text x="28" y="4" font-size="12" fill="{TEAL}" font-weight="700">NextLat (1 step ahead): carries</text>
    <line x1="0" y1="22" x2="22" y2="22" stroke="{ORANGE}" stroke-width="2.6"/>
    <text x="28" y="26" font-size="12" fill="{ORANGE}" font-weight="700">GPT (vanilla): defers</text>
  </g>

  <text x="{(PL+QX)/2:.0f}" y="{PB+34:.0f}" text-anchor="middle" font-size="11" fill="#8b93ad">story position &#8594;  (ball placed, then moved sentence by sentence, then queried)</text>
  <text x="26" y="{(PT+PB)/2:.0f}" text-anchor="middle" font-size="11" fill="#8b93ad" transform="rotate(-90 26 {(PT+PB)/2:.0f})">monitor read-accuracy</text>

  <text x="{W/2:.0f}" y="{H-14}" text-anchor="middle" font-size="12.5" fill="#c7cde0">GPT drops the ball mid-story, recovering it only <tspan fill="#fff" font-weight="700">at the question</tspan>: too late</text>
</svg>
'''
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(svg)
    print(f"[chart] wrote {OUT}")
    g, n = curve(d, "gpt"), curve(d, "nextlat_h1")
    print(f"[chart] GPT last-move acc {g[PMAX]:.2f} -> question {fork(d,'gpt'):.2f}; "
          f"NextLat last-move acc {n[PMAX]:.2f} -> question {fork(d,'nextlat_h1'):.2f}")


if __name__ == "__main__":
    main()
