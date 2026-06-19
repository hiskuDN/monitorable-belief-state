# NextLat belief-state monitor — interactive demo

A small static site that shows the Experiment 0 result viscerally: freeze a vanilla **GPT** and a
**NextLat** model trained on the same partially-observed gridworld, then watch a linear probe
decode each one's internal position ("belief state") step by step. NextLat's belief is sharp and
correct; GPT's is muddy and often wrong — the predictive objective makes the belief state
*linearly monitorable*.

It's **fully static and precomputed**: no model runs in the browser and there's no backend. All
data lives in [`public/data/demo.json`](public/data/demo.json) (~0.3 MB).

## Run locally

```bash
cd web
pnpm install        # or: npm install / yarn
pnpm dev            # http://localhost:5173
pnpm build          # -> dist/  (static, deploy anywhere)
pnpm preview        # serve the production build
```

`vite.config.js` sets `base: './'`, so the `dist/` build works on GitHub Pages, Netlify, Vercel,
or any static host without extra config.

## Regenerate the data

`demo.json` is produced from trained checkpoints on the Modal volume by
[`../experiments/gridworld/export_demo.py`](../experiments/gridworld/export_demo.py), driven by the `export_demo`
entrypoint:

```bash
# from the repo root, with the venv + modal configured:
./.venv/bin/python -m modal run experiments/gridworld/modal_app.py::export_demo \
    --arms gpt,nextlat --tag v2b --n-demo 24
# writes web/public/data/demo.json
```

`--arms` accepts the full panel (`gpt,nextlat,mtp,jtp,nextlat_h1`); per-step belief heatmaps are
exported for `gpt` and `nextlat`, and by-layer accuracy + effective rank for every arm listed
(used by the by-layer chart / design-space view).

## Data shape (`demo.json`)

```jsonc
{
  "meta":  { "grid_size": 9, "n_cells": 81, "trap_cells": [..], "note": ".." },
  "arms":  { "gpt": { "by_layer_acc": [..], "best_layer": 2, "best_acc": 0.75, "eff_rank": 81 },
             "nextlat": { .. } },
  "trajectories": [
    { "tokens": [..],
      "steps": [ { "pos": 1, "true": 36,
                   "gpt":     [[36, 0.66], ..],   // top-8 [cell, prob], argmax first
                   "nextlat": [[36, 0.63], ..] } ] }
  ]
}
```
