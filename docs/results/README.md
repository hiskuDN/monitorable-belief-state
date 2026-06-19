# Experiment results

Probe results are organized as `docs/results/<run-date>/<variant>/`. Each variant folder
holds per-(arm, seed) probe JSONs (`probe_<arm>_seed<seed>.json`), an aggregated
`summary.md`, and by-layer / position-conditioned plots, all produced by
`experiments/analyze.py <date>/<variant>`.

## 2026-06-17 — Experiment 0 (see [../experiments/exp0.md](../experiments/exp0.md))

| variant | setup | headline |
| --- | --- | --- |
| `v1`  | fully observed, 8-layer/d=256 | both arms saturate (~1.0); NextLat rank 30 vs GPT 54 |
| `v2`  | fully observed, 2-layer/d=128 (capacity-limited) | still near-ceiling; tiny NextLat edge; ranks equal |
| `v2b` | **partial observability**, 4-layer/d=256, **5-arm design-space sweep** | position decodability: GPT 0.71 → MTP 0.94 / JTP 0.95 / NextLat 0.95 — **generic to predict-ahead objectives, not NextLat-specific**; NextLat distinct on compression (rank 30 vs MTP 181) |

Arms: `gpt` (floor), `nextlat`, `mtp`, `jtp`, `nextlat_h1` (`mtp_horizon=1` control). Regenerate a
summary with, e.g., `./.venv/bin/python experiments/analyze.py 2026-06-17/v2b`. (Some runs were
probed on 2026-06-18 and consolidated into the `2026-06-17/v2b` folder.)

## 2026-06-19 — Experiment 0 v3 (scale-robustness; see [../experiments/exp0.md](../experiments/exp0.md))

| variant | setup | headline |
| --- | --- | --- |
| `v3_L8d512` | **partial observability at 4× scale**, 8-layer/d=512, 12k steps, 5-arm panel | **the gap holds — widens — at scale, not a capacity artifact:** 81-way position GPT **0.52 ± 0.02** (3 seeds) vs MTP 0.90 / JTP 0.95 / NextLat 0.95; GPT effective rank 80→221 (diffuses more) while NextLat stays compact (rank ~36–46) vs MTP ~440 |

Regenerate with `./.venv/bin/python experiments/analyze.py 2026-06-19/v3_L8d512`. All five arms at
n=3 (NextLat-h1 n=2): GPT 0.52±0.02, NextLat 0.948±0.001 (rank ~46), MTP 0.90±0.04, JTP 0.95 — the
aux arms have ~0 seed variance. `summary.md` is current.

## 2026-06-19 — Step 2: does compactness buy a monitor anything? (see [../experiments/step2-compactness.md](../experiments/step2-compactness.md))

`v3_L8d512_step2/` — `step2_<arm>_seed1234.json` per arm (PCA / sample-efficiency / short→long-shift
battery on the v3 checkpoints). **Verdict: no — compactness is safety-neutral.** `k95` (PCA dims for
95% accuracy) does not track effective rank: JTP (diffuse, rank 207) needs **4**, NextLat (compact,
rank 83) needs 16, MTP 32; all predictive arms are equally robust to the short→long shift (~0 drop).
NextLat is not preferable for monitoring. *(n=1 seed, 1234.)*
