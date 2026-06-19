# Experiment 0 — Step 2: does NextLat's compactness buy a monitor anything?

Status: **preliminary (n=1 seed, v3 8-layer/d=512 checkpoints).** Verdict:
**compactness is safety-neutral on the monitor-cost axes tested** — NextLat's low
effective rank does *not* yield a cheaper, more sample-efficient, or more robust
monitor. If anything **JTP** (a *less*-compact multi-token objective) is the practical
monitor winner. The durable claim stays class-level; the case for NextLat being
*specifically* preferable now rests on its forward-rollout mechanism (Step 3), not
compactness.

## Question

Step 1 (the v3 scale check in [`exp0.md`](exp0.md)) established that the monitorability
gain is **generic** to predict-ahead training and that NextLat's one distinctive axis is
**compression** (low effective rank). Step 2 asks the only remaining question where
NextLat could be specifically "the answer": **does that compactness buy a real *monitor*
anything** — a cheaper, lower-dimensional, more sample-efficient, or more robust one?

## Method

On the v3 (8-layer/d=512) checkpoints, for each of the 5 arms, at each arm's **best
decode layer** for 81-way position (`y_cell`):

- **(A) Low-dimensional monitor.** PCA the hidden states; fit a linear probe on the top-k
  PCs; accuracy vs `k ∈ {1,2,4,8,16,32,64,128,256}`. Headline `k95` = #PCs to reach 95%
  of the full-dim accuracy. *Tests "how few dimensions to monitor."* The compactness
  hypothesis predicts NextLat (low rank) needs far fewer PCs than MTP (high rank).
- **(B) Sample-efficiency.** Fit the probe on `N ∈ {50…15000}` labeled examples;
  accuracy vs N. *Tests "how few labels to build a monitor."*
- **(C) Robustness under shift.** Train the monitor on **short** trajectories, evaluate on
  **long** ones. This behavioral shift keeps the *model* in-distribution, so it isolates
  the **monitor's** generalization. (A trap-density/map shift instead makes the model
  fully OOD and *every* arm floors to ~chance — no signal; this is why we use a behavioral
  shift.) `drop` = in-dist minus shifted accuracy.

Code: `experiments/gridworld/step2_compactness.py` (Modal entrypoint `step2`). Raw results:
`../../results/2026-06-19/v3_L8d512_step2/`.

## Results (n=1, seed 1234)

| arm | decode | eff. rank (best layer) | **k95** | sample (≈labels→0.94) | short→long drop |
| --- | --- | --- | --- | --- | --- |
| GPT | 0.57 | 285 | 256 | caps 0.50 | +0.061 |
| **NextLat (h=8)** | 0.95 | **83** (compact) | 16 | ~500 | −0.019 |
| NextLat-h1 | 0.95 | 79 | 16 | ~500 | −0.018 |
| **JTP** | 0.95 | 207 (diffuse) | **4** | ~250–500 | −0.017 |
| MTP | 0.92 | 432 (most diffuse) | 32 | ~2k (caps 0.92) | −0.012 |

PCA accuracy by k = `[1, 2, 4, 8, 16, 32, …]`:
- NextLat: `0.13 0.21 0.28 0.83 0.95 0.95 …` (saturates ~16 PCs)
- JTP: `0.28 0.76 0.92 0.94 0.95 …` (saturates ~4 PCs)
- MTP: `0.02 0.03 0.21 0.45 0.80 0.88 0.91` (saturates ~32–64, caps 0.92)

### Findings

1. **`k95` does NOT track effective rank.** The most *diffuse* predictive arm — JTP (rank
   207) — gives the **lowest-dimensional monitor (k95 = 4)**; NextLat (rank 83, compact)
   needs 16; MTP needs 32. The task-relevant subspace is low-dim for *all* predictive arms
   and is **decoupled from overall compactness**. NextLat's low effective rank does not make
   its monitor lower-dimensional.
2. **Sample-efficiency does not favor NextLat.** JTP ≈ NextLat ≈ NextLat-h1 (all ~0.94 by
   ~500 labels); only MTP lags. NextLat is not distinctly more label-efficient.
3. **Robustness does not favor NextLat.** Every predictive arm generalizes short→long with
   ~0 drop (−0.01 to −0.02); only the weak GPT baseline drops (+0.06). NextLat is not
   distinctly more robust.

## Verdict

**Compactness is safety-neutral on these monitor-cost axes; the "NextLat is cheaper / more
robust to monitor" hypothesis is falsified.** The practical monitor winner is **JTP**, a
multi-token (not latent-prediction) objective that is *less* compact than NextLat. So:

- The durable claim remains **class-level**: predictive-representation objectives make the
  belief state monitorable; NextLat is one instance, not preferable for monitoring.
- NextLat's effective-rank compactness is real but **does not translate into monitor
  benefits**. Its value, if any, must come from elsewhere — the trained latent-transition
  model and **predictive forward-rollout gating (Step 3)**, which is the only remaining
  NextLat-distinct mechanism. Step 2 sharpens the case: if Step 3 shows nothing, NextLat is
  fully demoted to one instance of the class.

## Caveats

- **n=1 seed**; the `k95` grid is coarse (powers of two). The qualitative conclusion is
  robust across the full curves (no axis where NextLat distinctly wins; JTP ties or beats),
  but the exact values want confirming with seeds 1235/1236.
- Effective ranks here (79–432) are at each arm's best *decode* layer, higher than the
  canonical *final*-layer ranks (NextLat ~46, MTP ~440) but with the same ordering.
- Still the small (~33M-param) synthetic gridworld.
