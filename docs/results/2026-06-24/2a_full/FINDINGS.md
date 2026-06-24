# Concealworld 2a — early read (full sweep `2a_full`, 3 seeds for 4 arms)

**Status: PRELIMINARY but informative.** All 15 jobs trained to 32k steps. This read
covers **3 seeds (1234, 1235, 1236) for gpt / mtp / jtp / nextlat_h1**, and **2 seeds
(1234, 1236) for nextlat** (its seed-1235 run hit a failure and is retrying). Config:
K=4, L=24, density=0.3. `summary.md` has the full raw tables; this note is the
interpretation. **Read per-seed, not pooled** — pooled arm means are misleading here
because task competence varies by seed (below).

## The decisive read — late-position retention, per seed

The honest question is *is the masked secret still present at late positions (offset ≥ 16)?*
answered by the per-offset **MLP** probe (chance ≈ 0.27). But first check **task
competence** (can the model decode its own commit target `S_final` at the fork — if not,
its "absence" is task-failure, not a carrying result):

**Per-seed gather-vs-carry** (`S_final` decode: last-wander → fork):

| arm | seed | last-wander | fork | learned task? | carries? |
|---|---|---|---|---|---|
| gpt | 1234/1235/1236 | ~0.25 | ~1.00 | ✅ all | ❌ defers (jump ~0.75) |
| nextlat_h1 | 1234/1235/1236 | 0.88–1.00 | ~1.00 | ✅ all | ✅ all (jump <0.12) |
| nextlat | 1234 / 1236 | 0.89 / 0.53 | 1.00 / 0.97 | ✅ both | ✅ both |
| jtp | 1234/1235/1236 | 0.25 / 0.25 / 0.51 | ~1.00 | ✅ all | ❌ defers (jump +0.49…+0.75) |
| mtp | **1235** | **0.76** | **0.93** | ✅ | ✅ carries |
| mtp | 1234 / 1236 | 0.28 / 0.24 | **0.26 / 0.26** | ❌ **never learned** | n/a |

**Late-window (off ≥ 16) MLP lift over chance:**

| arm | lift (per-seed where it matters) | reading |
|---|---|---|
| **gpt** | +0.02 (all 3) | secret **genuinely destroyed** — strong MLP can't recover it |
| **jtp** | +0.07 (all 3) | learned the task but **defers** → loses the secret mid-mask |
| **mtp** | seed1235 **+0.59**; seeds1234/1236 ≈0 | **bimodal**: the seed that *learned* carries strongly; the two that failed the task are uninformative |
| **nextlat** | +0.43 (2 seeds) | retained (recoverable) |
| **nextlat_h1** | +0.68 (all 3) | retained **and linear** (linear 0.88) |

## What is solid

1. **A vanilla GPT truly obliterates the masked secret.** All 3 seeds learn the task,
   defer at the fork (jump ~0.75), and leave **no late-position trace even non-linearly**
   (MLP lift +0.02). Not "encoded non-linearly" — **absent**. Clean, seed-robust failure
   baseline. (Resolves availability-vs-linearization in the negative for GPT.)
2. **NextLat keeps it monitorable, robustly.** `nextlat_h1` carries and stays **linearly
   readable** (0.88) in all 3 seeds; `nextlat` carries in both its seeds (MLP +0.43). The
   predict-ahead objective keeps a masked, instrumentally-needed secret recoverable where
   a vanilla model destroys it — the core safety claim of the rung.

## What is nuanced — "predict-ahead" is not monolithic for *carrying*

3. **JTP learns the task but does not carry.** All 3 JTP seeds commit correctly
   (fork ≈ 0.999) yet defer (jump +0.49…+0.75) and lose the secret mid-mask
   (late MLP +0.07) — behaving **GPT-like on the carrying axis** despite being
   predict-ahead. So *which* predict-ahead objective matters: predicting your own future
   **latent** (NextLat) induces carrying; **joint-token** prediction (JTP) does not.
4. **MTP is unstable to learn here, but carries when it learns.** Only 1 of 3 MTP seeds
   learned to commit (fork 0.93 vs chance for the other two); that seed carries strongly
   (last 0.76, late MLP +0.59). MTP's failures are **learnability at these
   hyperparameters**, not a carrying result. Conditional on learning the task, MTP carries.
5. **Horizon-1 surprise.** `nextlat_h1` (mtp_horizon=1) is the **strongest, most linear**
   carrier — so it is *minimal predict-ahead*, not a GPT-like control, and the effect is
   **not** "more horizon → more carrying." Even a 1-step latent-prediction target suffices,
   and a shorter horizon yields a *cleaner linear* belief state.

## Caveats on this read

- **nextlat is 2 seeds** (1234, 1236); seed 1235 retrying. Both carry, but seed-robustness
  for nextlat itself is not yet nailed.
- **Pooled arm means mislead** when task competence varies by seed (MTP). Always read
  per-seed competence (fork decode) before any retention claim.
- The auto-decision in `summary.md` ("Greenlight 2b", pooled Δ) rides the **blunt pooled
  `S_run_wait` metric**. The honest read is the **per-seed late-window** picture above.
- **Open items before a 2b decision:** (a) MTP learnability — re-run with more steps / a
  gentler schedule so ≥2 seeds learn, to confirm "MTP carries when it learns"; (b) the
  final nextlat seed-1235 for a clean 3-seed NextLat; (c) decide whether JTP's
  learns-but-defers behavior is a robust property or hyperparameter-specific.

## Bottom line

The safety-relevant headline holds and **strengthens**: predict-ahead of the **NextLat**
kind keeps a concealed instrumental secret monitorable (linearly, at horizon 1), while a
vanilla GPT genuinely destroys it. But **"predict-ahead" is not one thing for carrying** —
NextLat carries, JTP learns-but-defers, MTP carries only when it learns the task. The
distinction (predict-your-own-future-**latent** vs joint-token) is the most interesting
new result and the thing to characterize next.
