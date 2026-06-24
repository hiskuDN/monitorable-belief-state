# Concealworld 2a — full sweep `2a_full` (complete: 5 arms × 3 seeds)

**Status: complete sweep, n=3 seeds/arm.** 32k steps, K=4/L=24/density=0.3.
`summary.md` = raw tables; this note = interpretation. **Read per-seed** — pooled arm
means are misleading because both task competence *and* the carry-vs-defer outcome vary
by seed.

## The decisive read — per-seed carry classification

Two gates: **(1) did the model learn the task** (decode its own commit target `S_final`
at the fork)? **(2) does it carry** the masked secret through the wandering window (late
offset ≥ 16 MLP lift over chance ≈ 0.27 > 0.30)? A model can ace the task by *carrying*
**or** by *deferring-and-gathering* at the fork — both hit fork-decode ≈ 1.0. The
predict-ahead auxiliary loss is what tips that balance.

| arm | learned task | **robust carry** | per-seed (fork / last-wander / late-MLP-lift) |
|---|---|---|---|
| **gpt** | 3/3 | **0/3** | all defer: fork~1.0, last~0.25, lift ~+0.02 |
| **jtp** | 3/3 | **0/3** | 1234/1235 defer (lift ~0), 1236 partial (+0.23) |
| **mtp** | **1/3** | 1/3 | seed1235 carry (fork 0.93, +0.59); 1234/1236 **fail the task** (fork ~0.26) |
| **nextlat** (h8) | 3/3 | **1/3** | 1234 carry (+0.67), 1236 partial (+0.20), **1235 defers like GPT** (+−0.02) |
| **nextlat_h1** | 3/3 | **3/3** | all carry, linear: lift +0.67 / +0.68 / +0.68; last-wander 0.97 / 1.00 / 0.88 |

## What is solid

1. **A vanilla GPT is a clean, robust failure baseline.** All 3 seeds learn the task,
   defer at the fork (jump ~0.75), and leave **no late-position trace even non-linearly**
   (MLP lift ~+0.02). Not "encoded non-linearly" — **destroyed**. Resolves
   availability-vs-linearization in the negative for GPT.
2. **Minimal (horizon-1) predict-ahead robustly keeps the secret monitorable.**
   `nextlat_h1` carries in **all 3 seeds**, **linearly** (late-window linear ~0.86,
   MLP +0.68), with a stable, tight attractor (last-wander 0.88–1.00). This is the headline
   positive: a predict-ahead objective can keep a masked, instrumentally-needed secret
   linearly monitorable where a vanilla model destroys it.

## What is nuanced — predict-ahead is NOT monolithic, and *more horizon ≠ more carry*

3. **Full-horizon (h8) NextLat does not robustly carry.** Only 1 of 3 seeds carries; one
   seed (1235) **defers and destroys the secret exactly like GPT** despite learning the
   task. So scaling the prediction horizon *up* **destabilizes** carrying rather than
   strengthening it.
4. **JTP does not carry** (0/3) despite being predict-ahead — joint-token prediction
   behaves GPT-like on the carry axis. So *which* predict-ahead objective matters:
   predict-your-own-future-**latent at horizon 1** induces carrying; joint-token does not.
5. **MTP is learnability-bound here** — only 1/3 seeds learned to commit; that seed carries
   strongly (+0.59). Its failures are learnability at these hyperparameters, not a carry
   result. Conditional on learning, MTP can carry.

## The central new question

The task admits **two task-optimal solutions** (carry vs defer-and-gather). The
auxiliary objective tips the balance, and **only horizon-1 tips it reliably**. A plausible
mechanism: horizon-1's target (predict the *next* latent, which during wandering is the
next update to the running state) most directly rewards maintaining a clean running state
every step; longer horizons give a diffuse target that admits non-carrying solutions.
**This horizon → carry-robustness relationship is now the most interesting and
decision-relevant finding** — and it is testable (horizon sweep 1/2/4/8 × more seeds).

## Caveats

- **n=3 seeds**, and carry-vs-defer is a **bimodal training outcome**, so the exact ratios
  (1/3, 3/3) are noisy. The robust endpoints are trustworthy (`nextlat_h1` 3/3 at a tight
  +0.68; GPT 0/3); the intermediate arms need more seeds to quantify.
- The auto-decision in `summary.md` ("Greenlight 2b", pooled Δ) rides the **blunt pooled
  `S_run_wait` metric** — superseded by the per-seed read above.
- Toy task (K=4/L=24); "horizon-1 carries robustly" may be hyperparameter-specific.

## Bottom line

The safety-relevant claim holds in a **sharper, more cautious** form: a predict-ahead
objective *can* keep a concealed instrumental secret linearly monitorable where a vanilla
GPT robustly destroys it — but **robustly only for minimal (horizon-1) latent prediction.**
Full-horizon NextLat is seed-variable, JTP is GPT-like, MTP is learnability-bound. The new
headline is **the horizon → carrying relationship**, which is what to characterize next.

## Before any 2b decision

1. **Horizon sweep** (mtp_horizon ∈ {1,2,4,8}) × **more seeds** (≥5) — map when carrying is
   robust vs seed-variable. This is now the primary experiment, not 2b.
2. More seeds generally to quantify the carry/defer bimodality.
3. **MTP learnability** — more steps / gentler schedule so ≥2 seeds learn, to confirm
   "MTP carries when it learns."
