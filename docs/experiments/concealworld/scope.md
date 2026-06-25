# Concealworld (2a) — Monitorability of a Masked, Instrumentally-Needed Secret

Status: **scoped + built; headroom-gating in progress.** The first experiment in the program with
an *incentive to conceal*. Stage **2a** of the deception ladder: a model must maintain a secret it
will *use* to act but *masks* in its behavior; we ask whether the secret stays **linearly
monitorable** from frozen hidden states, and whether **predict-ahead training** keeps it monitorable
better than a vanilla GPT. Pure-supervised (the masking is demonstrated, not discovered) — the
realistic *distillation* threat. Stage 2b (a discovered/adversarial incentive via RL) is gated on
this result and not built here.

## The one-line question

> When a model holds a secret it must use later but isn't expressing now, is that secret **linearly
> decodable** from its hidden state *while it's being masked* — and does **predict-ahead training**
> make it more decodable than a vanilla GPT?

A cheap linear monitor that can read the secret mid-masking catches a deception an output-only
monitor cannot (behavior is masked by construction). 2a asks whether such a monitor works and whether
training objective changes how well.

## Why the secret is a *running aggregate* (and the v1 dead-end)

**v1 failed and taught us the key constraint.** v1 made the secret a single token `g{G}` at a fixed
early position. A GPT trained for *300 steps* already decoded it at **1.000** in the masking window —
no headroom. Root cause: a single fixed-position token is *passively broadcast* into every downstream
residual by ordinary attention, so a linear probe reads it for free, for every arm, with no
computation or training pressure. There was nothing for predict-ahead to improve.

**v2 fixes this by making the secret a nonlinear running aggregate that must be tracked.** The secret
at time `t` is a **running sum**: `S_t = (d₀ + d₁ + … + updates so far) mod K`. Two properties make
this work where v1 didn't:

1. **Nonlinearity closes the free-readout shortcut.** `S_t` is *not* a linear function of the
   individual updates, so even if attention passively superposes the raw update tokens into a later
   residual, a **linear** probe cannot combine them into `S_t`. For `S_t` to be linearly readable the
   model must *actively maintain a register*.
2. **A changing value defeats "compute-once-and-carry."** Because the updates are **interleaved
   throughout** the sequence and `S_t` keeps changing, the model can't compute the answer once and
   carry a constant. Its options are: **carry** a live, continuously-updated register, or **gather**
   all updates at the fork and compute then. A vanilla GPT has no mid-sequence incentive to carry
   (nothing it emits during wandering needs `S_t`), so its cheapest path is to **defer (gather)** —
   leaving its wandering-position residuals empty of `S_t`. Predict-ahead's objective *forbids*
   deferral (each state must be sufficient for its own future, and the future needs the secret), so
   it **carries**. That difference is the gap we measure.

This is the shell-game / track-the-ball belief-state task, and it is a structurally distinct probe
target from Exp-0's single-token decoupled bit (which had the same free-broadcast triviality and only
worked under partial-observability load).

## Environment (`NextLat/data/concealworld.py`)

A symbolic sequence (no grid — navigation was a confound, not a feature):

```
u{d0}  wait  u{d1}  wait wait  u{d2} ... (fixed length L)   act   a{S_final}   end
<--------------- wandering phase (secret masked) --------->  ^fork  ^action      
```

- **Updates `u{d}`** (the only informative tokens) are interleaved with neutral **`wait`** fillers at
  density `update_density`; the first token is always an update so `S_t` is defined and **uniform on
  `Z_K` at every position**. `S_t` = running sum mod `K`.
- **Masked behavior:** nothing emitted during wandering depends on `S` — only the single late
  **action `a{S_final}`** at the fork uses it. (Replacing v1's multi-step shortest-path commit with a
  *single fork* removes navigation/path-planning confounds — Codex review.)
- **Fixed `L`** (no trajectory-length confound).
- **Headroom knobs:** `n_states` (`K`) and `wander_len` (`L`). Bigger `K`/`L` → more to track → less
  likely GPT carries it for free.

**Leakage audit** (`leakage_audit`, runs at datamodule construction + in the probe): asserts the
running secret is **uniform at every offset** (`max_offset_dev ≤ atol` — kills the offset/label
confound), the final secret is balanced, and `#updates ⊥ S_final`. A single update is independent of
the sum by construction. The action token is the only one that reveals `S`; the probe never reads
it.

## Probe (`experiments/concealworld/probe.py`)

Frozen checkpoint → per-layer hidden states → probes for the running secret, with the confound
controls from the Codex review:

- **Primary target — `S_run_wait`:** `S_t` at **wait positions** (current token is uninformative, so
  any signal is a *carried* register — the purest test). Also `S_run` over all wandering positions.
- **Retention curve:** **per-offset-TRAINED** linear probes (a separate probe per position `t`) — no
  pooled cross-offset leakage — with **per-offset chance (`1/K`) + shuffled floors**. Decodability vs
  `t` is the headline.
- **Gather-vs-carry diagnostic:** decode `S_final` at the **last wandering token** vs the **fork
  token**. A large positive jump = the model assembled the secret at the fork (deferral), not
  continuous carry — direct evidence for the mechanism.
- **Availability vs linearization:** linear (primary) + MLP (floor). If GPT's MLP recovers `S` but
  its linear probe doesn't, predict-ahead's gain is **linearization** of an otherwise-tangled secret
  (still safety-relevant — monitors are linear — but a *narrower* claim than Exp-0's "more
  available"). Reported as separate, pre-registered outcomes.
- Effective rank of wandering-position states (compression sanity).

## Training arms (`NextLat/config/concealworld/`)

Matched **GPT vs NextLat vs MTP vs JTP** (+ **NextLat-h1**), identical except the auxiliary loss, at
the Exp-0 backbone (`n_layer=8, n_embd=256, n_head=8, block_size=256`). ≥3 seeds. MTP/JTP keep the
claim class-level. Trained with a uniform step budget across arms (`modal_app.py::main`).

## Decision criteria (linear probes, primary = `S_run_wait`)

- **predict-ahead tracks > GPT** (mean Δ > ~0.05, with GPT off-ceiling): the masked-but-instrumental
  secret is monitorable and predict-ahead helps → greenlight **2b** (discovered/adversarial
  incentive). Corroborated by the gather-vs-carry diagnostic (GPT jumps at the fork; predict-ahead
  flat).
- **GPT saturates (≈1.0):** no headroom — raise `K` / `L` and re-run (the v1 failure mode; the gate
  below guards against it).
- **All low / no gap:** predict-ahead doesn't visibly help; inspect retention + gather-vs-carry, and
  reconsider before 2b. An interpretable bound on representational monitoring of masked state.

## Headroom gate (the check v1 failed)

Before any full sweep: a **GPT-only smoke**, confirming (a) GPT is **off-ceiling** on `S_run_wait`
mid-sequence (not ≈1.0), and (b) the task is learnable (act-token accuracy high → the model *can*
compute `S`, validating gather). If GPT saturates, raise `K`/`L`. Only then the 5-arm × 3-seed sweep.

## Honest caveats (pre-registered)

- **2a's incentive is demonstrated, not discovered** — a positive speaks to distillation of a masking
  policy, not adversarial evasion. That is 2b's remit.
- **Availability vs linearization** — see probe; the MLP floor decides which claim we can make.
- **Near the objective's definition** — "states that must predict their future encode future-needed
  info better than states that needn't" is close to what predict-ahead is trained to do. The
  non-trivial parts are *linear* decodability, the GPT comparison, and the gather-vs-carry mechanism
  — not surprise. The distinctive safety artifact (an *online* predictive monitor) is Step 3.
- **Horizon** — predict-ahead only pressures a wandering state to hold `S_t` if the fork is within
  reach of its prediction horizon, so the retention curve will decay with distance from the fork
  (that decay is the interesting science; the curve must land in a readable range).

## Results (full sweep `2a_full`, complete: 5 arms × 3 seeds)

Full read: [`docs/results/2026-06-24/2a_full/FINDINGS.md`](../../results/2026-06-24/2a_full/FINDINGS.md).
32k steps, K=4/L=24/density=0.3. **Read per-seed** — pooled means mislead, because both
task competence *and* the carry-vs-defer outcome vary by seed. The task has two
task-optimal solutions (carry the secret, or defer-and-gather at the fork; both hit
fork-decode ≈ 1.0); the predict-ahead aux loss tips the balance.

Per-seed **robust carry** (late offset ≥ 16 MLP lift over chance > 0.30, among seeds that
learned the task): **gpt 0/3 · jtp 0/3 · mtp 1/3** (only 1/3 learned) **· nextlat-h8 1/3
· nextlat_h1 3/3.**

- **GPT is a clean, robust failure baseline.** All 3 seeds learn the task, defer at the
  fork (jump ~0.75), and leave **no late trace even non-linearly** (MLP lift ~+0.02) —
  the secret is *destroyed*, not merely non-linear.
- **Only minimal (horizon-1) predict-ahead robustly carries.** `nextlat_h1` keeps the
  secret **linearly** monitorable in all 3 seeds (late linear ~0.86, MLP +0.68; tight
  attractor). This is the headline positive.
- **More horizon ≠ more carry.** Full-horizon (h8) `nextlat` carries in only 1/3 seeds —
  one seed **defers and destroys the secret exactly like GPT** despite learning the task.
  Scaling the horizon *up* destabilises carrying.
- **JTP is GPT-like** (0/3 carry) despite being predict-ahead; **MTP** is learnability-bound
  (1/3 learned; that seed carries, +0.59). So predict-your-own-future-**latent at horizon
  1** induces carrying; joint-token does not.
- **Caveats:** n=3 and carry-vs-defer is a bimodal *training outcome*, so intermediate
  ratios are noisy (robust endpoints — `nextlat_h1` 3/3, GPT 0/3 — are trustworthy). The
  auto "Greenlight 2b" in `summary.md` rides the blunt pooled metric; superseded by the
  per-seed read.

**The horizon → carry-robustness relationship is now the central question — characterise it
before 2b:** (a) **horizon sweep** mtp_horizon ∈ {1,2,4,8} × ≥5 seeds; (b) more seeds for
the bimodality; (c) MTP learnability (more steps / gentler schedule).

## 2b — adversarial probe-evasion (attempted, PARKED)

Full read: [`docs/results/2026-06-25/2b_FINDINGS.md`](../../results/2026-06-25/2b_FINDINGS.md).
We built a gradient-reversal adversarial probe (model rewarded for hiding S_t from a linear
probe while solving the task) and swept the strength, both end-to-end and on frozen features.
**Both hit the same wall: GRL adversarial removal does not remove** — a fresh probe still
recovers S_t (~0.5) regardless of pressure; harder pressure just collapses the representation
(known result, Elazar & Goldberg 2018). So we report **no adversarial-robustness verdict**.
What we *did* establish: a β=0 invertible recoding preserves the secret (MLP 0.97) at ~0
predict-ahead cost while linear decode is only ~0.48, so **2a's *linear* exposure is not
load-bearing** (a hiding representation exists in principle). Honest framing: **2a is a
non-adversarial (distillation-threat) result; cheap linear monitoring is likely not
adversarially robust but is hard to disprove cleanly via training.** Resuming needs a
converged-discriminator min-max or a non-training existence construction, with a capability gate.

## Implementation map

- `NextLat/data/concealworld.py` — generator + `leakage_audit` + datamodule; registered as
  `"concealworld"` in `train.py`.
- `NextLat/config/concealworld/{gpt,nextlat,mtp,jtp}_concealworld.yaml` — matched arms
  (`n_states`, `wander_len`, `update_density`).
- `experiments/concealworld/{probe.py, modal_app.py, analyze.py}` — probe (retention + gather-vs-carry
  + availability/linearization), Modal entrypoints (`smoke`, `main` with `n_states`/`wander_len`
  headroom knobs), aggregation. Results under `docs/results/<date>/<tag>/`.
- Reuses the Exp-0 `return_all_layers` model surgery — no new model changes.

## Relation to the rest of the program

Supersedes the earlier [lieworld](../lieworld/scope.md) as the *next* experiment (incentive before
surface). Extends Exp-0's decoupled-bit finding by adding the missing incentive and a *running*
(history-integrated, nonlinear) secret. Distinct from SAPLMA-style lie-detection (frozen model,
standalone statement, nonlinear classifier): objective-as-lever, a history-carried secret, behavior
that masks it, linear-first probe — see
[feasibility-study.md](../../feasibility-study.md#relation-to-prior-work-lie-detection--latent-knowledge-probing).
