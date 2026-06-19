# Experiment 0 — Expressed-vs-Decoupled Probing (NextLat-for-safety killer test)

Status: **v1 + v2 + v2b + design-space sweep + v3 scale-robustness complete** (GPT, NextLat,
MTP, JTP, NextLat-h1 × 3 seeds). Headline: in a partially-observable task that *requires*
belief-state tracking, predictive auxiliary objectives make latent state far more linearly
monitorable than a vanilla LLM (position 0.71 → ~0.95, genuine info not linearization) — but
this is **generic to predict-ahead training (MTP/JTP match NextLat), not NextLat-specific.**
NextLat's distinctive axis is **compression**: it reaches the same monitorability at ~6× lower
effective rank than MTP (30 vs 181). **The v3 scale check (4× backbone, 8-layer/d=512) confirms
the gap is not a capacity artifact — it *widens*: GPT drops to ~0.52 (3 seeds) while the
predictive arms hold ~0.90–0.95.** See Results (v2b), the design-space sweep, and Results (v3).

## Purpose

The cheapest hard gate on the agenda (`../../nextlat-ai-safety-research-agenda.md`). It
tests the central framing: NextLat's compression should make **behaviorally-expressed**
safety variables more decodable from hidden states, while it may **compress away
behaviorally-decoupled** variables. Outcome decides whether to greenlight Workstream 2
(predictive forward-rollout gating) or reclassify NextLat as a capability method.

## Environment (`NextLat/data/gridworld.py`)

A 9×9 grid. Each example is a random walk the model is trained to continue
(next-token prediction over move directions), encoded as:

```
[start_cell] [g{G}] d_1 d_2 ... d_L [t{G}] end
   idx 0       r=1   <-- directions -->  T   T+1
```

Traps are a **fixed** set of forbidden cells the walk routes around (not terminators),
so trajectories run to a controlled length and ~100% have a usable window.

**Labels (probing only; never used in training):**
- `y_exp_1[t]` = current cell within Manhattan distance 1 of a trap (trivial, local floor).
- `y_exp_k[t]` = within distance `k ≤ window_d` of a trap (**primary expressed label**).
  Deterministic from the agent's current cell + the fixed map → tests whether position
  is decodable. Trap-proximity restricts the legal-move set, so it is behaviorally
  expressed in the near-future token distribution.
- `y_dec[t] = G`, a secret bit revealed at `r=1` (`g0`/`g1`), affecting only the terminal
  token `t{G}` at `T`. Probed **only inside** `W = {r+d < t < T−d}`, `d = window_d` (=8),
  the carried-latent dead zone.

**Leakage control (validity-critical):** the entire path is sampled *before* `G`, and
`G ~ Bernoulli(½)` is drawn independently, so `G ⟂ {path length, start/goal, trap
frequency, trajectory template}` and terminal classes are 50/50. `leakage_audit()`
asserts this; it runs at datamodule construction and again in the probe. Measured on a
5k sample: p(G=1)=0.50, all G-confound correlations < 0.004, 100% windowed, expressed
base rate ≈ 0.53.

## Training arms (`NextLat/config/gridworld/`)

Matched GPT vs NextLat, identical except the NextLat auxiliary loss: `n_layer=8,
n_embd=256, n_head=8, block_size=256` (~6.1M params), Adam lr 3e-4. NextLat uses the
paper's Manhattan setting: `lambda_mse=1.0, lambda_kl=0.1, lambda_ce=0, mtp_horizon=8,
proj_factor=2.0`. ≥3 seeds per arm.

Optional **mechanism control** (v1.5): ablated NextLat with `mtp_horizon=1` / reduced
`lambda_kl` to attribute any decoupled gap to compression/horizon, not "different model."

## Probe (`experiments/gridworld/probe_exp0.py`)

Standalone (does not reuse the future-token harness). Loads the frozen checkpoint via
`core_train.initialize_model(..., checkpoint_path=...)`, runs the model with the new
`return_all_layers=True` flag (8 blocks + final-norm = 9 layers), caches per-layer
hidden states + labels for a held-out eval set, then fits **linear (primary) and MLP
(secondary)** probes per (layer × label). Discipline:
- `y_dec` evaluated only in `W`; **position-conditioned** accuracy vs `t−r` reported.
- Identical trajectory-level train/val split across arms & seeds (eval set regenerated
  deterministically), so arms differ only in their representations.
- Chance + shuffled-label floors per (label, layer).
- Effective rank of final-layer states (compression sanity).

## How to run (Modal; see [[modal-gpu-access]])

```
# from project root, using the venv:
./.venv/bin/python -m modal run experiments/gridworld/modal_app.py::smoke   # tiny end-to-end
./.venv/bin/python -m modal run experiments/gridworld/modal_app.py::main \
        --seeds 1234,1235,1236 --train-batches 10000 --n-eval 6000          # full sweep
python3 experiments/gridworld/analyze.py 2026-06-17/v2b                     # aggregate -> docs/results/<date>/<variant>/
```

## Decision criteria (made on linear probes; `y_exp_k` primary, `y_exp_1` only a floor)

- NextLat > GPT on `y_exp_k` **and** ties/loses on `y_dec`-in-`W` → framing holds; greenlight WS2.
- NextLat wins on **both** → compression-deletes-decoupled worry falsified; widen scope.
- NextLat ≤ GPT on `y_exp_k` → core premise false; reclassify NextLat as capability/efficiency.

`y_dec` is a diagnostic, not a pass/fail bar (tie-or-lose is the predicted, acceptable outcome).

## Results (v1)

Full table + plots: `../../results/2026-06-17/v1/` (`summary.md`, `*.png`).
3 seeds/arm, 6000 steps, 8-layer/256-dim (~6.1M params). Probe layers 0–8 (8 blocks +
final-norm). All leakage/shuffled floors at chance (≈0.47–0.53), so every number is genuine.

**Clean positive — compression reproduced:**
- Effective rank (final layer): **GPT 53.9±2.2 vs NextLat 29.7±0.4** → NextLat compresses
  ~1.8× harder, matching the paper's signature.

**Probeability — saturated, no NextLat advantage:**
Every probe target saturates to ~1.00 by layer 1–2 for *both* arms (an 8-layer transformer
trivially, linearly encodes gridworld position). The only unsaturated layer is **layer 0**,
where **GPT consistently leads** — opposite the hypothesis:

| target (layer-0) | GPT | NextLat |
|---|---|---|
| y_exp_1 AUROC | 0.934 | 0.909 |
| y_exp_k AUROC | 0.986 | 0.968 |
| y_cell acc (81-way) | 0.89 | 0.78 |

→ NextLat's heavier compression slightly *reduces* early-layer linear availability of the
belief state, with no difference once both saturate.

**Decoupled — no compression-deletion:**
`y_dec`-in-W = 1.00 at every layer and every offset `t−r` for **both** arms (GPT broadcasts
the secret bit; NextLat compresses ~2× yet still broadcasts it identically). The
"compression deletes behaviorally-decoupled info" worry is **not supported** here.

**Verdict:** in this easy regime the experiment **cannot discriminate** — position is
trivially linearly decodable for the baseline, so there is no headroom for a NextLat
probeability gain to appear, and where headroom exists (layer 0) GPT is marginally ahead.
This neither confirms nor refutes Workstream 1; it says the *task is too easy*. The
compression result is the one solid finding.

## Results (v2 — capacity-limited, 2-layer / d=128)

`../../results/2026-06-17/v2/summary.md`; same 3 seeds, 6000 steps. Goal: remove the v1
ceiling so a probeability difference can appear.

- **Compression difference vanishes at small scale:** effective rank GPT 19.9±1.0 vs
  NextLat 20.3±0.7 (equal) — a 2-layer/128-dim model is inherently low-rank, so the v1
  compression gap is a larger-model phenomenon.
- **Small, consistent NextLat edge on expressed decodability** (linear, per-layer acc,
  across all layers + 3 seeds + 3 targets):
  - y_exp_k: GPT 0.974–0.981 vs **NextLat 0.991**
  - y_cell (81-way): GPT 0.989 vs **NextLat 0.999** (layers 1–2)
  - y_exp_1 (layer 0): GPT 0.971 vs **NextLat 0.985**
  Direction is now *toward* the hypothesis (opposite v1's layer-0 hint), but the effect is
  ~1–1.5% and the baseline is still near ceiling (~0.97–0.99).
- **Decoupled still 1.00 for both** at every layer/offset — no compression-deletion.

**Combined verdict (v1 + v2):** the gridworld belief variable (grid position) is too easily
linearly decodable to give a *decisive* read — even a 2-layer model nearly saturates. Signals
seen: NextLat compresses much harder at scale (v1) and shows a *small* probeability edge under
capacity pressure (v2), but nothing large or mechanism-confirmed (compression gap disappears
exactly where the probeability edge appears). This is **weak, suggestive support** for
Workstream 1, not confirmation, and the decoupled worry is unsupported.

## Results (v2b — partial observability) — **the decisive, positive result**

`../../results/2026-06-17/v2b/summary.md`; 4-layer/d=256, trap_frac=0.12, 8000 steps, 3 seeds.
No absolute start anchor; interleaved wall-pattern observations, so **position must be
integrated over the trajectory** (a genuine belief-state task). The baseline finally lands
off the ceiling, and a large NextLat advantage appears. Shuffled floors at chance.

**`y_cell` (81-way position localization) — linear accuracy by layer (chance 0.02):**

| layer | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| GPT | 0.66 | 0.70 | **0.71** | 0.68 | 0.67 |
| NextLat | 0.77 | 0.90 | 0.94 | 0.95 | **0.95** |

NextLat's belief state **sharpens with depth** to 0.95 while GPT plateaus ~0.70 — a **~28-point**
gap. And it does so while being **far more compressed**: effective rank **GPT 79.8±1.0 vs
NextLat 30.3±0.2** (~2.6×). So NextLat is simultaneously *more compact* and *more linearly
monitorable* — exactly the Workstream-1 hope.

- **Expressed (`y_exp_k`)** AUROC rises with depth for NextLat (0.96→**0.996**) vs flat GPT
  (~0.93); Δ≈+0.06 at the final layer (base rate 0.88, so AUROC is the right metric).
- **Decoupled (`y_dec`)**: GPT 0.86 (and oscillating 0.5–0.9 across the window) vs **NextLat a
  flat 1.00** everywhere. NextLat *retains the decoupled bit better*, not worse — the
  "compression deletes decoupled info" worry is **falsified** in this setting; NextLat
  maintains a more complete sufficient statistic.

**Mechanism control (`mtp_horizon=1`).** The ablated NextLat (1-step latent prediction instead
of 8) localizes *just as well* — `y_cell` 0.948 vs 0.949 — at the same low effective rank
(32 vs 30, both far below GPT's 80). So the advantage is **not** about the multi-step rollout
depth: the **next-latent-prediction objective itself, even in its minimal 1-step form, is what
produces the compact, more-decodable belief state**. (Rules out "horizon depth"; the MTP/JTP
arms below test whether it's even specific to the *latent*-prediction loss.)

## Results (v2b design-space sweep) — is the monitorability gain NextLat-specific?

Same partial-obs task and 4-layer/d=256 backbone, 3 seeds. Two *generic* predictive auxiliary
objectives are added — **MTP** (Gloeckle multi-token) and **JTP** (joint multi-token) — plus a
**strengthened nonlinear-probe floor** (256-hidden MLP, 40k samples). `y_cell` = 81-way position
localization (chance 0.02). Total params differ (the MTP heads add parameters), but the *probed
backbone* is matched.

| arm | y_cell linear | y_cell strong-MLP | effective rank |
| --- | --- | --- | --- |
| GPT (no aux) | 0.71 | 0.65 | 80 |
| MTP | 0.94 | 0.94 | **181** |
| JTP | 0.95 | 0.95 | 100 |
| NextLat (h=8) | 0.95 | 0.95 | **30** |
| NextLat (h=1) | 0.95 | 0.95 | 32 |

1. **Monitorability is generic to predictive auxiliary objectives, not NextLat-specific.** MTP,
   JTP and NextLat all lift position decodability from GPT's 0.71 to ~0.94–0.95. "Predict-ahead
   training makes the belief state more linearly monitorable" is a property of the *class*.
2. **Not linearization.** The strong MLP matches linear on the aux arms (info genuinely present)
   but cannot exceed ~0.71 on GPT — the baseline encodes genuinely *less recoverable* belief
   state; the aux objectives add information availability, not just linear arrangement.
3. **Compression is the axis where NextLat is distinctive, and it dissociates from monitorability.**
   Effective rank NextLat 30 vs MTP 181 (more diffuse than even GPT) vs JTP 100. MTP gets equal
   decodability while being the *least* compressed. NextLat is the only arm that is both highly
   decodable *and* highly compressed.

Capability caveat: MTP/JTP raw val loss is inflated by their multi-token terms; we did not
isolate the next-token component here, so "capability-matched" is approximate (matched
architecture/data/compute, not verified-equal next-token loss).

## Results (v3 — scale-robustness check, 8-layer / d=512) — **the gap holds at scale**

`../../results/2026-06-19/v3_L8d512/summary.md`. The v2b read rests on a **capacity-starved
baseline** (4-layer GPT caps at 0.71 — does it close the gap if given room?). v3 re-runs the
*identical* partial-obs task and 5-arm panel at **4× the backbone** (8-layer/d=512, ~33M params
for the NextLat arm), training extended to **12k steps** so GPT is genuinely converged. Task and
training recipe (partial_obs, trap_frac, grad_accum=1, effective batch 256) held identical —
only model size + training length scale up, so any change is attributable to capacity. A100-80GB.

**`y_cell` (81-way position) best-linear accuracy + final-layer effective rank:**

| arm | y_cell acc | vs v2b | effective rank | vs v2b | seeds |
| --- | --- | --- | --- | --- | --- |
| GPT (no aux) | **0.52 ± 0.02** | 0.71 → 0.52 | 221 ± 22 | 80 → 221 | 3 |
| MTP | 0.90 ± 0.04 | 0.94 → 0.90 | 440 | 181 → 440 | 3 |
| JTP | 0.95 | = | 177 | 100 → 177 | 3 |
| NextLat (h=8) | 0.95 | = | 46 | 30 → 46 | 3 |
| NextLat-h1 | 0.95 | = | 36 | 32 → 36 | 2 |

All five arms now at n=3 (NextLat-h1 at n=2). NextLat (h=8) = 0.948±0.001 (seeds 0.949/0.947/0.947),
rank ~45.6 — ~0 seed variance. GPT's 3-seed per-layer plateau is a stable 0.51–0.52, *below* v2b's
0.71, so the gap **widens** with scale.

1. **The monitorability gap survives — and widens — at scale.** Giving GPT 4× the capacity does
   *not* let it linearly expose position; it gets **worse** (0.71 → 0.52), spreading position
   across more dimensions (effective rank 80 → 221). The predictive arms stay at 0.90–0.95. The
   v2b result is therefore **not** a small-model / capacity-starvation artifact — the opposite of
   the "gap collapses" failure mode we were testing for.
2. **Task-sanity gate passes** (rules out "undertrained masquerading as capacity-starved"): GPT
   reaches the *same* val loss (~0.62) as the aux arms and decodes trap-proximity well (`y_exp_k`
   0.92, `y_exp_1` AUROC 0.97). It learned a functional belief state; it simply does not make fine
   81-way position linearly available. So 0.57 is a genuine representational property.
3. **Generic, confirmed at scale.** MTP, JTP, and NextLat all cluster at 0.90–0.95 — the
   monitorability gain remains a class property, not NextLat-specific.
4. **Compression dissociation is sharper at scale.** NextLat-h1 matches JTP's 0.948 at effective
   rank **36 vs MTP's 440** (~12× spread) — and MTP's 440/512 is barely compressed at all. NextLat
   alone is *monitorable-and-compact*; MTP is *monitorable-but-near-full-rank*.
5. **Decoupled-deletion still absent.** `y_dec` in-window: NextLat / NextLat-h1 flat **1.00**
   across the whole window; GPT oscillates ~0.6–1.0; MTP 0.95, JTP 0.98. Every predictive arm
   retains the carried bit as well or better than GPT.

**Read:** the Workstream-1 foundation is confirmed and scale-robust (in this regime):
predictive-objective monitorability is real, not a baseline-capacity artifact, generic across
the objective class, and NextLat's distinctive property is compactness. This greenlights step 2
of the program — *does that compactness have independent monitor value* (robustness under shift,
low-dimensional monitors, transfer) — the only remaining question where NextLat could be the
specific answer rather than one instance of the class. See
[`../../research-overview.md`](../../research-overview.md).

## Verdict (v1 + v2 + v2b + design-space + v3 scale)

- **Easy regime (v1/v2):** position is readable off a start token; everything saturates; no signal.
- **Belief-state regime (v2b):** predictive auxiliary objectives make safety-relevant latent state
  **much more monitorable** than a vanilla LLM (0.71 → ~0.95), and this is **genuine information
  availability**, not linearization. **But it is a generic property of predict-ahead training —
  MTP and JTP match NextLat.** So the *monitorability* story is real and robust, but it is **not a
  reason to bet on NextLat specifically**; the durable claim is "predictive-representation
  objectives improve belief-state monitorability."
- **NextLat's specific contribution** is delivering that monitorability at a **far lower effective
  rank** (compact *and* monitorable, where MTP is monitorable *but* diffuse). Whether that
  compactness has independent safety value — simpler/cheaper/more-robust monitors, or the
  decoupled-deletion risk the agenda worries about — is the open question this experiment did not
  settle (the decoupled bit was *better* retained here, in all aux arms).
- **Scale-robustness (v3):** the gap is **not** a capacity-starvation artifact. At 4× the
  backbone (8-layer/d=512) the vanilla baseline's linear position decodability *drops* to ~0.52
  (it diffuses across more dimensions) while the predictive arms hold ~0.90–0.95 — so scaling the
  baseline makes its belief state *harder* to monitor, widening the gap. The compression
  dissociation also sharpens (NextLat rank 36–46 vs MTP 440 at equal decodability).

Caveats throughout: ≤~33M-param models, synthetic gridworld, not language/agentic; capability not
exactly matched. v3 adds a 4× scale point (the gap survives it), but this is still a controlled
existence proof at small absolute scale, not a frontier-scale or language/agentic claim.

Caveats: tiny synthetic gridworld, ~3M params, not language/agentic — the paper's scale caveats
apply, and the decoupled result is the opposite of the agenda's worry (good for monitoring, but
means this env doesn't exhibit the compression-deletion failure mode). Next: confirm with the
horizon control, then scale toward harder/longer-horizon and language-like state-tracking.
