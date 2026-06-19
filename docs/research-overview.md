# Research Overview — Monitorable Belief State via Predictive Objectives

*Current-state synthesis as of 2026-06-19. This is the orientation layer for the program:
the general question we're now chasing, how we got here, and what's next. NextLat was the
starting point and remains one studied instance — the detailed NextLat-specific plan is the
[research agenda](nextlat-ai-safety-research-agenda.md); the first experiment is
[Experiment 0](experiments/gridworld/exp0.md).*

## The question

> **Do predictive-representation training objectives make a model's internal belief state
> more monitorable — and how do monitorability, compactness, capability, and robustness
> trade off across the space of such objectives?**

A "predictive-representation objective" is any auxiliary loss that pushes the hidden state to
predict its own future — next latent (NextLat), multiple future tokens (MTP, JTP), and
relatives. The safety motivation is **monitoring**: if these objectives make safety-relevant
internal state (about to call a tool, entered an irreversible state, saw a hazardous
instruction) more reliably decodable from hidden states, monitors built on probes get simpler
and less brittle.

We entered through one method (NextLat) and a specific hypothesis about it. The first
experiment generalized the question: the interesting object is the **class of objectives and
the design-space frontier**, not any single method.

## Which tasks does this apply to? (the belief-state regime)

The monitorability gain appears only when the task **requires belief-state tracking**: the
relevant variable is a function of the *history* that cannot be read off any single observation,
so the model must maintain and update an internal summary of hidden state (partial observability
/ POMDP). When the variable sits in the current input, a vanilla LLM reads it off directly and
both models saturate — no gap. Examples, toy → safety-relevant:

- **Toy:** localization from local observations (our gridworld); the shell game (track the ball
  through swaps); a hidden counter; hidden-information card games.
- **Language:** entity / world-state tracking and theory-of-mind / false-belief (where does *X*
  think the object is); long-context fact updates (a value set early, overwritten later — track
  the *current* one, don't copy); program state while reading code; dialogue state.
- **Agentic:** environment state across tool calls (what files exist, what's been done);
  permission / boundary state (does the agent currently hold a credential, is it past a sandbox
  boundary, did the user authorize this earlier); plan / subgoal state.

**Why this is the safety-relevant regime, not a niche one:** the variables a monitor most wants
are *themselves* belief-state variables — "is this action irreversible *given history*," "did the
user consent earlier," "is the model still under that hazardous instruction from 50 turns ago,"
"has it crossed a sandbox boundary." None are in the current token; all are integrated over the
trajectory. So the regime where predictive objectives help is exactly the regime safety cares
about. The important exception is **behaviorally-decoupled** state (dormant deception that never
affects near-future tokens): that is *not* a belief-state variable in this sense, and is where
this approach is least likely to help (the expressed-vs-decoupled framing in the agenda).

## TL;DR

- **Predict-ahead training really does improve belief-state monitorability** — large effect,
  but only once the task actually requires belief-state tracking (in easy/fully-observed
  regimes everything saturates and the test can't discriminate). It **survives a 4× scale-up
  of the baseline** — in fact the gap *widens*, because the larger vanilla model diffuses its
  belief state across more dimensions (so it is not a small-model artifact).
- **The monitorability gain is generic, not method-specific.** NextLat, MTP, and JTP all
  deliver it about equally. The durable claim lives at the level of the *class*.
- **The methods separate on other axes — compactness above all.** At matched monitorability,
  effective rank ranges ~3–6× across methods (NextLat most compact, MTP most diffuse — more
  diffuse than even a vanilla baseline).
- **But compactness turns out to be *safety-neutral* for monitoring (Step 2).** NextLat's low
  effective rank does *not* buy a cheaper, more sample-efficient, or more robust monitor: the
  task-relevant subspace is low-dimensional for *every* predictive arm and is decoupled from
  effective rank (the most *diffuse* arm, JTP, gives the lowest-dimensional monitor). So
  NextLat is not preferable for monitoring; its case for being *specifically* useful now rests
  entirely on its forward-rollout mechanism (Step 3, untested).
- So the durable claim is firmly the **class-level** one — *predictive-representation objectives
  make belief state monitorable* — and the open program is whether NextLat's trained
  latent-transition model enables a *predictive* monitor that a static probe can't (Step 3), and
  whether any of this survives off the toy (the language bridge).

## How we got here (origins)

The program began with **NextLat** (next-latent prediction; Teoh et al., NeurIPS 2025), whose
training pushes the hidden state toward a compact sufficient statistic for near-future tokens.
That motivated an **expressed-vs-decoupled** framing (detailed in the agenda):

- **Behaviorally-expressed risk** (about to act, reveal, cross a boundary) shifts the
  near-future token distribution, so a sufficient statistic must *preserve* it → should be
  more monitorable.
- **Behaviorally-decoupled risk** (dormant deception, sandbagging) is not predictive of
  near-future tokens, so a *minimal* sufficient statistic has pressure to **compress it away**
  → such objectives might make the hardest risks *harder* to see.

That framing is method-agnostic in principle, but it was posed about NextLat specifically. The
data is what broadened it.

## What Experiment 0 established

Exp 0 probes a matched **GPT vs a panel of predictive-objective models** in a custom gridworld
with full per-token label control: an expressed variable (trap-proximity) and a decoupled
variable (a secret bit, probed only in a controlled mid-trajectory dead-zone). Full writeup
and tables: [exp0.md](experiments/gridworld/exp0.md). Four findings, in order of impact on the thesis:

1. **A real, large monitorability gain — but only under partial observability.** In
   fully-observed regimes every model trivially, linearly encodes position and saturates. In a
   **partially-observable** variant where position must be integrated from local observations,
   the predictive objectives lift 81-way position decodability from the GPT baseline's
   **0.71 to ~0.95**.
2. **It is generic to predict-ahead training, not specific to any one method.** NextLat, MTP
   (Gloeckle), and JTP all reach ~0.94–0.95.
3. **It is genuine information availability, not linearization.** A strengthened nonlinear
   probe matches the linear probe on the predictive arms but **cannot exceed ~0.71 on GPT** —
   the baseline genuinely encodes *less recoverable* belief state.
4. **The methods dissociate on compactness.** Effective rank: **NextLat 30 vs JTP 100 vs GPT
   80 vs MTP 181**. Same decodability, very different compactness — NextLat alone is both
   highly decodable *and* highly compressed; MTP is decodable but more diffuse than the
   baseline.

Plus one update against the original worry: **the decoupled-deletion failure did not appear**
— every predictive arm retained the secret bit *better* than GPT. That falsifies the worry
*in this setting*; it does not test a regime where the model has incentive to hide the
variable.

| arm | 81-way position (linear) | strong-MLP | effective rank |
| --- | --- | --- | --- |
| GPT (no aux) | 0.71 | 0.65 | 80 |
| MTP | 0.94 | 0.94 | 181 |
| JTP | 0.95 | 0.95 | 100 |
| NextLat (h=8) | 0.95 | 0.95 | **30** |
| NextLat (h=1) | 0.95 | 0.95 | 32 |

*(Partial-observability gridworld, 4-layer/d=256, 3 seeds. The `h=1` ablation shows even a
minimal 1-step latent objective produces the compact, decodable state — the effect is not
about rollout depth.)*

## The reframed thesis

> **Predictive-representation objectives improve belief-state monitorability** — a class-level
> property, well-evidenced (in toy) and not a reason to prefer any one method for monitoring.
> The methods occupy **different corners of a design space**: NextLat is the
> *monitorable-and-compact* corner; MTP is *monitorable-but-diffuse*. **Step 2 then showed that
> the compactness corner is not advantaged: compactness is safety-neutral for the monitor (no
> cheaper / more sample-efficient / more robust monitoring).** So the only remaining way NextLat
> could be *specifically* useful is its trained latent-transition model enabling a *predictive*
> (forward-rollout) monitor that a static probe can't replicate — Step 3, untested.

This reframes the work from "is method X good for safety?" to a **comparative, design-space
program**: characterize what each objective does to internal state, and identify which
property is the one worth engineering for.

## Research program (prioritized)

Ordered by dependency: don't build on the monitorability finding until it survives scale, and
don't privilege any method until a distinguishing axis is shown to have value.

### 1. Scale-robustness check — *done: the gap holds (widens) at scale* ✓

The Exp-0 read rested on a **capacity-starved baseline** (4-layer GPT caps at 0.71). We re-ran the
identical partial-obs panel at **4× the backbone (8-layer/d=512)**, training extended so GPT is
genuinely converged. **Result: the gap is not a capacity artifact — it widens.** GPT's linear
position decodability *drops* to **~0.52** (3 seeds; the larger model diffuses position across
more dimensions; effective rank 80→221) while the predictive arms hold **~0.90–0.95**. The task-sanity
gate passes — GPT reaches the same val loss (~0.62) and decodes trap-proximity well, so it learned
a functional belief state and simply doesn't expose fine position linearly. The compression
dissociation sharpens too (NextLat rank ~36–46 vs MTP ~440 at equal decodability). → **Foundation
confirmed; greenlight step 2.** Full numbers: [exp0.md](experiments/gridworld/exp0.md) (Results — v3).

### 2. Does compactness buy a monitor anything? — *done: no, compactness is safety-neutral* ✓

We compared all arms at matched monitorability on the monitor-cost axes: (a) **low-dimensional
monitors** (PCA dims to reach 95% of full accuracy), (b) **sample-efficiency**, (c) **robustness**
to a short→long behavioral shift (which keeps the model in-distribution, isolating the monitor's
generalization). **Result: NextLat's compactness does *not* buy a cheaper, more sample-efficient,
or more robust monitor.** The task-relevant subspace is low-dimensional for *every* predictive arm
and is decoupled from effective rank — the most *diffuse* arm, **JTP**, gives the
*lowest*-dimensional monitor (4 PCs vs NextLat's 16), and all predictive arms are equally robust to
the shift. So compactness is **safety-neutral** on these axes, and JTP (a less-compact multi-token
objective) is the practical monitor winner. → NextLat is **not** preferable for monitoring; its
case for being *specifically* useful now rests entirely on Step 3. Full writeup:
[step2-compactness.md](experiments/gridworld/step2-compactness.md). *(n=1 seed; qualitative conclusion robust
across the curves.)*

### 3. Predictive forward-rollout monitor

*(After 1–2 motivate it.)* The most distinctive *mechanism* a predictive objective enables, and
still untested: methods that train an explicit latent transition model (NextLat's `p_ψ`) let you
roll the state forward over candidate actions and gate *before* emission. Question: does an
*online, predictive* monitor catch about-to-act risk earlier/more reliably than a reactive,
post-hoc probe? This is the real safety artifact ("act before the model does"); biggest build,
gated on the foundation holding.

### 4. Harder decoupled / deception test

*(Lowest priority.)* The Exp-0 decoupled probe was easy and showed no deletion. To actually
stress the deception worry, construct a setting where the model has an *incentive* to not
represent the secret. Only worth it to deliberately revive the expressed-vs-decoupled spine;
current evidence doesn't demand it.

### Open directions beyond the current panel

The design-space lens invites work past the three objectives studied so far: other predictive
objectives (e.g. longer-horizon or stop-gradient variants, BST-style training), other monitor
constructions (ensembles, calibration under shift), and eventually the move from synthetic
gridworlds to language and agentic settings — the candidate testbeds below.

### Candidate testbeds for the language / agentic bridge (worth exploring later)

These turn the belief-state examples [above](#which-tasks-does-this-apply-to-the-belief-state-regime)
into concrete next experiments. The selection criterion is the one the toy got right: a belief
variable with **clean per-token ground truth** that the baseline cannot read off the current
input. Ordered nearest-to-furthest from the gridworld:

- **Long-context fact-updates / variable binding** *(cleanest language bridge — best first step)* —
  a value is set early and overwritten later ("the key is 7" … 200 tokens later "the key is now
  3"); probe the *current* value at each token. Exact ground truth (we control it); it is
  essentially the toy's carried-bit idea scaled to language and long context.
- **Entity / world-state tracking & theory-of-mind** — track where an object is, who holds what, or
  what an agent *believes*, after a sequence of moves (Sally–Anne, bAbI, TextWorld-style). Ground
  truth from the simulator; tests richer multi-entity belief states.
- **Program / interpreter state** — track a variable's value or definedness as the model reads
  code; probe at each line, ground truth by execution. Safety-adjacent: the model's world-model of
  code it is about to run.
- **Agentic environment state across tool calls** *(the destination)* — probe "what files exist,"
  "has consent been obtained," "is the agent past a sandbox boundary," "is it still under a
  hazardous instruction from earlier." Where the safety-relevant belief variables actually live
  (and matching the agenda's Workstream-1 variable list), but labels are harder and the harness
  heavier — so it is the target, not the first step.

The throughline: keep the probe target a belief variable with controllable ground truth, escalate
realism from toy → language → agent, and at each rung re-ask the Exp-0 questions (does the gap
hold, is it generic, does compactness buy anything).

## Current status

- **Exp 0 (v1/v2/v2b + design-space + v3 scale):** complete. The monitorability gain is real,
  generic to predict-ahead training, **scale-robust** (survives a 4× backbone), and dissociates
  from compactness — which is NextLat's distinctive axis. [exp0.md](experiments/gridworld/exp0.md); results
  in [`docs/results/`](results/).
- **Step 2 (does compactness buy a monitor anything):** **done — no.** Compactness is
  safety-neutral; NextLat is not preferable for monitoring (JTP, a less-compact arm, is the
  practical winner). [step2-compactness.md](experiments/gridworld/step2-compactness.md). *(n=1 seed;
  seeds 1235/1236 not yet run.)*
- **Step 3 (predictive forward-rollout monitor):** the **only** remaining test that could make
  NextLat specifically useful; not started.
- **Step 4 + the language bridge:** planned, not started.

## Document map

- [`research-overview.md`](research-overview.md) — this file: the general question and program.
- [`feasibility-study.md`](feasibility-study.md) — current judgment on novelty, feasibility,
  risks, and the go/no-go logic for continuing the program.
- [`nextlat-ai-safety-research-agenda.md`](nextlat-ai-safety-research-agenda.md) — the original,
  NextLat-specific agenda: workstreams, the expressed/decoupled framing, decision criteria,
  risks, and the paper-vs-hypothesis evidential boundary. Still the deepest reference for the
  mechanism and the safety motivation.
- [`experiments/gridworld/exp0.md`](experiments/gridworld/exp0.md) — Exp 0 design, leakage controls, per-variant
  results (incl. the v3 scale check), and the verdict the reframe above is built on.
- [`experiments/gridworld/step2-compactness.md`](experiments/gridworld/step2-compactness.md) — Step 2: does
  compactness buy a monitor anything? (PCA / sample-efficiency / shift battery; verdict: no).
- [`results/`](results/) — per-run probe JSONs, aggregated summaries, plots.
- [`../experiments/`](../experiments/) — the Modal harness, standalone probe, analysis code;
  gridworld additions live in the cloned [`NextLat/`](../NextLat/EXP0_CHANGES.md).

## Honest caveats

All evidence to date is ≤~9M-parameter models on a synthetic gridworld — a controlled existence
proof, not a scale or language/agentic claim. Capability is matched on architecture/data/compute
but not on verified-equal next-token loss. The decoupled result is the *opposite* of the
original worry here, which is good for monitoring but means this environment does not exhibit the
compression-deletion failure mode. Step 1 is precisely the test of whether the headline survives
leaving the toy's smallest regime.
