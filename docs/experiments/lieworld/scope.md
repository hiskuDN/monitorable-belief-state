# Lieworld — Suppressed-Truth Monitorability under Induced Deception

Status: **scoped, deferred behind [concealworld](../concealworld/scope.md).** The *language-surface*
rung of the deception program. Tests whether predict-ahead training (NextLat / MTP / JTP) makes a
model's *suppressed true belief* more linearly monitorable than a vanilla GPT — in a setting where the
model is induced to **state something false** while we probe whether the truth is still legible.

> **Sequencing note (first-principles route).** The binding constraint on a meaningful deception test
> is *incentive*, not *surface*. So the program adds the incentive first on the clean gridworld
> harness — [concealworld (2a)](../concealworld/scope.md) — and treats this language surface as a
> *separate, later* axis layered on once concealworld establishes (or bounds) the effect. Build
> lieworld only after concealworld reads out.

This is the first experiment that touches "is the model lying," and it is deliberately designed to
stay novel against the lie-detection / latent-knowledge probing line (SAPLMA et al.) — see
[Relation to prior work](#relation-to-prior-work) and the positioning note in
[`../../feasibility-study.md`](../../feasibility-study.md).

## The question

> When a model emits a **false** answer, is the **true** state it is contradicting still **linearly
> recoverable** from its hidden state — and does **predict-ahead training** make that suppressed
> truth more monitorable than a vanilla baseline?

A monitor that can decode the truth and compare it to the emitted token has a **lie detector**:
`probe(truth) ≠ emitted_token ⇒ the model is lying`. The safety question is whether training
objective changes how reliably such a monitor works, in the regime where the truth must be *tracked*
across context rather than read off the current token.

## Why this is the right next language experiment

- It is the cleanest controllable instance of "lying" that preserves **exact per-token ground truth**
  (we author the world state, the deception directive, and the emitted answer), so it inherits the
  Exp-0 label discipline.
- It carries the program's two differentiators against prior lie-detection work intact: a
  **train-from-scratch objective contrast** and a **belief-state (history-integrated) target**, with
  the added twist that the target is **suppressed** at the output (the model is actively saying
  something else). SAPLMA and relatives test neither.
- It sits on the expressed-vs-decoupled spine: the truth's monitorability under deception is exactly
  the variable the agenda's framing makes predictions about, and the design exposes a tunable
  **expressed→decoupled axis** (below) that turns the qualitative worry into a curve, as Exp 0 did
  for the reveal→relevance gap.

## Environment (`NextLat/data/lieworld.py`)

A compact symbolic QA-with-deception task, encoded for next-token training. Each example:

```
[fact_1] [fact_2] ... [fact_K]   <mode>   q: <obj> ?   a: <color>   end
   the box is red . the ball ...  H | D                 (emitted)
```

- **World state.** `K` objects (e.g. `box ball cup key lamp`), each assigned a color from a palette
  of `C` colors (e.g. `red blue green yellow`). Facts are stated in language-flavored templates
  (`the box is red .`). Only one object is queried, so the model must **bind query→fact**, not just
  remember the last token.
- **Fact updates (headroom — the crux).** A fraction of objects are **re-colored later** in the
  trace (`the box is now green .`); the probe target is the **current** color at query time. This is
  the long-context variable-binding mechanism that forces history integration and keeps the baseline
  off the ceiling (the v1/v2 saturation lesson — see [exp0](../gridworld/exp0.md)). Distance between
  the binding fact and the query is padded with filler facts so the truth must be *carried*.
- **Mode token.** `<honest>` (H) → the emitted answer is the **true** current color. `<deceive>` (D)
  → the emitted answer is a **false** color (per the lie policy below). The model is trained, by
  plain next-token prediction, to obey the mode — i.e. it **learns to lie on command**.
- **Lie policy (the expressed→decoupled axis).** How the false color is chosen controls how
  *instrumentally necessary* the truth is for producing the lie:
  - **derived (`lie = f(truth)`), expressed end:** no lie color is given; a fixed rule maps truth to
    output (e.g. "next color in the palette cycle"). To emit the correct lie token the model **must
    hold the true color** at the answer position → the truth is behaviorally expressed (in the
    near-future token distribution). Favorable regime; expected predict-ahead win.
  - **given-independent (`lie ⟂ truth`), decoupled end:** a specific wrong color is supplied in
    context (`... say blue .`), drawn uniformly from colors `≠ truth`. The truth is **not needed** to
    emit the lie → at the answer position the truth is behaviorally decoupled. Hard regime; the real
    deception stress test (closest to [safety-task-menu](../../safety-task-menu.md) #11). Also the
    **cleanest read**, because `output ⟂ truth` by construction (see leakage control).
  - (optional middle rung: a 2-of-3 mixed policy to get a third point on the axis.)

The generator emits, per example: `token_ids`, `mode`, `queried_obj`, `true_color` (current, post-
update), `output_color`, `lie_policy`, and per-token position metadata (fact span, mode position,
answer position).

### Why predict-ahead might help — and the honest null

Predict-ahead objectives push the hidden state toward a sufficient statistic for the future. In the
**derived** regime the lie is a function of the truth, so the truth is instrumentally necessary for
the next tokens and a sufficient statistic must retain it → predict-ahead should retain it *more
legibly* (the Exp-0 mechanism, on a deception surface). In the **given-independent** regime the truth
is irrelevant to the next tokens, so a *minimal* sufficient statistic has pressure to **compress it
away** — predict-ahead might not help, or could even hurt. **Both outcomes are informative:** the
first transfers the monitorability result to language deception; the second is direct evidence that
deception decoupled from behavior is exactly where this approach stops helping (the agenda's scope
boundary, made concrete). The design is built so the negative result is interpretable, not just a
shrug.

## Labels (probing only; never used in training)

All evaluated at the **answer position** (the hidden state used to predict the color), and at a
**pre-answer position** (the `<mode>` / `q:` token) as a carried-belief control:

- `y_true[t]` = current true color of the queried object (`C`-way). **Primary target.**
- `y_out[t]` = the color the model is about to emit (`C`-way). Sanity: should be high for all arms
  under both modes (the model knows what it is saying).
- `y_mode[t]` = honest vs deceive (binary). Sanity / "is deception itself legible."
- Derived metric — **lie-detectability:** decode `y_true`, compare to `y_out`; flag mismatch. Report
  detector AUROC for "is this a lie" under each lie policy.

## Leakage control (validity-critical)

The world state and the deception directive are sampled **independently**, and the audit
(`leakage_audit()`, run at datamodule construction and again in the probe) asserts:

- `true_color` is uniform over the palette and `⟂ {queried-object identity, position, mode, trace
  length, fact-update presence}`; mode is 50/50.
- **In the given-independent policy, `output_color ⟂ true_color`** — so decoding `y_true` from the
  hidden state cannot be the probe secretly reading the about-to-be-emitted token. This is what makes
  the decoupled condition the *clean* suppressed-truth read.
- The **derived policy is output↔truth-invertible**, so its answer-position `y_true` number is
  confounded with reading the staged output. Controls for it (do **not** trust the raw number alone):
  1. report `y_true` at the **pre-answer position**, before the answer token is formed, where the
     belief is carried but the output is not yet staged;
  2. require the **honest ≈ deceive** equality on `y_true` (if the truth is equally recoverable
     whether or not the model is about to say it, the probe is reading belief, not output);
  3. report the **strong-MLP-on-GPT floor** — if a nonlinear probe on GPT cannot recover the truth
     that a linear probe on a predict-ahead arm can, the gain is genuine availability.
- Shuffled-label and chance (`1/C`) floors for every (target, layer, position, policy).

## Training arms (`NextLat/config/lieworld/`)

Matched **GPT vs NextLat vs MTP vs JTP** (+ **NextLat-h1**), identical except the auxiliary loss, at
the Exp-0 backbone (`n_layer=8, n_embd=256, n_head=8, block_size=256`; NextLat keeps
`lambda_mse=1.0, lambda_kl=0.1, mtp_horizon=8`). ≥3 seeds per arm. The MTP/JTP arms are the
non-NextLat predict-ahead baselines that keep the claim class-level (and guard against an
"it's just a different model" read), exactly as in the gridworld design-space sweep.

## Probe (`experiments/lieworld/probe.py`)

Standalone, reusing the Exp-0 machinery and the `return_all_layers=True` flag already added upstream:

- Load each frozen checkpoint; run the eval set; cache per-layer hidden states at the answer and
  pre-answer positions plus all labels and policy/mode metadata.
- Fit **linear (primary) and 1-hidden-MLP + strong-MLP (floor/secondary)** probes per
  (layer × target × position × lie-policy × mode). Identical trajectory-level train/val split across
  all arms and seeds (deterministic eval regeneration), so arms differ only in representations.
- Report, conditioned on `(lie-policy, mode)`: `y_true` accuracy/AUROC by layer, the lie-detector
  AUROC, the linear-vs-MLP gap (genuine-info check), and chance/shuffled floors.
- Effective rank of answer-position states per arm (compression sanity / dissociation, as in Exp 0).

## The expressed→decoupled sweep (primary analysis)

Run the full panel across the lie-policy axis (`derived → [mixed] → given-independent`) and read
`y_true`-under-`deceive` monitorability as a function of how decoupled the truth is from the output:

- **Expected at the expressed (derived) end:** predict-ahead arms decode the suppressed truth well;
  GPT lags — the Exp-0 belief-state result reproduced on a deception surface.
- **Expected at the decoupled (given-independent) end:** retention degrades for everyone; the open
  question is whether predict-ahead degrades *slower* than GPT (compression-helps-expressed /
  hurts-decoupled signature) or all arms floor together.

This is the lieworld analogue of Exp 0's gap-vs-horizon sweep: one hyperparameter (truth-necessity of
the lie) turns the expressed-vs-decoupled framing into a falsifiable curve.

## Verification / how we read the result

1. **Task-sanity (gates everything).** Both/all arms learn to obey the mode: under `<deceive>` they
   emit the policy-correct false color, under `<honest>` the true color, at high accuracy. Without a
   model that reliably lies on command, the suppressed-truth probe is meaningless.
2. **Headroom check (the v1/v2 lesson).** Confirm the baseline is **below ceiling** on `y_true` under
   deceive *before* trusting any gap. If GPT already saturates, deepen the tracking burden (more
   objects, more updates, longer carry distance) until headroom exists.
3. **Leakage audit** — the independence asserts above, and shuffled floors at chance.
4. **Compression sanity** — NextLat shows lower effective rank than GPT/MTP on answer-position states.
5. **Primary deliverable** — `y_true`-under-deceive monitorability by layer, **linear first**, ≥3
   seeds with error bars, against chance/shuffled/strong-MLP floors, across the expressed→decoupled
   sweep; plus the lie-detector AUROC per arm and policy.

## Decision criteria (made on linear probes, primary = derived/expressed anchor)

- **Predict-ahead > GPT on `y_true`-under-deceive (derived):** the belief-state monitorability result
  **transfers to a language deception surface** → greenlight (a) the decoupled-end stress test as the
  real deception probe, and (b) a scale check, mirroring the gridworld program.
- **Predict-ahead ≈ GPT, both high:** likely the task is too easy — re-check headroom (#2) and
  deepen tracking before concluding.
- **Predict-ahead ≈ GPT, both low under deceive:** the truth is discarded once the lie is computed —
  deception is decoupled even in the derived case. Sobering but clean: this approach does not reach
  this form of lying, and we say so.
- **Across the sweep:** retention high at the expressed end, degrading toward the decoupled end; if
  predict-ahead degrades slower than GPT, that is the compression-helps-expressed signature and the
  most interesting positive outcome.

## Build order

1. Build `NextLat/data/lieworld.py` with the **leakage audit baked in** (unit-test the independence
   asserts before any training); get one tiny GPT run training end-to-end (CPU smoke → Modal),
   confirming the model learns to lie on command (task-sanity).
2. Add the predict-ahead arm configs; confirm they train and the compression signature appears.
3. Write `experiments/lieworld/probe.py` (answer + pre-answer positions, mode/policy conditioning,
   linear + strong-MLP + floors); produce the by-layer `y_true`-under-deceive table for one seed in
   the **given-independent** policy first (the clean, output⟂truth read).
4. Add the expressed→decoupled sweep + multi-seed aggregation in `experiments/lieworld/analyze.py`.
5. Write up results + the go/no-go call here.

## Implementation map (mirror of gridworld)

- `NextLat/data/lieworld.py` — generator + datamodule + `leakage_audit`; register `"lieworld"` in
  `train.py` `DATAMODULES`.
- `NextLat/config/lieworld/{gpt,nextlat,mtp,jtp}_lieworld.yaml` — matched arms.
- `experiments/lieworld/{probe.py, modal_app.py, analyze.py}` — probe, Modal entrypoints (reusing
  [`../common/modal_base.py`](../../../experiments/common/modal_base.py)), aggregation. Results land
  under `docs/results/<date>/<tag>/`.
- Reuses the upstream `return_all_layers` surgery already added for Exp 0 — no new model changes
  expected.

## Relation to prior work

Azaria & Mitchell's SAPLMA (and Burns et al., Marks & Tegmark, RepE) establish that truth-relevant
state is probe-recoverable from a **frozen** model reading a **standalone statement**, with a
**nonlinear** classifier. Lieworld differs on every one of those: the **training objective is the
variable**, the truth is a **history-integrated belief** that is **suppressed at the output** while
the model utters a falsehood, and the decision is made on **linear** decodability with the MLP only as
a floor. The full four-axis positioning is in
[`../../feasibility-study.md`](../../feasibility-study.md#relation-to-prior-work-lie-detection--latent-knowledge-probing).

## Risks / open questions

- **The derived-policy confound** (output↔truth invertible) is real; the clean read leans on the
  given-independent policy and the pre-answer position. If both come back ambiguous, redesign the lie
  policy to be many-to-one (non-invertible) so the output cannot pin the truth.
- **Headroom is not guaranteed** on a language-flavored surface; the fact-update/binding burden is the
  knob, and we must verify GPT is off the ceiling before reading any gap.
- **Capability matching** is approximate (matched architecture/data/compute, not verified-equal
  next-token loss), as in the gridworld arms.
- **Negative-at-decoupled is the likely and acceptable outcome** at the hard end; the experiment is
  designed so that is an interpretable finding about scope, not a failure.
