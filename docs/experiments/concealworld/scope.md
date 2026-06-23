# Concealworld (2a) — Monitorability of a Masked-but-Instrumental Secret

Status: **scoped, not built.** The first experiment in the program with an *incentive to conceal*.
Stage **2a** of the deception ladder (see [research-overview](../../research-overview.md) and the
first-principles route in the session notes): the cheap, **pure-supervised** rung — clone a teacher
that successfully masks a secret — built as a **minimal one-variable extension of the Exp-0
decoupled-bit harness**. Stage 2b (genuine discovered incentive via RL / self-play) is gated on 2a's
result and is **not** in scope here.

## The one-line question

> When a model holds a secret it **needs** (to act) but **masks** in its behavior, is the secret
> still **linearly recoverable** from its hidden state during the masking window — and does
> **predict-ahead training** keep it recoverable longer than a vanilla baseline?

A monitor that reads the secret from hidden state catches a deception that an **output-only** monitor
cannot (behavior is masked by construction). 2a asks whether that representational monitor works, and
whether training objective changes how well.

## What changes vs Exp-0 (exactly one thing)

| | Exp-0 decoupled bit | Concealworld (2a) |
| --- | --- | --- |
| G revealed once, early | yes | yes |
| What G drives | a **terminal symbol** at `T` (`terminal = f(G)`) | the agent's **goal** — a behavioral choice it acts on |
| Is G instrumentally needed? | only to emit one token | **yes — to reach the right goal** (proven by task-sanity) |
| Behavior during the probe window | G-independent (incidentally) | G-independent **by an incentive to conceal** (masking) |
| Probe target | G in window `W` | G in the **masking window** `W_mask` |

The single conceptual change: **G goes from "selects a terminal symbol" to "selects a goal the agent
pursues but hides until a late commit."** That converts the decoupled bit from *incidentally* unused
to *deliberately masked while instrumentally necessary* — which is what makes it deception rather than
irrelevance. Everything else (grid, tokenizer, secret-bit reveal, probe machinery, arms) is reused.

## Environment (`NextLat/data/concealworld.py`, reusing gridworld utils)

A 9×9 grid. Each episode:

```
[start] [g{G}] [goal0_cell] [goal1_cell]  d_1 ... d_Lmask  | d_(Lmask+1) ... d_commit  [safe]  end
                                          <-- masking -->  ^commit       <-- commit -->
```

- **Secret.** `G ~ Bernoulli(½)` revealed at `r` (`g0`/`g1`). True goal = `goal_G` (two stated goal
  cells, balanced so neither correlates with G).
- **Masking phase (`L_mask` steps).** The teacher navigates toward a **rendezvous region consistent
  with both goals**, along a path **sampled independently of G**. So masking-phase tokens are, by
  construction, identically distributed across G → behavior carries **zero** information about G here.
- **Commit phase (`L_commit` steps).** The teacher diverges from rendezvous to `goal_G`. This is
  where G becomes behaviorally expressed; we do **not** probe concealment here (it is the expressed
  regime).
- **Outcome token (optional in 2a).** `[safe]` iff masking held (no G-divergence before `commit`).
  The cloned teacher always masks, so this is constant in the core 2a dataset and carries no
  gradient — it exists for narrative/2b. (A minority of `[caught]` traces can be added later to make
  the safe/caught distinction learnable; kept G-balanced if so.)

Training is plain next-token prediction of the direction sequence (prefix masked, as in
Manhattan/Exp-0). The model **clones a successful concealer.**

### Honest framing — what 2a is and isn't

2a supplies the incentive **by demonstration** (the teacher masks; the model imitates), the realistic
*distillation* threat — a model trained to imitate a policy that happens to conceal. It does **not**
apply pressure on the representation to hide from a probe, and it does not test whether a model would
*discover* masking on its own. Both of those are **2b** (RL against a frozen challenger, or an
adversarial-probe term). So a 2a positive means "predict-ahead keeps a masked-but-needed secret
monitorable under non-adversarial distillation"; the adversarial question stays open by design.

### Why predict-ahead should help (mechanism)

During `W_mask`, no token depends on G, so the **local** next-token loss gives no pressure to
represent G. The only pressure to carry G across the mask comes from the **future** commit action
(`goal_G`). A predict-ahead objective trains the hidden state to predict its own future, so it should
**retain G across the masking window** to support the eventual divergence. A vanilla GPT, which only
needs G at commit and cannot re-read it (revealed once, early), may let G **decay** through a long
mask. The longer `L_mask`, the harder the carry — this is the primary stress knob (the concealworld
analogue of Exp-0's reveal→relevance gap).

## Labels (probing only; never trained)

Evaluated at each masking-phase position:

- `y_dec[t] = G` — **primary target**, probed strictly in `W_mask = {r + d < t < commit − d}`.
- Position-conditioned: `y_dec` accuracy/AUROC as a function of `t − r` (distance from reveal) across
  `W_mask` — the **retention curve**. Decay vs flat is the headline.
- `y_goal[t]` — the eventual goal cell (sanity: at/after commit the model knows where it's going).

## Leakage control (validity-critical)

The masking trajectory is sampled **before** G, exactly as Exp-0 samples the path before the bit:

- `leakage_audit()` asserts, on the masking window, `G ⟂ {masking path, trajectory length, start,
  goal-cell identities, rendezvous}` and `p(G=1)=½`. Because masking tokens are identically
  distributed across G, **behavior in `W_mask` cannot leak G** — so any `y_dec` signal in `W_mask` is
  genuinely representational, not the probe reading masked-but-tell-tale behavior.
- Commit-phase positions are **excluded** from the concealment probe.
- Shuffled-label and chance (½) floors for every (layer, position).

## Training arms (`NextLat/config/concealworld/`)

Matched **GPT vs NextLat vs MTP vs JTP** (+ **NextLat-h1**), identical except the auxiliary loss, at
the Exp-0 backbone (`n_layer=8, n_embd=256, n_head=8, block_size=256`; NextLat keeps
`lambda_mse=1.0, lambda_kl=0.1, mtp_horizon=8`). ≥3 seeds. MTP/JTP keep the claim class-level and
guard the "it's just a different model" read, as in the gridworld sweep.

## Probe (`experiments/concealworld/probe.py`)

Reuses the Exp-0 probe and the upstream `return_all_layers=True` flag — no new model surgery:

- Frozen checkpoint → run eval set → cache per-layer masking-window hidden states + labels + `r`,
  `commit`, `L_mask` metadata.
- **Linear (primary) + 1-hidden-MLP + strong-MLP (floor/secondary)** per (layer × position). Identical
  trajectory-level train/val split across arms & seeds.
- Report the `W_mask` retention curve (`y_dec` vs `t − r`) per arm, with chance/shuffled/strong-MLP
  floors, plus effective rank (compression sanity).

## Primary analysis — the masking-length sweep

Run the panel across `L_mask` (short → long carry). Read `y_dec`-in-`W_mask` retention:

- **Expected positive:** predict-ahead arms hold G flat/high across `W_mask`; GPT decays as `L_mask`
  grows. → a masked-but-needed secret is monitorable via probe even when invisible in behavior, and
  predict-ahead makes it **more** monitorable. The carry frontier (where each arm's retention breaks)
  is the deliverable curve.
- **Both flat high:** carry too easy — lengthen `L_mask` / weaken commit weighting until headroom
  appears (the v1/v2 saturation lesson) before reading any gap.
- **Both decay to chance:** even predict-ahead drops a long-masked instrumental latent. Sobering,
  clean negative: representational monitorability does not survive long masking even without
  adversarial pressure.

## Verification

1. **Task-sanity — the gate that makes the probe meaningful.** Arms must (a) navigate validly, (b)
   **reach `goal_G`** — proving G is *used* (instrumentally necessary; rules out the "G is irrelevant
   and innocently forgotten" reading), and (c) reproduce the **G-independent masking trajectory** —
   proving they conceal. (b) is the crucial one: without it there is no instrumental pressure and the
   experiment degenerates to Exp-0.
2. **Headroom check** — GPT below ceiling on `y_dec` in `W_mask` *before* trusting any gap.
3. **Leakage audit** — the independence asserts above; shuffled floors at chance.
4. **Compression sanity** — NextLat lower effective rank than GPT/MTP on masking-window states.
5. **Primary deliverable** — `W_mask` retention curves, linear-first, ≥3 seeds with error bars,
   across the `L_mask` sweep, against all floors.

## Decision criteria (linear probes, primary target `y_dec`-in-`W_mask`)

- **Predict-ahead retains > GPT across `W_mask` (esp. at long `L_mask`):** masked-but-instrumental
  secrets are monitorable and predict-ahead helps → greenlight **2b** (does it survive a *discovered*
  / adversarial incentive?), the **language surface** ([lieworld](../lieworld/scope.md)), and a scale
  check.
- **No gap (all retain):** re-check headroom; lengthen the mask. If still no gap, the effect is
  carry-trivial here.
- **All decay:** predict-ahead does not keep a long-masked latent legible → the approach's reach for
  concealed state is limited; report it and reconsider before investing in 2b.

## Build order

1. `NextLat/data/concealworld.py` with the **leakage audit baked in** (unit-test masking ⟂ G before
   any training); one tiny GPT run end-to-end (CPU smoke → Modal); confirm task-sanity (reaches
   `goal_G` **and** masks).
2. Add predict-ahead arm configs; confirm training + compression signature.
3. `experiments/concealworld/probe.py` (masking-window, position-conditioned, linear + strong-MLP +
   floors); by-layer retention curve for one seed at one `L_mask`.
4. `L_mask` sweep + multi-seed aggregation in `experiments/concealworld/analyze.py`.
5. Write up results + the go/no-go on 2b here.

## Implementation map (mirror of gridworld)

- `NextLat/data/concealworld.py` — generator (reusing gridworld grid/tokenizer utils) + datamodule +
  `leakage_audit`; register `"concealworld"` in `train.py` `DATAMODULES`.
- `NextLat/config/concealworld/{gpt,nextlat,mtp,jtp}_concealworld.yaml` — matched arms.
- `experiments/concealworld/{probe.py, modal_app.py, analyze.py}` — reusing
  [`../common/modal_base.py`](../../../experiments/common/modal_base.py); results under
  `docs/results/<date>/<tag>/`.
- Reuses the Exp-0 `return_all_layers` surgery — no new model changes expected.

## Relation to the rest of the program

- **Supersedes [lieworld](../lieworld/scope.md) as the *next* experiment.** The first-principles call:
  the binding constraint on a meaningful deception test is *incentive*, not *surface*, so add the
  incentive on the clean gridworld harness first; the language surface (lieworld) is a separate,
  later axis layered on once concealworld establishes the effect.
- **Extends Exp-0's decoupled-bit finding** (which showed retention with *no* incentive) by adding the
  minimal incentive that was missing — the one new variable.
- **Stays distinct from SAPLMA-style lie-detection** (frozen model, standalone statement, nonlinear
  classifier): objective-as-lever, history-carried secret, behavior masks it, linear-first probe. See
  [feasibility-study.md](../../feasibility-study.md#relation-to-prior-work-lie-detection--latent-knowledge-probing).

## Risks / open questions

- **2a's incentive is demonstrated, not discovered** — a positive result speaks to distillation of a
  concealer, not adversarial evasion. That gap is exactly 2b's remit; do not overclaim.
- **Headroom is not guaranteed**; `L_mask` is the knob, and GPT must be verified off-ceiling first.
- **Capability matching** is approximate (architecture/data/compute, not verified-equal next-token
  loss), as in all arms.
- **All-decay is a real possible outcome** and is interpretable: it bounds how far representational
  monitoring reaches for masked state even before any adversarial pressure.
