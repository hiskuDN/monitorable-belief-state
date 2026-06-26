# Storyworld — Belief Tracking on a Language Surface

The base rung of the **language track**: does predict-ahead's belief-state monitorability
(established symbolically in Exp-0 / concealworld) **survive a natural-language surface**?
A model reads a short templated **narrative** in which an object moves around; it must track
the object's **current location**, and we probe whether that belief is **linearly decodable**
from the hidden state — predict-ahead arms vs a vanilla GPT. *Lieworld* (the model is made to
lie/conceal) is the next rung, gated on this.

## The one-line question

Reading a narrative ("Alice moves the ball up two shelves. Then it slides down one. …"), is the
object's **current shelf** linearly readable from the model's hidden state mid-story — and does
a predict-ahead objective make it more readable than a vanilla GPT, as in the symbolic tasks?

## Why a running shelf (the concealworld v1→v2 lesson, carried over)

The belief must **not be a recent surface token**, or every arm reads it for free and saturates
(concealworld v1). So moves are **relative** — "up d" / "down d" — and the current shelf is a
**running sum mod K** of the deltas. It is a nonlinear function of scattered updates, uniform at
every position by construction (leakage-safe), and cannot be "computed once and carried as a
constant" because it keeps changing. This is concealworld's proven trick; the **new variable here
is the language surface**, not the hidden-variable structure.

## What makes it *language* (the genuine new test)

- **Light templated sentences** with function words and **coreference** ("Alice moves the ball up
  two shelves. Then **it** slides down one."), not bare symbolic tuples.
- **Distractor objects** moved in interleaved sentences — the model must **carry the target's shelf
  across irrelevant language tokens** and bind moves to the right object (the target is named; e.g.
  "the ball" is always the queried object in v1).
- An **end query** ("Where is the ball? → shelf3") so tracking is *required*; we probe the belief
  **during the narrative** (before the query) — the language analog of the gridworld
  belief-during-navigation read.

So the value over concealworld is the **surface** (coreference, distractors, sentence structure),
not real-LLM scale. This is a controlled, small-scale, from-scratch existence proof — framed
honestly as *language-surface robustness*, not a claim about pretrained LLMs.

## Environment (`NextLat/data/storyworld.py`)

Per episode: a fixed-length interleaving of **target** move-sentences and **distractor**
move-sentences, then the query + answer. The target's running shelf `S_t = (Σ target deltas) mod K`
is the probe target (defined token-by-token across the narrative; flat during distractor
sentences — the model must *not* update its target-belief on distractor moves). Emits
`S_running` (per-token target shelf), `y_dec` (final shelf = answer), `in_window` (narrative
tokens), `is_target_update` (positions where a target move completes), fixed length `L`.

**Leakage control** (asserted by `leakage_audit`): target shelf **uniform over K at every probed
position** (mod-K with uniform deltas + first target move sets it); distractor deltas ⊥ target
shelf; `#target-updates` ⊥ final shelf; total length **fixed**. Positive control: the answer
token equals the final running shelf (task non-trivial).

## Probe (`experiments/storyworld/probe.py`)

Reuse the concealworld/Exp-0 probe machinery verbatim, target = `S_running` (target shelf):
per-offset-trained **linear** (primary) + **MLP** (availability floor) probes, per-offset chance
(1/K) + shuffled floors, and the **retention curve** (decodability vs narrative position). Probe
at **target-update** positions and at all narrative positions; the purest read is decodability
*between* target updates (the belief must be carried across distractor/coreference tokens).

## Training arms & knobs

Matched arms (`gpt`, `nextlat`, `nextlat_h1`, `mtp`, `jtp`), Exp-0 backbone (8L/256d), same
optimizer/aux-loss fields as concealworld. **Headroom knobs:** `n_states` (K shelves),
`story_len` (L), `n_distractors`, `target_density` (fraction of target vs distractor events),
`max_delta`. Word-level fixed vocab (a `SimpleTokenizer`, same pattern as concealworld).

## Decision criteria (linear probes, primary = between-update narrative positions)

- **predict-ahead tracks > GPT** (mean Δ > ~0.05, GPT off-ceiling): belief-state monitorability
  **transfers to a language surface** → greenlight *lieworld* (lie/conceal in language).
- **GPT saturates (≈1.0):** no headroom — raise `K` / `n_distractors` / `story_len`, re-run.
- **All low / no gap:** monitorability doesn't visibly transfer to language; inspect the retention
  curve and reconsider. An interpretable negative.

## Honest caveats (pre-registered)

- Synthetic **templated** language, trained **from scratch** — controlled, not real-LLM scale.
- The hidden variable is engineered (running mod-K shelf) for headroom; the language contribution
  is the **surface** (coreference, distractors), which we should not overstate as full NL.
- Capability gate: arms must **answer the end query** well at short stories (they *can* track) —
  a model that fails the task is a capability confound, not a monitorability result.

## Build order (confidence gates)

1. **Env + leakage_audit**, unit-tested locally (no training): uniformity over K at every probed
   position, distractor ⊥ target shelf, answer == final shelf, every record has a window. **Gate:
   audit green.**
2. **Register + CPU smoke** (`train.py`): trains end-to-end, loss drops.
3. **Arm configs + Modal training** (5 arms × ≥3 seeds). **Gate:** each arm answers the end query
   at short stories (can track).
4. **Probe + retention curve** for one seed/arm vs chance/shuffled/MLP floors. **Gate (headroom):**
   GPT below ceiling between target updates before trusting any gap.
5. **Aggregate** (`analyze.py`) → retention curves ± error bars, predict-ahead vs GPT. Write into
   this scope. Decision → lieworld.
