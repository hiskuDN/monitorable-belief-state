# Pretrained-LM Transfer — Does the Carry Effect Survive in a Model That Already Knows Language?

Status: **scoped, not built.** The scale/realism step. Every prior result (Exp-0 gridworld, 2a
concealworld, storyworld) rides a **~7M-parameter, from-scratch** GPT and ends on the same caveat:
*not a real pretrained LLM.* This experiment attacks that caveat head-on — it re-runs the
carry-vs-defer test on a **pretrained** language model (Pythia), fine-tuned **with vs without** the
NextLat-1step auxiliary, and asks whether the split reproduces. It is the single experiment that
moves the program from "toy" to "this reproduces in a model that already knows language."

## The one-line question

> Take a real pretrained LM. Fine-tune the *same init* twice — once with plain LM loss, once with LM
> loss + NextLat-1step aux — on a belief-tracking story. Does the **carry-vs-defer split** reproduce:
> the aux-trained model keeps the belief **linearly monitorable** across the story while the plain
> fine-tune defers it to the query? Or was the split an artifact of training tiny models from scratch?

A positive says the objective-as-monitorability-lever is a property of *the objective*, not of the
from-scratch regime — and that it survives a language prior. That is the claim the whole series has
been implicitly promising and the difference between a workshop result and a main-track one.

## Why *within-model* (the design lever that from-scratch can't give us)

Every prior arm compares **two different training runs** (a GPT init trained one way vs a NextLat init
trained another). That confounds *the objective* with *two distinct optimization trajectories and
inits*. Capability matching has been approximate (architecture/data/compute, never verified-equal
next-token loss — a standing caveat in 2a).

A **pretrained backbone dissolves the confound.** Both arms start from *the identical Pythia
checkpoint*, see *identical data*, run *identical steps*. The **only** difference is whether the aux
loss is on. This is the cleanest possible isolation of the objective in the whole program, and it is
only available once we stop training from scratch. The realism upgrade and the rigor upgrade are the
same move.

## The headroom inversion (the new failure mode to pre-register)

2a-v1 died of **too-easy** (a fixed-position secret broadcast for free). story_v1 died of **too-hard**
(capability confound, all arms dark). This experiment's failure mode is a *third* thing and the one
most likely to bite: a pretrained 160M model may **ace `story_easy` and defer nothing worth
measuring** — capability-side headroom collapse. A model that already parses English will likely
track a ball across four shelves and six moves without breaking a sweat, saturating the probe for
*both* arms and leaving no gap.

**So the task moves up, not sideways.** The **frontier** storyworld config that *frayed* the 7M model
(K=6, more moves, real "it"-coreference, two distractors) is the natural starting difficulty here,
and we expect to push further. The headroom gate below is load-bearing: **vanilla-FT must be
off-ceiling in the masking window before any gap is trusted.**

## Environment / task (`NextLat/data/storyworld.py`, BPE variant)

Reuse the **storyworld** generator (not concealworld — its `u{d} wait` symbol soup carries no language
prior, which defeats the entire point of a pretrained backbone). The task stays: a short narrative
moves a ball among K numbered shelves by relative moves, a distractor object shuffled alongside, then
"Where is the ball?". The probed belief is the **running shelf** — a nonlinear running total, never a
surface token after line 1 (the standing design invariant).

The one real change is the **tokenizer**:

- **Render on the pretrained BPE tokenizer** (Pythia's GPT-NeoX tokenizer), not the synthetic
  `StoryworldTokenizer`. The text is already templated English; it just gets tokenized as real
  subwords.
- **Label alignment.** The per-token belief label must be re-aligned to subwords: map each belief
  label to the **last subword** of its span. Probe the running shelf at the **masking positions**
  (mid-story), which remain non-surface — never at the answer word.
- **Leakage audit on BPE-tokenized data.** Re-run the two-disjoint-RNG leakage idiom and re-assert:
  running shelf uniform at every offset, balanced answer, moves ⊥ answer — now over the *subword*
  stream, since BPE re-segmentation is a fresh chance to leak. `frac_with_window == 1` after
  re-alignment.
- **Capability gate** unchanged: the model's own next-token argmax at the query == answer shelf.

## Model / the graft (`NextLat/models/model_nextlat_hf.py`, new)

Verified: the NextLat aux is **loosely coupled** — `_nextlat_compute_losses`
(`model_nextlat.py:354-432`) touches only the **final hidden state**, the **input token embeddings**
(dynamics-MLP input, `:411`), and the **LM head** (`:376`). No attention/RoPE/block internals. So we
wrap, not reimplement.

- `NextLatHF` holds an `AutoModelForCausalLM` (Pythia-160M → 410M). Forward with
  `output_hidden_states=True`; lift `NextLatDynamicsModel` + `_nextlat_compute_losses` **verbatim**,
  computing MSE against later-position hidden states + KL through `get_output_embeddings()`.
- **Drop the two-phase Fabric backward** (`model_nextlat.py:488-582`) — that detach/manual-grad trick
  is a gradient-accumulation memory optimization, not a correctness requirement. Use a single
  `loss.backward()` on non-detached hidden states.
- **Probe-reuse contract.** The probes call `model.model(inp, return_all_layers=True)` expecting
  `(logits, [per_layer_state, …, final_state])` and read `.lm_head.weight`
  (`experiments/storyworld/probe.py:71,357`). `NextLatHF` exposes `.model` translating HF's
  `hidden_states` tuple into that `(logits, list)` shape and surfaces `.lm_head`. ~20-line adapter →
  the entire probe/analyze stack runs untouched.
- One new branch in the flag selector (`core_train.py:38-60`), mirroring `use_nextlat`.
- **Avoid** the legacy `init_from: gpt2` path — flagged broken (`core_train.py:184`), forces
  `bias=True` while the arch is `bias=False` (RMSNorm/SwiGLU); architectures diverge. HF-wrap is the
  clean route.
- **Full fine-tuning, not LoRA** — the claim is that the objective *reshapes* the representation;
  LoRA would confound that with adapter capacity.

## Training arms (`NextLat/config/pretrained/`)

Two arms, **identical Pythia init, identical steps/data/tokens**, differing only in the aux loss:

- **`vanilla_pythia`** — LM loss only (the defer baseline; the plain-GPT analog, now sharing B's init).
- **`nextlat_h1_pythia`** — LM loss + `lambda_mse=1.0`, `lambda_kl=0.1`, `mtp_horizon=1` (the carrier).

×3 seeds × {Pythia-160M, Pythia-410M}. **Bonus arm if cheap:** `nextlat_h8_pythia` — does the
horizon-1 sweet spot (h1 3/3 vs h8 1/3 in 2a) survive pretraining, or does a language prior stabilize
longer horizons? A cheap, high-interest side result.

## Probe (`experiments/pretrained/`, reuses `storyworld/probe.py`)

Frozen checkpoint → per-layer hidden states (via the adapter) → linear-first probe for the running
shelf, unchanged from storyworld:

- **Retention curve** — per-offset linear-probe accuracy vs story position, with per-offset chance
  (`1/K`) + shuffled floors. The headline: vanilla-FT decays toward chance across the mask, aux-FT
  stays flat.
- **Gather-vs-carry** — decode at last-masking token vs the query; a big `jump` (fork − last_wander)
  = defer. Expect vanilla-FT jumps, aux-FT flat.
- **Availability vs linearization** — linear (primary) + MLP floor. If the pretrained vanilla-FT keeps
  the belief in an MLP-recoverable-but-linearly-tangled form, the aux's gain is *linearization* (still
  monitor-relevant, narrower claim). Reported as separate pre-registered outcomes, as in 2a.

## Decision criteria (linear probes, primary = masking-window retention)

- **aux-FT carries > vanilla-FT** (retention gap in the masking window, GPT off-ceiling; corroborated
  by gather-vs-carry): **the carry effect transfers to pretrained LMs.** Main-track-grade — scale the
  ladder (410M → 1B) and write it as the series headline.
- **Both carry** (the pretrained prior already carries beliefs): a genuine finding, not a null —
  belief-carrying is partly a *language prior*, and the objective's marginal value shrinks with
  pretraining. Reframes the claim; still publishable, and interesting on its own.
- **Both defer / low**: task too hard at this size, or the aux doesn't transfer through pretraining;
  inspect retention + gather-vs-carry, drop difficulty or scale the backbone before concluding.
- **Vanilla saturates (≈1.0)**: no headroom — raise K / lengthen stories / add coreference and
  distractors (the gate below guards against reading a saturated gap).

## Headroom gate (before any full sweep)

A **vanilla-FT-only smoke** on the frontier config, confirming (a) vanilla-FT is **off-ceiling** on
masking-window retention (not ≈1.0), and (b) the task is learnable (query accuracy high → the model
*can* compute the answer, validating that a flat vanilla curve means defer, not incompetence). If it
saturates, harder config. Only then the 2-arm (+bonus) × 3-seed × 2-size sweep.

## Honest caveats (pre-registered)

- **Capability match** — read the gap only where *both* arms pass the capability gate; same rule as
  2a, now with the confound largely dissolved by shared init.
- **Objective vs compute** — identical step/data/token budget across arms is what makes the
  within-model comparison load-bearing; enforce it in `modal_app.py::main`.
- **Tokenization confound** — the belief must stay a **non-surface running aggregate** under BPE
  re-segmentation; the re-run leakage audit on subwords is the guard. A shelf *number* is a surface
  token at the query — we never probe there.
- **Pretrained-prior confound** — a pretrained model may carry beliefs for free; the vanilla-FT
  baseline controls for exactly this. If vanilla-FT carries, that is the "both carry" finding above,
  not a failure.
- **Still synthetic** — templated text read by a real tokenizer is a step onto language, not natural
  web text. A genuine-web belief task (authorization provenance, plan-consistency) is a further step,
  not this one.
- **Catastrophic forgetting** — the aux term can wreck LM ability at high λ; monitor generation
  quality and tune λ down from the from-scratch defaults if the LM loss diverges.
- **Horizon** — as in 2a, the carrier only holds the belief within its prediction horizon's reach, so
  the retention curve decays with distance from the query; the curve must land in a readable range.

## Compute / Modal

Full-FT of Pythia-160M/410M on synthetic templated text is **cheap** — tens of minutes to ~1–2h per
seed on a single A100; the whole 2-arm × 3-seed × 2-size matrix is roughly **one GPU-day**. New
`modal.App("nextlat-pretrained")` + `Volume.from_name("nextlat-pretrained")`, mirroring
`modal_app.py` (uniform step budget in `main`, `size`/`config` fan-out, `_save_local` →
`docs/results/<date>/<tag>/`). `build_image()` gains `transformers` + `torch` HF weights (or pre-bake
Pythia into the image / cache on the volume to avoid re-download per run).

## Implementation map

- `NextLat/models/model_nextlat_hf.py` — **new.** `NextLatHF` wrapper (dynamics MLP + lifted
  `compute_loss` body, single backward, `.model`/`.lm_head` probe adapter).
- `NextLat/core_train.py` — one new branch in the `initialize_model` selector
  (`use_nextlat_hf` / `backbone: pretrained`), + skip the broken `init_from: gpt2` path.
- `NextLat/data/storyworld.py` — a BPE-tokenizer render path (text + subword label alignment) and a
  re-run `leakage_audit` over subwords; or a sibling `storyworld_bpe.py` if cleaner.
- `NextLat/config/pretrained/{vanilla,nextlat_h1}_pythia{160m,410m}.yaml` — matched arms (backbone
  name, λ knobs, `mtp_horizon`, frontier difficulty), + optional `nextlat_h8`.
- `experiments/pretrained/{probe adapter, modal_app.py, analyze.py}` — probe reuses
  `storyworld/probe.py` behind the adapter; new Modal entrypoints (`smoke` for the headroom gate,
  `main` for the sweep).

## Relation to the rest of the program

The **scale/realism step**. Extends storyworld (same task, real backbone) and closes the caveat every
post ends on. Composes with the open horizon question from 2a (does the h1 sweet spot survive a
language prior?). Natural precursor to a genuine-web belief task
([toolworld](../toolworld/README.md)-style authorization/provenance) and to the main-track paper. If
positive, it is the result the series was built toward; if "both carry," it reframes the objective's
value as complementary to pretraining rather than a substitute for it.
