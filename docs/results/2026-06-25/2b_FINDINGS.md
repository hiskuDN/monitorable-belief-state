# Concealworld 2b — adversarial probe-evasion: INCONCLUSIVE (methods wall)

**Status: parked.** We tried to test whether predict-ahead's linear monitorability (2a)
survives a model *incentivized to evade the probe*. Two independent implementations both hit
the same wall: **gradient-reversal adversarial feature removal does not remove the feature**
(a co-trained probe is fooled; a *fresh* probe still recovers it). This is a known result
(Elazar & Goldberg 2018). So GRL is the wrong instrument, and we do **not** report a verdict
on adversarial robustness. We *do* establish that the hiding representation exists in
principle, which reframes the honest claim (below).

## What we ran

**Mechanism (committed `b77764b`, smoke-validated):** a gradient-reversal linear probe reads
the running secret S_t off the last hidden state; the probe trains to decode it while the GRL
pushes the trunk to hide it (strength `lambda_adv`), next-token loss untouched. GRL verified
gradient-correct (reversal cosine −1.000 vs a cooperative probe). `lambda_adv=0` ⇒ no head ⇒
byte-identical to 2a.

**Run 1 — end-to-end λ sweep** (`gpt, nextlat_h1` × λ∈{0,0.3,1.0} × 3 seeds, 32k steps,
tags `2b_l0.0/0.3/1.0`). Eval with the independent probe (fresh reader). **Incoherent:**

| arm | λ | fork(task) | late-lin | late-MLP |
|---|---|---|---|---|
| gpt | 0.0 | 0.83 | 0.26 | 0.28 |
| gpt | 0.3 | 1.00 | **0.88** | 0.88 |
| gpt | 1.0 | 1.00 | **0.99** | 0.99 |
| nextlat-1 | 0.0 | 1.00 | 0.61 | 0.91 |
| nextlat-1 | 0.3 | **0.66** | 0.27 | 0.28 |
| nextlat-1 | 1.0 | **0.46** | 0.34 | 0.34 |

GPT became *more* decodable under "attack" (representation collapsed to rank ~7–17, in which
the task-needed secret is more linearly prominent); NextLat's **task broke** (fork 0.46 = a
capability confound, not hiding). The **λ=0 controls reproduce 2a exactly** (GPT destroys 0.26;
NextLat carries 0.61/0.91), confirming the harness is fine and the pathology is specific to
adversarial training.

**Run 2 — frozen-features price test** (`price_of_concealment.py`): learn an invertible
encoder φ (reconstruction = predict-ahead-preservation proxy) + an ensemble GRL linear
adversary on the `nextlat_h1` carrier's late-window features; sweep β.

| β | recon_err | lin | mlp | chance |
|---|---|---|---|---|
| 0.0 | 0.000 | 0.48 | 0.97 | 0.26 |
| 0.3 | 0.000 | 0.65 | 0.96 | 0.26 |
| 1.0 | 0.002 | 0.48 | 0.96 | 0.26 |
| 3.0 | 0.039 | 0.54 | 0.87 | 0.26 |
| 10.0 | 0.144 | 0.52 | 0.50 | 0.26 |

The adversary **never reduces fresh-probe linear decodability** (lin ~0.5 regardless of β);
high β just degrades the whole representation (recon ↑, MLP ↓) without hiding the linear
signal. Same wall as Run 1.

## The wall

**GRL adversarial removal doesn't remove.** The trunk learns to fool the *specific co-trained*
probe, not to make S_t linearly unrecoverable in general; a fresh probe recovers it, and pushing
harder collapses the representation rather than finding a clean "encrypted" solution. Reproduced
in two independent setups. This is a methods limitation, not a science result.

## What we *did* establish

- **β=0 invertible recoding:** preserves the secret fully (MLP 0.97) and predict-ahead info
  (recon ≈ 0) while linear decode is only ~0.48. So even in the carrier the secret is encoded
  **more non-linearly than linearly**, and a representation that hides S_t from a *linear* probe
  while keeping it present **exists in principle** (invertible nonlinear maps can linearly
  scramble; the concentric-rings construction). ⇒ **2a's *linear* exposure is not load-bearing.**
- **2a is a non-adversarial result:** predict-ahead → linear monitorability **under the
  distillation threat** (a model not fighting the probe). It is the honest, solid claim.

## Honest synthesis

Adversarial robustness of cheap (linear) monitoring is **likely weak** — the hiding solution
exists and isn't expensive in principle — but is **hard to demonstrate cleanly**, because
adversarial removal by training is unstable/ineffective (the wall). So: **predict-ahead helps
monitorability against distillation (2a, solid); it should not be assumed robust to an
adversary, and proving the failure cleanly needs a better method than GRL** (a
converged-discriminator min-max, or a non-training existence construction).

## If resumed

- Swap GRL for a **converged-discriminator** min-max (retrain a linear probe to optimality each
  outer step, minimize *its* accuracy) — the methodologically correct adversary.
- Or a **non-training existence construction** (explicitly build the invertible nonlinear map
  that linearly scrambles S_t while preserving the dynamics-MLP predictions).
- Either way: capability gate (task held fixed) + fresh held-out linear *and* MLP evaluators.

Artifacts: `2b_l0.0/0.3/1.0/` (probes), `2b_l0.0_features/` (dumped features),
`price_of_concealment.py`. Background: [[concealworld 2a]] (`2a_full/FINDINGS.md`).
