# NextLat for AI Safety: Research Agenda

## Goal

Evaluate whether Next-Latent Prediction (NextLat) can make model internals more compact, predictable, and monitorable in ways that are useful for AI safety work.

This is not an assumption that NextLat is itself an alignment method. The working hypothesis is narrower: if NextLat pushes models toward more coherent latent state transitions, it may improve our ability to inspect, probe, detect, and constrain model behavior.

## What the Paper Demonstrates vs. What We Hypothesize

The NextLat paper makes **no safety claims**. Everything below is our extrapolation from its capability and representation results. This table marks the evidential boundary so the workstreams are not mistaken for established results. ("Paper" = Teoh et al., NeurIPS 2025, arXiv:2511.05963.)

| Premise this agenda relies on | What the paper actually shows | Status |
| --- | --- | --- |
| NextLat yields more coherent latent transitions | Core mechanism (next-hidden loss with stop-gradient targets + KL term) and Theorem 3.2 | Supported, but Theorem 3.2 holds only under *exact* joint optimization — asymptotic/idealized, no finite-sample or rate guarantees |
| Latents are more compact | Manhattan effective latent rank ~52.7 vs GPT ~160 (lower = more compact) | Supported empirically |
| Hidden states are easier to decode (Workstream 1) | Linear probes predict *future tokens* 1–20 ahead with lower cross-entropy than GPT | Extrapolation — the paper never probes for safety-relevant variables (hazard seen, irreversibility, policy compliance) |
| Forward-rolling the latent dynamics model to gate actions before emission (Workstream 2) | The paper trains `p_ψ` to be unrolled ~8 steps and decoded to matching token distributions — but only on realized trajectories, never as an inference-time safety gate | Untested as a monitor, but the underlying rollout capability is validated by the training objective; counterfactual (off-path) rollout fidelity is the open question |
| Latent state is more robust under distribution shift (Workstream 3) | Manhattan detour robustness ~95% vs GPT ~85% | Weakly supported — a single narrow synthetic setting |
| Gains transfer to agentic / tool-use settings (Workstream 5) | — | No evidence — experiments use ~60–114M-parameter models on synthetic tasks + TinyStories |

Treat Workstreams 2 and 5, and the safety-variable framing of Workstream 1, as the highest-uncertainty, highest-novelty bets: they have no direct support in the paper and are the parts most worth de-risking early.

**Empirical status — Experiment 0 (2026-06-17/18, see [experiments/gridworld/exp0.md](experiments/gridworld/exp0.md)):** first controlled test of the Workstream 1 premise, with a design-space sweep. In an *easy*, fully-observed gridworld every arm saturates — no signal. In a *partially-observable* variant that **requires belief-state tracking** (no start anchor; position must be integrated from local observations), the Workstream 1 premise **holds — but it is generic, not NextLat-specific**: predictive auxiliary objectives (NextLat **and** MTP/JTP) all lift position decodability from a vanilla LLM's 0.71 to ~0.95, and it's genuine information availability (a strong nonlinear probe can't recover it from the GPT baseline), not mere linearization. The durable claim is therefore *"predict-ahead training improves belief-state monitorability,"* of which NextLat is one instance. **NextLat's distinctive axis is compression** — same monitorability at ~6× lower effective rank than MTP (30 vs 181) — so the open question becomes whether *compactness* (not monitorability per se) has independent safety value. The decoupled-deletion worry did **not** appear in this toy (all aux arms retained the bit *better* than GPT). Caveats: ≤~9M-param models, synthetic, not language/agentic. Methodological lesson: the probe target must not be trivially decodable for the baseline, or both arms ceiling and the test cannot discriminate.

## Central Framing: Behaviorally-Expressed vs Behaviorally-Decoupled Risk

The belief state `h_t` is, by Theorem 3.2, a sufficient statistic *for predicting future tokens under the training distribution*. That single fact splits safety-relevant variables into two classes, and the split determines what NextLat can and cannot help with:

- **Behaviorally-expressed risk** — about to call a tool, about to reveal a secret, entering an irreversible state, currently operating under a hazardous instruction. These change the distribution of near-future tokens, so sufficiency requires `h_t` to **preserve whatever distinctions separate them** — otherwise the next-token distribution could not differ. That is weaker than the variable being cleanly or linearly recoverable; it only guarantees the information is not discarded, which is the precondition any probe needs. So this is a stronger basis for Workstream 1 than "compact latents are easier to probe," but recoverability itself is an empirical question (Workstream 1), not a corollary of the theorem.
- **Behaviorally-decoupled risk** — dormant deceptive intent, sandbagging, a withheld plan that only surfaces much later. These are by construction *not* predictive of near-future tokens. A *minimal* sufficient statistic has active pressure to **compress them away**, so NextLat may make exactly the hardest-to-catch risks *harder* to monitor, not easier.

Operating thesis: **NextLat is a monitor for behaviorally-expressed risk, not a deception detector.** Claim the first class; explicitly disclaim the second. Most of the value — and the cleanest experiments — live in the expressed class, and the decoupled class is where this approach most plausibly fails (see Risks).

## Core Questions

1. Are behaviorally-expressed safety variables easier to decode from NextLat hidden states than from a matched baseline — and does compression *degrade* decoding of behaviorally-decoupled variables?
2. Can the latent transition model be rolled forward over candidate actions to predict an unsafe state *before* it is emitted (predictive gating), and does that beat reactive, after-the-fact scoring?
3. Are latent transitions more stable, and probe outputs better calibrated, under distribution shift than standard next-token training?
4. Does better latent compression trade off against capabilities, or can both improve together?
5. Do the internal gains persist in agentic settings with tools, memory, and multi-step plans?

## Proposed Workstreams

### 1. Probeability of Safety-Relevant State

Train matched baseline models with GPT-style next-token prediction and NextLat. Freeze both and train simple probes on hidden states to predict:

- whether the model has seen a hazardous instruction earlier in context
- whether a hidden variable has changed
- whether the current step is reversible or irreversible
- whether the current action is policy-compliant
- whether the model is about to invoke a tool, reveal sensitive data, or cross a sandbox boundary

Always include at least one **behaviorally-decoupled control variable** (a hidden fact that does not affect near-future tokens) alongside the expressed variables above. The central framing predicts NextLat helps on expressed variables and ties-or-loses on the decoupled control; an experiment that only tests expressed variables cannot falsify that.

What the theory does and does not buy us:

- Sufficiency (Theorem 3.2) implies `h_t` must *preserve the distinctions* an expressed variable induces in the future-token distribution — not that the variable is *linearly*, or even cleanly, decodable. The component that plausibly makes it linearly decodable is the KL term, which aligns latent geometry with predictive-distribution geometry. So report both linear and small-MLP probes, and treat the linear-vs-nonlinear gap as a measured quantity, not an assumption.

Success metric:

- higher probe accuracy on expressed safety variables at equal parameter count and training budget, with the linear-probe gain called out separately
- earlier detection of risky trajectories from intermediate layers

The decoupled control variables are a **diagnostic, not a pass/fail bar**: tie-or-lose against the baseline is the predicted outcome (compression should not help, and may hurt — see Risks) and is acceptable. A NextLat win there would be a welcome surprise that falsifies the compression-deletes-decoupled worry; it is not required for success.

Why it matters:

If behaviorally-expressed safety facts become easier to decode from hidden states, monitoring systems can be simpler and less brittle — and mapping where the gain disappears (the decoupled class) tells us what *not* to rely on NextLat for.

### 2. Predictive Belief-State Monitoring (Forward Rollout)

This is the strongest distinctive use of NextLat. The latent transition model `p_ψ` is cheap (a 3-layer MLP) and is *already trained to be unrolled several steps and decoded back to matching token distributions*. So instead of scoring anomalies after the fact, roll the state forward over candidate actions and gate *before* emission:

```
At decision point t with belief state h_t:
  for each candidate next action / token a_i:        # e.g. {call_tool, refuse}, or top-k tokens
      h_hat_i = f_psi(h_t, a_i) + h_t                # one MLP pass — negligible vs a transformer step
      risk_i  = g(h_hat_i)                           # pretrained probe from WS1: P(irreversible), ...
  gate if the risk of the chosen / likely action exceeds tau   # block, force second pass, or escalate
```

Why predictive rollout rather than the reactive residual `||h_{t+1} - h_hat_{t+1}||`:

- **Lead time is structural**, not hoped-for — the unsafe state is scored before the action is taken, removing the one-step lag inherent to residual scoring.
- It scores *predicted safety state* (the actual question), whereas raw residual magnitude mostly measures **novelty / self-consistency** — a jailbroken model is internally self-consistent, so the residual alone is expected to be a high-false-positive novelty detector (see Risks).
- It reuses a capability the paper already validates (short-horizon latent rollout + decode), rather than betting on an emergent one.

Keep the reactive residual, but reframed as a second, complementary signal:

- **Two-model disagreement as an off-manifold tripwire.** `p_ψ` is a compressed prior over *legitimate* transitions. When tampered context (prompt injection, manipulated tool output) pushes the transformer's realized `h_{t+1}` away from `ĥ_{t+1}`, the disagreement flags an off-manifold input — a sharper framing than thresholding residual magnitude for "anomaly."

Experiments:

- clean vs jailbreak prompts; benign vs exfiltration-style tasks; normal vs manipulated tool outputs
- predictive gate (rollout + probe) vs reactive residual vs two-model disagreement, on the same trajectories

Metrics:

- AUROC separating benign and unsafe trajectories
- stepwise lead time before an unsafe output is emitted, predictive vs reactive, **at equal false-positive rate**
- rollout fidelity vs horizon (how many steps `p_ψ` stays accurate off the realized path) — predictive gating is only as trustworthy as the rollout
- robustness under paraphrase and formatting changes

Why it matters:

A useful safety signal does not need to solve alignment. It only needs to fire reliably and early enough to support escalation, gating, or human review — and acting *before* emission is worth more than detecting *after*.

### 3. Distribution Shift and OOD State Tracking

Test whether NextLat hidden states degrade more gracefully under shift.

Shift settings:

- longer contexts than seen in training
- adversarially reordered evidence
- misleading intermediate observations
- tool failures, missing observations, or stale memory
- domain shifts from training tasks to deployment-style tasks

Metrics:

- calibration of probe outputs under shift
- hidden-state separability for critical latent variables
- downstream task regret after corrupted context
- recovery speed after a misleading observation

Why it matters:

Many safety failures are not pure capability failures. They are state-tracking failures in unusual conditions.

### 4. Counterfactual Intervention Studies

Build controlled environments where the correct internal state is known and can be perturbed.

Candidate environments:

- gridworld navigation with hidden doors or keys
- simulated cyber-defense environments
- tool-use tasks with partial observability
- long-horizon QA with delayed evidence

Interventions:

- inject contradictory observations
- hide a critical fact and reveal it later
- modify a tool result mid-trajectory
- create near-identical prompts that should induce different beliefs

Metrics:

- whether latent states change in the correct direction
- whether probes recover the new hidden state quickly
- whether the model maintains policy compliance after the intervention

Why it matters:

Interpretability claims are stronger when the correct latent update is known in advance.

### 5. Agentic Safety Benchmarks

Extend beyond sequence modeling into simple agents that plan and act.

Target tasks:

- coding agents with file, shell, or retrieval tools
- web agents with navigation and form submission
- sandboxed ops tasks with allowed and disallowed actions

Compare:

- standard objective only
- standard objective plus NextLat
- NextLat plus runtime monitors derived from predictive rollout, transition error, and probes

Metrics:

- unsafe action rate
- policy violation rate
- false positive rate on benign runs
- task success under monitoring and gating

Why it matters:

Even if latent structure improves in offline benchmarks, the safety value is only real if it transfers to action-taking systems.

### 6. Belief-State Forensics and Attribution

Because the belief state is fixed-size with an *explicit* update rule, a trajectory's states can be snapshotted and diffed, and large belief-state jumps localized in time — "when did the agent's belief shift toward the risky action." Whether a jump can be **attributed to a specific causing token or tool result** is an open experimental question, not something the representation guarantees: diffing shows *when* the state moved, but isolating unique causal responsibility to one input requires intervention (e.g. ablating or substituting candidate inputs and checking whether the jump survives). Standard transformers offer no comparably clean, diffable state to even start from.

Experiments:

- on trajectories that end in an unsafe action, locate the belief-state update that commits to it and trace it to a specific input token or tool result
- compare attribution sharpness (how localized the causal update is) for NextLat vs baseline

Why it matters:

Post-hoc incident analysis and red-team triage need to answer "what changed the model's mind, and when." A compact state with an explicit transition rule turns that into a measurable diff rather than an attention-pattern guess.

### Higher-Risk Extension (parked): Training-Time State Shaping

Beyond monitoring, one could add a safety term to the latent objective to carve out an explicitly monitored "safe-state" coordinate. Parked deliberately: it is capability-coupled (better world models aid unsafe capability too) and invites Goodharting the very probe used to monitor. Pursue only after the monitoring workstreams show signal.

## Experimental Design Principles

- Keep parameter count, data, optimizer, and compute matched across conditions.
- Use the same base architecture where possible so the objective is the main difference.
- Track both safety metrics and capability metrics to surface real tradeoffs.
- Evaluate across seeds; latent-state claims are often noisy.
- Prefer simple probes first. If linear probes already work better, that is a strong result.
- Always pair every expressed-risk probe with a decoupled-risk control, so a positive result has a boundary.

## Immediate Experiment Set

### Experiment 0 (minimal killer test): Expressed vs decoupled probing

In a gridworld / Manhattan-style environment with known hidden state, plant two labeled variables: one **behaviorally-expressed** (e.g., about to step onto an irreversible tile) and one **behaviorally-decoupled** (e.g., a hidden goal that only matters many steps later). Train matched GPT and NextLat models; train probes for both variables on frozen states.

Prediction: NextLat beats the baseline on the expressed variable and ties-or-loses on the decoupled one. Either outcome is informative — confirming it maps the boundary of the approach; a NextLat win on *both* falsifies the compression-deletes-decoupled worry.

Deliverable:

- a 2×2 table — (GPT vs NextLat) × (expressed vs decoupled) — of probe accuracy, by layer

### Experiment 1: Safety-variable probing

Extend Experiment 0's setup to the fuller variable list in Workstream 1 (hazardous-instruction-seen, reversibility, policy-compliance, about-to-act). Train probes on frozen states; the NextLat training targets are unchanged — only the *probe* labels are safety-relevant.

Deliverable:

- a table comparing GPT vs NextLat probe accuracy by layer and horizon, linear and small-MLP probes reported separately

### Experiment 2: Predictive gate vs reactive residual

On benign and adversarial trajectories, compare (a) forward-rollout gating — unroll `p_ψ` over candidate actions and score with the Experiment 1 probe — against (b) reactive residual scoring and (c) two-model disagreement.

Deliverable:

- ROC curves and per-step detection plots; lead-time at equal false-positive rate, predictive vs reactive; rollout fidelity vs horizon

### Experiment 3: Evaluate under hidden-state perturbations

Run controlled interventions where the true belief state should change abruptly (Workstream 4).

Deliverable:

- recovery-time plots and error breakdowns; probe calibration under shift

### Experiment 4: Add runtime gating

If predictive risk (or, as a fallback, transition error) exceeds a threshold, block tool execution or require a second pass.

Deliverable:

- safety/capability tradeoff curve for threshold settings

## Risks and Failure Modes

- **Compression can delete the variables we most want to monitor.** A minimal sufficient statistic discards information not needed to predict near-future tokens, so behaviorally-decoupled risks (dormant deception, sandbagging) may be compressed *out* of the belief state — the opposite of what monitoring needs. This is the central limitation, not a footnote.
- More compact latents are not automatically more truthful or more aligned.
- The reactive transition residual largely measures novelty / self-consistency, not danger, and is expected to over-fire; predictive rollout addresses *what* is scored but inherits its own failure mode below.
- Forward rollouts are only trustworthy near the training manifold — exactly the novel, unsafe actions we care about may be where `p_ψ` is least reliable. (Its disagreement with the transformer is then a fallback signal, but rollout-based risk scores there should be down-weighted.)
- Safety gains in synthetic tasks may not transfer to realistic language-agent settings.
- Better world models can improve both safe and unsafe capabilities.

## Decision Criteria

This line of work is promising if at least one of the following holds:

- NextLat substantially improves probe access to behaviorally-expressed safety state (with the linear-probe gain called out), without degrading decoupled-variable access more than the baseline.
- predictive forward-rollout gating buys useful lead time at an acceptable false-positive rate, beating reactive residual scoring.
- OOD state tracking is measurably more robust than standard training.
- runtime monitors built on top of NextLat reduce unsafe actions without unacceptable capability loss.

If none hold, NextLat is probably more interesting as a capability or efficiency method than as a safety primitive. Note the scope deliberately excludes behaviorally-decoupled deception: that is not a target of this agenda, and a negative result there is expected, not disqualifying.
