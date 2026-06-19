# Feasibility Study — Predictive Objectives for Monitorable Belief State

*As of 2026-06-19. This document records the current judgment on whether the research direction in
[research-overview.md](research-overview.md) is worth pursuing as an AI safety program, what is
already supported, what remains speculative, and what would change the verdict.*

## Executive summary

**Bottom line:** this is a **feasible and worthwhile safety research direction**, with a credible
path to publishable results and a nontrivial chance of yielding a useful monitoring methodology.
It is **not yet** a validated practical safety mechanism, and it should not be framed as an
alignment solution.

The strongest current claim is:

> **Predictive-representation training objectives can make belief-state variables substantially more
> monitorable from hidden states, especially in tasks that actually require history-dependent state
> tracking.**

The weaker, still-open claims are:

- whether this transfers from toy POMDP-style tasks to language and tool-using agents
- whether compactness has independent monitor value beyond raw monitorability
- whether explicit latent rollouts can support reliable pre-action safety gating

## The research question

The program asks:

> **Do predictive-representation objectives make internal belief state more monitorable, and how do
> monitorability, compactness, robustness, and capability trade off across the space of such
> objectives?**

This is narrower than "can we align models by changing the objective." It is a monitoring question:
can we train models so safety-relevant internal state becomes easier to recover and track?

## Why this direction is plausible

Three separate ideas already have support in the literature and fit together coherently:

1. **Transformers can represent belief-state-like structure internally.**
   Prior work on belief-state geometry and latent world-state probing supports the premise that
   hidden states can encode state variables recoverable by probes.

2. **Predictive / world-model objectives can shape internal representations.**
   NextLat, MTP, and JTP were all proposed as representation-shaping or future-predictive training
   objectives, not just as output-level capability tweaks.

3. **Monitoring can exploit internal activations.**
   Recent safety-adjacent work shows that hidden-state probes can detect or diagnose latent state,
   hallucination, and other properties more directly than output-only methods.

The research direction here is the synthesis:

> **Use training-objective choice as the safety lever to engineer monitorable belief state, then
> compare objectives by safety-relevant representation properties.**

That synthesis is more novel than any one ingredient.

## What is already supported by our results

From [experiments/gridworld/exp0.md](experiments/gridworld/exp0.md), the current evidence supports the following:

- In **fully observed / easy** settings, there is no useful signal because every model saturates.
- In **partial-observability / belief-state** settings, predictive-objective models make hidden
  state much more linearly decodable than a vanilla GPT baseline.
- That gain appears **generic to predict-ahead training** rather than specific to NextLat alone.
- The methods differ sharply on **compactness**, with NextLat occupying the compact-and-monitorable
  corner.
- The gain survives a moderate scale-up in model size inside the same synthetic regime.

This is enough to establish a credible **existence proof** for the core monitorability idea.

## What remains unproven

The current evidence does **not** yet establish:

- transfer to natural-language belief-state tasks
- transfer to agentic/tool-use settings
- usefulness against deliberate deception or behaviorally-decoupled hidden intent
- practical online safety gains from runtime monitors
- that NextLat specifically is the best answer, rather than one member of a broader objective class

This means the direction is promising, but the present evidence is still best read as
**research-feasible, deployment-unready**.

## Feasibility by layer

### 1. Scientific feasibility: **high**

This is the strongest part of the case.

- The hypotheses are sharp and falsifiable.
- The task design is controllable enough to isolate representation effects.
- The early results are already informative: they ruled out trivial setups, found the
  partial-observability regime where the effect appears, and showed the effect is class-level.

Even if later steps fail, the program is still likely to produce useful science about how training
objectives shape internal state.

### 2. Methodological novelty: **medium-high**

This is not the first work on:

- probing hidden states
- monitoring latent world state
- belief-state representations
- representation-shaping objectives

What is novel is the methodological framing:

- treat **monitorable belief state** as the target property
- treat **training objective** as the control knob
- compare objective families on a **monitorability / compactness / robustness frontier**

That is a defensible novelty claim. It is best framed as a **novel safety-oriented research
program**, not as a wholly unprecedented idea.

### 3. Engineering feasibility: **medium-high**

The implementation path is practical.

- The core experiments are cheap enough to iterate on.
- The current harness already supports matched arms, probing, and comparative analysis.
- The next step does not require frontier-scale models; it requires better-controlled tasks and
  evaluation axes.

The main engineering risk is not cost. It is whether more realistic tasks can preserve the same
clean label discipline that made Exp 0 interpretable.

### 4. Safety-product feasibility: **medium**

This is where caution is warranted.

The approach is most plausible for **behaviorally-expressed risk**:

- state that affects imminent action selection
- state that depends on accumulated history
- state that must be tracked across partial observability or tool interaction

It is much less plausible as a detector for:

- dormant deception
- hidden long-horizon plans that do not affect near-future predictions
- intentionally obfuscated internal goals

So this is probably not a general alignment method. It is more plausibly a foundation for
**monitoring, auditing, and action-gating** in the subset of settings where safety-relevant state is
belief-state-like and behaviorally expressed.

## The key strategic update from Experiment 0

The most important update is that the original "why NextLat?" argument changed.

Earlier version:

- NextLat might help because it compresses history into a compact sufficient statistic.

Current version:

- **Predict-ahead objectives in general** appear to improve belief-state monitorability.
- **NextLat specifically** may matter because it reaches similar monitorability at far lower
  effective rank.

That shift is healthy. It narrows the decisive question:

> **Does compactness buy a monitor anything real?**

If the answer is no, the project still yields a good class-level result about predictive
objectives, but NextLat becomes one example rather than the destination.

## Main risks

### 1. The effect may be toy-specific

The cleanest failure mode is that the monitorability gain is real in synthetic POMDPs but does not
survive language tasks or tool-use trajectories.

### 2. Compactness may be scientifically interesting but safety-irrelevant

If NextLat is only "more compact" without making monitors more robust, cheaper, or easier to
transfer, then it is not yet a safety winner.

### 3. The program may diagnose without enabling intervention

Many monitoring ideas can classify unsafe trajectories better than baselines but still fail to help
with real-time control or correction. The same risk applies here.

### 4. Deception remains mostly untouched

The strongest current results concern belief-state tracking, not hidden strategic misalignment.
That is a scope boundary, not a flaw, but it should be explicit.

## Best next experiments

The current ordering in [research-overview.md](research-overview.md) is correct.

### Step 2: test whether compactness has monitor value

This is the highest-value next experiment because it decides whether NextLat deserves to remain the
lead method.

The right questions are:

- Are NextLat probes more robust under distribution shift?
- Can the same signal be recovered from fewer principal components?
- Do probes trained on one setting transfer better?
- Is sample efficiency better when the belief state is more compact?

If these all come back flat, the generic class-level result stands but the NextLat-specific case
weakens substantially.

### Step 3: predictive forward-rollout monitor

This is the highest-upside mechanism, but it should stay gated behind Step 2.

Why:

- It is the most distinctive safety artifact.
- It directly targets **pre-action** intervention rather than post-hoc diagnosis.
- It also depends on the riskiest assumption: that off-path latent rollouts are reliable enough to
  score.

### Language bridge: long-context fact updates / variable binding

This is the cleanest next realism jump because it preserves exact ground truth while moving beyond
gridworld.

If the effect fails there, confidence in the broader program should drop materially.

## Go / no-go criteria

This program remains strongly justified if one or more of the following hold:

- the monitorability gain replicates on language belief-state tasks
- compactness yields robustness, transfer, or low-dimensional-monitor benefits
- predictive rollouts provide actionable lead time before unsafe actions

Confidence should drop sharply if:

- the effect disappears outside synthetic navigation-style tasks
- the compactness axis shows no practical monitor benefit
- predictive rollouts are too unstable off-manifold to support gating

## Overall judgment

**Recommendation: continue.**

Reason:

- As a **research program**, this is clearly feasible and already productive.
- As a **novel safety methodology**, it is plausible and differentiated enough to justify further
  investment.
- As a **practical safety mechanism**, it is still early and should be framed carefully.

The right description today is:

> **Representation engineering for monitorable belief state** with a credible safety motivation and
> early positive evidence.

The wrong description today would be:

> "We have a general method for detecting deception or aligning models."

## Related-work anchors

The current framing is most closely adjacent to the following lines of work:

- belief-state geometry in transformer residual streams
- latent world-state probing / propositional probes
- predictive-objective training such as NextLat, MTP, and JTP
- hidden-state-based runtime monitors and monitorability evaluations

Those lines make the program legible and credible. The novelty lives in how they are being combined
and operationalized for safety.
