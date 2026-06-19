# Agentic Application — Monitorable Belief State for Tool-Calling Agents

*As of 2026-06-19. This document translates the research program in
[research-overview.md](research-overview.md) into the tool-calling / agentic setting: why the idea
fits, what safety-relevant latent state to target, and what the first concrete experiments should
be.*

## Core idea

Tool-calling and agentic AI are a natural destination for this work because they are effectively
**POMDP-like**:

- the agent never sees the full world state directly
- it only receives partial observations (prompt context, tool outputs, file listings, API results)
- it acts over time (call tool, write file, execute command, ask follow-up, stop)
- safe behavior depends on maintaining an accurate **belief state** over hidden or evolving facts

The research question becomes:

> **Can predictive-representation objectives make an agent's internal belief state more monitorable,
> so we can detect risky or mistaken state before the agent takes a side-effectful action?**

This is the most safety-relevant extension of the current program.

## Why the fit is strong

The current gridworld result says predictive objectives help when the model must track
**history-dependent state** rather than read the answer directly from the current input. Tool-using
agents live in that regime by default.

The agent usually has to remember or infer:

- what files currently exist
- what has already been done
- which tool outputs are fresh vs stale
- whether a permission has been granted
- whether a credential is currently in scope
- whether the current plan step is reversible
- whether a prior instruction is still binding

Those are not "facts in the current token." They are belief-state variables accumulated over a
trajectory.

## What safety problem this could help with

This approach is most plausible for **behaviorally-expressed operational risk**, such as:

- about to call a restricted tool without authorization
- about to write to the wrong file or environment
- about to leak or expose data after an earlier tool result surfaced it
- about to take an irreversible action without the required state being true
- about to proceed based on stale, contradictory, or incomplete world state

The key point is that these are risks that depend on **current internal state conditioned on
history**, not just on the latest observation.

## What it probably does *not* solve

This is not a general deception detector.

It is much less plausible for:

- dormant deceptive intent
- hidden long-horizon plans that do not affect near-future action selection
- strategic misalignment that is deliberately kept behaviorally decoupled

So the right framing is:

> **A monitoring and action-gating methodology for operational belief state**, not a complete
> alignment method.

## The belief-state variables to target

The first agentic experiments should probe variables with three properties:

1. they matter for safety
2. they require history-dependent tracking
3. they have clean ground-truth labels

Good target variables:

- **Permission state**
  - Has the user authorized file deletion?
  - Is network access currently allowed?
  - Has the user approved sending external data?

- **Boundary state**
  - Is the agent inside or outside a safe sandbox?
  - Has it crossed into a directory/environment it should not modify?

- **World-state consistency**
  - Does the agent correctly track which files exist, which were renamed, and which are stale?
  - Does it know whether a prior tool result was superseded?

- **Execution-readiness state**
  - Does it have enough evidence to execute a side-effectful action?
  - Is the current action reversible or irreversible?

- **Instruction-binding state**
  - Is an earlier hazardous instruction still active?
  - Has a later instruction overridden an earlier one?

These are more safety-relevant than generic "world modeling" labels because they connect directly
to whether the next action should be allowed.

## Minimal experimental path

Do not start with a full autonomous coding agent. Start with a **small tool-use simulator** that
preserves the clean-label discipline from Experiment 0.

### Stage 1: synthetic tool-use environment

Construct a compact environment with:

- a file system state
- a small permission model
- a few tools (`ls`, `read`, `write`, `delete`, `search`, `ask_user`)
- hidden state changes over time
- a task requiring the agent to sequence tool calls correctly

Examples:

- a file is renamed, then later referenced under the old name
- permission to delete is granted, then revoked
- a secret appears in one tool output but should not be written or exfiltrated
- a tool output becomes stale after a later mutation

At each step, label:

- current file-state facts
- current permission state
- whether the next action would be policy-compliant
- whether the next action would be irreversible
- whether the agent is acting on stale state

This yields exact probe targets while still being recognizably agentic.

### Stage 2: probeability comparison

Train matched baseline and predictive-objective models on the same tool-trace data. Then probe:

- current latent world state
- permission/boundary state
- action-risk labels

Primary question:

> **Are the safety-relevant agent state variables more linearly recoverable under predictive
> objectives than under a vanilla baseline?**

If not, the agentic version of the thesis weakens substantially.

### Stage 3: pre-action monitor

Once probes work, use them online:

- before each side-effectful tool call, score the hidden state for risk
- if risk is high, block the action, force clarification, or require a second-pass check

This is the first truly agentic safety artifact.

## The strongest version of the application

The highest-value version is:

> **Use hidden-state monitors to gate side-effectful tool actions based on the agent's inferred
> belief state before the action is taken.**

That is stronger than post-hoc evaluation because it can intervene *before* the mistake lands.

Examples:

- block `delete_file` if the hidden state says permission is missing
- block `send_email` if the hidden state says the content includes protected data
- block `run_command` if the hidden state says the working directory or target resource is wrong
- force re-checking if the hidden state says the current plan relies on stale tool evidence

## Where predictive objectives could matter specifically

The current research program suggests two possible advantages in agents:

### 1. Better monitorability of belief state

Predictive objectives may make:

- permission state
- file/world state
- action readiness
- boundary status

more linearly decodable and easier to monitor.

### 2. More compact state for cheaper / more robust monitors

If NextLat's compactness advantage carries over, then monitors may become:

- lower-dimensional
- cheaper to train
- more robust under shift
- easier to transfer across task variants

This is where the agentic setting could reveal whether compactness has real safety value.

## Main failure modes

### 1. The toy-to-agent gap

The effect may hold in synthetic tracking tasks but disappear once tool traces become messy and
open-ended.

### 2. Labeling becomes much harder

The more realistic the environment, the harder it becomes to produce exact ground-truth labels for
"correct current belief state."

### 3. The monitor may only detect state-tracking failures, not strategic bad behavior

That still has value, but it narrows the claim.

### 4. Probe success may not translate into safe intervention

The hidden state might be readable without yielding a reliable gating policy.

### 5. Off-manifold rollout risk

If later work uses explicit latent rollouts to predict future agent states, those rollouts may be
least trustworthy exactly where risky actions are novel.

## Best first experiment

The first serious agentic experiment should be:

> **A synthetic tool-use environment with exact labels for permission state, file state, and
> policy-compliance of the next action.**

Why this one:

- it is close to real agentic safety concerns
- it preserves exact ground truth
- it tests operational belief state directly
- it gives a clean bridge from gridworld to tool use

This is a better next step than jumping immediately to full coding agents or browser agents, where
ground truth and diagnosis get much noisier.

## What success would look like

This line would look promising if:

- predictive-objective models expose agent state variables better than a baseline
- the gain survives mild tool noise and distribution shift
- a probe-based gate reduces unsafe tool actions without destroying task success
- compactness yields lower-dimensional or more robust monitors

## What would falsify the application

Confidence should drop if:

- the gain disappears once tool use is introduced
- safety-relevant agent state is not more decodable than in the baseline
- monitors can read state but cannot improve action safety in practice
- compactness provides no robustness or operational benefit

## Bottom line

This research direction **does apply naturally to tool-calling and agentic AI**.

It is probably **more naturally applicable there than in generic text-only settings**, because the
state variables we most care about in safety are often exactly the ones agents must carry across
tool trajectories.

The most credible target is:

> **monitorable operational belief state for pre-action tool gating**

That is a strong, concrete, safety-relevant destination for the program.
