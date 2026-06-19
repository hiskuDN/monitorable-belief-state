# Docs

Research on whether **predictive-representation training objectives** (next-latent / multi-token
prediction) make a model's internal **belief state** more *monitorable* for AI safety — and how
monitorability, compactness, and capability trade off across them. NextLat is the seed and one
studied instance, not the thesis.

**New here? Read [`research-overview.md`](research-overview.md) first** — it's the current-state
synthesis (the question, what the experiments established, and the prioritized program).

## Strategy & framing

| Doc | What it is |
| --- | --- |
| [`research-overview.md`](research-overview.md) | **Start here.** The general question, what's established, the reframed (class-level) thesis, and the research program. |
| [`feasibility-study.md`](feasibility-study.md) | Judgment on novelty, feasibility, risks, and the go/no-go logic for continuing. |
| [`nextlat-ai-safety-research-agenda.md`](nextlat-ai-safety-research-agenda.md) | The original, detailed NextLat-specific agenda: workstreams, the expressed-vs-decoupled framing, decision criteria, the paper-vs-hypothesis evidential boundary. |
| [`agentic-application.md`](agentic-application.md) | Translating the program to tool-calling / agentic settings (the most safety-relevant destination). |
| [`safety-task-menu.md`](safety-task-menu.md) | Ranked menu of candidate next experiments (synthetic, clean-label safety tasks) + the chosen first two. |

## Experiments

| Doc | Result |
| --- | --- |
| [`experiments/gridworld/exp0.md`](experiments/gridworld/exp0.md) | **Exp 0** — predictive objectives make belief state far more linearly monitorable (position 0.71→0.95) in a belief-state-tracking task; it's **generic** across objectives, **scale-robust** (v3, 4× backbone), and not linearization. NextLat's only distinctive axis is compression. |
| [`experiments/gridworld/step2-compactness.md`](experiments/gridworld/step2-compactness.md) | **Step 2** — that compactness is **safety-neutral**: NextLat's low effective rank buys no cheaper / more sample-efficient / more robust monitor. The durable claim is class-level. |

## Results data

- [`results/`](results/) — per-run probe JSONs, aggregated `summary.md`, and plots. See
  [`results/README.md`](results/README.md) for the run index.

## Where the program stands

Exp 0 + Step 2 are done (in the toy). The open questions: **(a) external validity** — does the gap
survive off the synthetic gridworld (the [`safety-task-menu.md`](safety-task-menu.md) tasks, starting
with a Permission "toolworld"); **(b)** the forward-rollout monitor (the only remaining
NextLat-distinct mechanism). Code for the experiments lives in [`../experiments/`](../experiments/);
the cloned upstream model + training harness in [`../NextLat/`](../NextLat/EXP0_CHANGES.md).
