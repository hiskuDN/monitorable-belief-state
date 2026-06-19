# Monitorable Belief State

Research on whether **predictive-representation training objectives** (next-latent prediction, multi-token prediction, and relatives) make a transformer's internal **belief state** more *monitorable*: the kind of internal state a safety monitor would want to read. The thesis is general; NextLat (Teoh et al., NeurIPS 2025) is the seed and one studied instance, not the bet.

**[Try the interactive demo](https://probe.hiskiasdingeto.com/)**: a linear probe reading position from a frozen GPT vs predict-ahead models, side by side.

## The question

> Do predictive-representation objectives make a model's belief state more monitorable, and how do monitorability, compactness, and capability trade off across them?

A lot of safety-relevant internal state is *belief state*: a variable the model has to track across a trajectory (is this action irreversible given history? did the user authorize it earlier? is the instruction from the user or from an injected tool output?), not something readable from the current token. If those become more linearly probeable through how you train, monitors get simpler and less brittle.

## What we have found (small synthetic models, a gridworld)

- **Predict-ahead training makes belief state far more linearly monitorable** than a vanilla LLM, but only in tasks that actually require belief-state tracking (exact-position decodability 0.71 to ~0.95). It is genuine information, not just a tidier linear arrangement.
- **It is generic across the objective class** (NextLat, MTP, and JTP all match), not NextLat-specific.
- **Scale-robust:** the gap holds, and in fact widens, at 4x the backbone.
- **Compactness is safety-neutral:** NextLat's distinctive low-rank representation does not buy a cheaper, more sample-efficient, or more robust monitor.

So the durable claim is class-level: *predict-ahead training improves belief-state monitorability.* Caveat: all evidence so far is on ~6-33M-parameter models in a synthetic gridworld, a controlled existence proof, not a language or agentic claim.

## Start here

- **[docs/research-overview.md](docs/research-overview.md)**: current-state synthesis (the question, what the experiments established, the prioritized program). Read this first.
- **[docs/agentic-application.md](docs/agentic-application.md)**: translating the program to tool-calling agents, the most safety-relevant destination.
- **[docs/safety-task-menu.md](docs/safety-task-menu.md)**: a ranked menu of next experiments (synthetic, clean-label safety tasks).
- **[docs/nextlat-ai-safety-research-agenda.md](docs/nextlat-ai-safety-research-agenda.md)**: the original NextLat-specific agenda (the seed framing).

## Experiments

- **[Experiment 0](docs/experiments/gridworld/exp0.md)**: the controlled probing test in a partially-observable gridworld (monitorability, generality, and the scale check).
- **[Step 2](docs/experiments/gridworld/step2-compactness.md)**: does compactness buy a monitor anything (it does not).

Code lives in [`experiments/gridworld/`](experiments/gridworld/) with shared Modal scaffolding in [`experiments/common/`](experiments/common/); the additions to a cloned upstream model are in [`NextLat/`](NextLat/EXP0_CHANGES.md); results in [`docs/results/`](docs/results/). Training and probing run on Modal GPUs.

## Demo

Live at **[probe.hiskiasdingeto.com](https://probe.hiskiasdingeto.com/)**. The source ([`web/`](web/)) is a fully static, precomputed app (Vite + React): a linear probe decoding a model's belief about its position step by step, GPT vs predict-ahead, plus a by-layer decodability chart. No model runs in the browser.

```
cd web && pnpm install && pnpm dev
```

## References

- Paper: Teoh, Tomar, Ahn, Hu, Pearce, Sharma, Krishnamurthy, Islam, Lamb, and Langford. *Next-Latent Prediction Transformers Learn Compact World Models*. NeurIPS 2025. https://arxiv.org/abs/2511.05963
- Code: `JaydenTeoh/NextLat`, https://github.com/JaydenTeoh/NextLat
- OpenReview: https://openreview.net/forum?id=Lh4ayjJIAW

```bibtex
@inproceedings{teoh2025nextlat,
  title     = {Next-Latent Prediction Transformers Learn Compact World Models},
  author    = {Teoh, Jayden and Tomar, Manan and Ahn, Kwangjun and Hu, Edward S. and
               Pearce, Tim and Sharma, Pratyusha and Krishnamurthy, Akshay and
               Islam, Riashat and Lamb, Alex and Langford, John},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2511.05963}
}
```
