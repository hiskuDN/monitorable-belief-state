# NextLat AI Safety Notes

This folder captures a concrete research agenda for testing whether Next-Latent Prediction (NextLat) can support AI safety work through better latent-state structure, monitoring, and anomaly detection.

- **Start here:** [docs/research-overview.md](docs/research-overview.md) — current-state synthesis around the general question: *do predictive-representation training objectives make a model's belief state more monitorable, and how do monitorability, compactness, and capability trade off across them?* Covers what Experiment 0 established and the prioritized program. NextLat is one studied instance, not the thesis.
- **Detailed plan (NextLat-specific):** [docs/nextlat-ai-safety-research-agenda.md](docs/nextlat-ai-safety-research-agenda.md) — the original agenda (workstreams, expressed/decoupled framing, decision criteria, risks).

## Experiments

- **[Experiment 0](docs/experiments/gridworld/exp0.md)** — a controlled probing test of Workstream 1 (does NextLat make safety-relevant belief state more linearly decodable?). In a **partially-observable** gridworld that requires belief-state tracking, NextLat localizes position ~28 points better than a matched GPT (0.95 vs 0.70) at ~2.6× lower effective rank — more compact *and* more probeable. In fully-observed (easy) regimes both models saturate, so the comparison only bites once real state-tracking is required. Code: [`experiments/`](experiments/) + the gridworld additions to a cloned [`NextLat/`](NextLat/EXP0_CHANGES.md); results: [`docs/results/`](docs/results/). Runs on Modal GPUs.

## References

- Paper: Teoh, Tomar, Ahn, Hu, Pearce, Sharma, Krishnamurthy, Islam, Lamb, and Langford. _Next-Latent Prediction Transformers Learn Compact World Models_. NeurIPS 2025. https://arxiv.org/abs/2511.05963
- Code: `JaydenTeoh/NextLat` — https://github.com/JaydenTeoh/NextLat
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
