# concealworld — the deception rung (incentive to conceal), stage 2a

Experiment glue for **concealworld**: a model maintains a **running secret** (sum of interleaved
updates mod K) it must use to act but masks in its behavior; we probe whether that secret stays
*linearly* decodable from hidden state mid-masking, and whether predict-ahead keeps it decodable
better than a vanilla GPT. The first experiment in the program with an incentive to conceal. Stage 2b
(discovered/adversarial incentive via RL) is gated on 2a and not built here.

What lives here: the probe / analyze scripts and the Modal entrypoints for this family
(`modal_app.py`), reusing the shared scaffolding in [`../common/modal_base.py`](../common/modal_base.py)
and the Exp-0 probe machinery.

What does **not** live here: the environment generator + datamodule. Like gridworld, it must be
registered in NextLat's `train.py`, so it goes in `NextLat/data/concealworld.py` with configs under
`NextLat/config/concealworld/`.

Scope + writeups: [`../../docs/experiments/concealworld/`](../../docs/experiments/concealworld/).
