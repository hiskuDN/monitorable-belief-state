# concealworld — the deception rung (incentive to conceal), stage 2a

Experiment glue for **concealworld**: clone a teacher that masks a secret it needs to act on, then
probe whether the secret stays linearly recoverable from hidden state during the masking window, and
whether predict-ahead keeps it recoverable longer. The first experiment in the program with an
incentive to conceal, built as a minimal one-variable extension of the Exp-0 decoupled-bit harness
(incentive added; surface unchanged). Stage 2b (discovered/adversarial incentive via RL) is gated on
2a and not built here.

What lives here: the probe / analyze scripts and the Modal entrypoints for this family
(`modal_app.py`), reusing the shared scaffolding in [`../common/modal_base.py`](../common/modal_base.py)
and the Exp-0 probe machinery.

What does **not** live here: the environment generator + datamodule. Like gridworld, it must be
registered in NextLat's `train.py`, so it goes in `NextLat/data/concealworld.py` (reusing gridworld
grid/tokenizer utils) with configs under `NextLat/config/concealworld/`.

Scope + writeups: [`../../docs/experiments/concealworld/`](../../docs/experiments/concealworld/).
