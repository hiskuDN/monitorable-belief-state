# lieworld — the language / deception rung

Experiment glue for **lieworld**: when a model is induced to emit a false answer, is the *suppressed
true belief* still linearly recoverable from its hidden state, and does predict-ahead training make it
more monitorable? The first experiment in the program that touches "is the model lying," kept novel
against SAPLMA-style lie-detection probing by a train-from-scratch **objective contrast** plus a
**history-integrated, output-decoupled** target.

What lives here: the probe / analyze scripts and the Modal entrypoints for this family
(`modal_app.py`), reusing the shared scaffolding in [`../common/modal_base.py`](../common/modal_base.py).

What does **not** live here: the environment generator + datamodule. Like gridworld/toolworld, it must
be registered in NextLat's `train.py` to be trainable, so it goes in `NextLat/data/lieworld.py` with
configs under `NextLat/config/lieworld/`.

Scope + writeups: [`../../docs/experiments/lieworld/`](../../docs/experiments/lieworld/).
