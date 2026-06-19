# toolworld — round 2 (agentic permission gating)

Experiment glue for the **Permission "toolworld"** (task #1 of
[`docs/safety-task-menu.md`](../../docs/safety-task-menu.md)): the external-validity test of
whether predict-ahead training improves belief-state monitorability *off the gridworld*, on a
tool-trace surface.

What lives here: the probe / analyze scripts and the Modal entrypoints for this family
(`modal_app.py`), reusing the shared scaffolding in [`../common/modal_base.py`](../common/modal_base.py).

What does **not** live here: the environment generator + datamodule. Like gridworld, it must be
registered in NextLat's `train.py` to be trainable, so it goes in `NextLat/data/toolworld.py`
with configs under `NextLat/config/toolworld/`.

Writeups + plan: [`../../docs/experiments/toolworld/`](../../docs/experiments/toolworld/).
