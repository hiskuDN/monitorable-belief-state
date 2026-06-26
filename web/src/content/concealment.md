# Catching a model that hides a secret

*Epistemic status: small synthetic experiments (~7M-parameter models, a symbolic task). A follow-up to the gridworld post, pushing on the thing I said it couldn't touch: a secret the model has a reason to hide. Early, and sharing for discussion.*

Last time I showed that a predict-ahead training objective makes a model's belief state easier to read with a simple linear probe, and I flagged the obvious limit: it said nothing about *deception*. Monitorability looked easiest exactly where safety is easy, because a plain belief-state variable is one the model has no reason to hide. So here is a first crack at the harder case. What if the model is *using* a piece of information but deliberately keeping it out of its behavior?

[Play with the live version (Demo 2).](https://probe.hiskiasdingeto.com/#/concealworld)

## The setup

A model tracks a **secret**: a running value it updates as it goes. Its behavior reveals nothing about that value until a single late decision point (the *fork*), where it finally acts on it. In between, it just acts normally. The secret is needed, but masked. We freeze the model mid-masking and ask a linear probe: what is the secret right now?

The design lesson I (again) learned the boring way: the secret cannot be a recent token, or every model reads it off for free. So it is a **running aggregate**, a sum that keeps changing as new updates arrive, which is a genuinely non-linear function of the scattered updates. The only cheap shortcut is to ignore it while wandering and reconstruct it at the fork. Which, it turns out, is exactly what some models do.

## Two ways to succeed

Two strategies both ace the task:

- **Carry.** Keep the running secret live in your state the whole time. Readable. Monitorable.
- **Defer.** Don't track it; gather all the updates at the last second and compute it at the fork. The task still gets solved, but the secret is *absent* from the model's state until the moment it acts.

From the outside these look identical. Only a probe into the internals tells them apart, and defer is the nightmare case for oversight: there is nothing to read until the decision is already happening.

## What happens

A plain next-token model **defers, every time.** Mid-masking, the secret is gone, and not just hidden from a linear probe: a strong non-linear probe can't pull it out either (late-window recovery sits at chance, across every random seed). It genuinely isn't there. It snaps into existence right at the fork. Too late to monitor.

A predict-ahead model **carries** it. Because its objective forces each internal state to predict its own future, deferring isn't an option, and the secret stays linearly readable across the masking window. Same task, same probe; one model is monitorable and the other is structurally blind. (You can watch this happen step by step in [the demo](https://probe.hiskiasdingeto.com/#/concealworld): the plain model's belief scatters across the wandering window and only snaps onto the right value at the fork, while the predict-ahead model's tracks it the whole way.)

Two things surprised me. First, this was *not* uniform across predict-ahead variants the way the gridworld result was. The reliable carrier was the **shortest-horizon** version, the one that predicts only a single step ahead. Longer horizons were hit-or-miss across seeds (sometimes carrying, sometimes deferring exactly like the plain model), which I don't fully understand yet. More predict-ahead was not more carrying. Second, looking at the representations, carrying shows up as **compression**: the carriers squeeze the secret into a clean, low-dimensional register, while the deferring runs keep a sprawling high-dimensional state and reconstruct at the end.

## Why this is the part that matters for safety

This is the failure mode that should worry anyone betting on reading model internals: **a model can use information without ever holding it where a probe can see it.** The plain model isn't even being adversarial; it just found the cheaper solution, and the cheaper solution happens to be unmonitorable. The encouraging half is that the *training objective* moves the needle: predict-ahead makes "carry it where I can see it" the path of least resistance instead of "stash it and grab it at the end."

## Limits

The honest version. This is the **distillation** case: the model is cloning a concealing policy, not *fighting* the probe. A model trained under pressure to evade a monitor is a harder, and genuinely open, question, and I don't have a clean answer there yet. It is a tiny symbolic task. And the carry-versus-defer outcome is seed-dependent for some objectives, so "predict-ahead helps" is a lever, not a guarantee. But as a first probe of deception it says something I didn't have before: when a model hides a needed secret, whether you can still read it depends on how it was trained.

---

Play with it: [Demo 2 (Concealment)](https://probe.hiskiasdingeto.com/#/concealworld). The gridworld result it builds on: [Demo 1](https://probe.hiskiasdingeto.com/) and [the first post](https://substack.com/home/post/p-202754673). Built on NextLat (Teoh et al., NeurIPS 2025).
