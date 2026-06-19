# Monitorability as a training objective

*Epistemic status: small synthetic experiments (~6-33M-parameter models, a gridworld). Early, and I'm sharing for discussion. It includes a negative result about my own starting hypothesis that I think is worth reporting.*

Most interpretability work takes a trained model as fixed and tries to read what's inside it. I've been poking at a different question: can you *train* a model so that its internal state is easier to read in the first place? If monitoring safety-relevant state is something we'll want to do, maybe it should be a property we design for, not one we just hope shows up.

Here is the picture that got me interested.

![The belief-state monitor: a linear probe decoding each model's internal position. The vanilla model's readout is scattered and lands on the wrong cell; the predict-ahead model's is a single sharp cell on the true position.](figures/belief-monitor.svg)

Two small transformers navigate a grid they can't see directly, so each has to track its own position internally. I freeze each one and train a simple linear probe to decode "where do you think you are?" from its hidden state. The model on the right was trained with an auxiliary objective that asks it to predict its own future; the one on the left is a plain next-token model. Same task, same probe. The predictive model's belief lands on a single bright cell. The plain model's smears across the board and backs the wrong one about as often as not.

You can [play with the live version on the home page](#/).

## The setup, briefly

The setup is where it's easy to fool yourself. The grid is *partially observed*: the agent never sees its absolute position, only local wall patterns and its own moves, so it has to integrate position over time. That is the whole point. If the variable you care about sits in the current input, both a plain and a predictive model read it off trivially and you see no difference. (I learned this the boring way; my first few task designs just saturated.) You need a genuine belief-state variable, hidden behind history, with exact per-token labels you control. Then freeze and probe.

The "predict your own future" objective is NextLat (next-latent prediction). I also ran two multi-token-prediction variants as controls, plus a plain GPT baseline.

## What happens

Across the panel, the predictive objectives lift exact-position decodability from the baseline's ~0.71 to ~0.95. Two checks I cared about. It is genuine information, not a tidier linear arrangement: a strong nonlinear probe still can't pull position out of the baseline. And it survives scale: at 4× the model size the baseline actually gets *worse* at exposing position to a linear probe, so the gap widens rather than closes.

## The part I'd want a skeptic to see early

The effect is generic. NextLat does not beat the plain multi-token objectives; they all land in the same place. So this is not a special method, it is a property of predict-ahead training as a class. And the one axis where NextLat is genuinely distinctive (it learns a much more *compressed* representation) turned out to be safety-neutral when I tested whether that compactness buys a cheaper or more robust monitor. It doesn't. The most *diffuse* objective in my panel gave the *lowest*-dimensional monitor. I went in hoping compactness would be the story; it isn't.

So the durable claim is smaller than the one I started with: predict-ahead training improves the linear monitorability of belief state the model is actually using. Not "NextLat is special for safety."

## Why a smaller claim is still interesting

Because a lot of what we'd actually want to monitor is exactly this kind of state: a variable the model carries across a trajectory rather than one written in the current token. Is the action about to happen irreversible, given the history? Did the user authorize this earlier? Is the instruction the agent is about to follow one the user gave, or one that arrived inside a tool output (a prompt injection)? Has the agent crossed a sandbox boundary? None of these live in the latest token; all of them are belief-state variables.

That points at a concrete artifact: probe the hidden state for the relevant variable and gate the action *before* it happens. Block the delete if the state says permission was revoked. Refuse the tool call if the instruction driving it came from untrusted content. The bet is that predict-ahead training makes those probes more reliable, and that "monitorability" is a knob you can turn at training time rather than a fixed property of a model you're handed.

## Limits, and the next test

The honest limits. This says nothing about deception or hidden goals. Those are belief-state variables a model has an incentive *not* to represent, and a predictive objective gives it no reason to expose them, so the approach is strongest exactly where safety is easiest. It also needs labeled belief-state data, which you don't always have. And it is a toy so far.

The next test I care about is external validity: port the same clean-label method to a small synthetic tool-use environment, with permission state and prompt-injection provenance as the probe targets, and see whether the gap survives off the grid. If it holds, "train models to be monitorable, then gate actions on what you can read" starts to look like a real, if narrow, safety tool. If it doesn't, that is worth knowing too.

---

Try the [interactive demo](#/). Built on NextLat (Teoh et al., NeurIPS 2025).
