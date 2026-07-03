# Reading a model's mind as it reads a story

*Epistemic status: small synthetic experiments (~7M-parameter models), now on a language surface instead of a grid. Third in a series. It includes two honest self-corrections: a single-seed run that briefly fooled me, and a first task design that turned out too hard to mean anything. Early, sharing for discussion.*

The [previous two posts](https://hisku.substack.com/p/monitorability-as-a-training-objective) ran in symbolic worlds: a gridworld where a model tracks its own position, then a symbolic task where a model hides a running secret. Both times a predict-ahead training objective kept the model's belief state legible to a simple linear probe, where a vanilla model let it go dark. The obvious objection writes itself: real models work on *language*, not grids and symbols. So this time I moved the same test onto text.

[Play with the live version (Demo 3).](https://probe.hiskiasdingeto.com/#/storyworld)

![One frozen frame from the middle of a story. The ball is on shelf 3 (gold-outlined). The predict-ahead model (right) reads shelf 3 sharply; the plain model (left) is confidently on shelf 1, and simply wrong.](figures/storyworld-demo.svg)

## The setup

A short narrative moves a ball around a few numbered shelves. "The ball starts on shelf 2." Then a run of sentences nudges it by relative steps: "Alice moves the ball up one shelf," "Bob moves it down one." A distractor object (a book) is being shuffled around at the same time, and pronouns ("it") stand in for the objects, so the model has to do a little coreference. At the end comes the question: "Where is the ball?" Both models answer it correctly.

The design lesson I keep relearning the boring way: the thing you probe cannot be a word on the page. The ball's *current* shelf is never written down after the first line. It is a running total of all the relative moves, a genuinely non-linear function of the scattered sentences. If it sat in a recent token, every model would read it off for free and there would be nothing to see. So the belief is engineered to live only in the model's head. Then we freeze the model mid-story and ask a linear probe one thing: which shelf is the ball on right now?

## Carry versus defer, one more time

As in the concealment post, there are two ways to ace this task:

- **Carry.** Keep the running shelf live in your state, updating it as each move arrives. Readable at every step. Monitorable.
- **Defer.** Don't track it; hold the raw moves and total them up only once the question arrives. The task still gets solved, but the answer is *absent* from the model's state until the final moment.

From the outside they are indistinguishable: both say "shelf 2" when asked. Only a probe into the internals tells them apart, and defer is the case that should worry a monitor, because there is nothing to read until the decision is already being made.

## What happens

On the eased task (four shelves, six moves, one distractor), both a plain GPT and a NextLat model that predicts a single step ahead answer the final question perfectly, on every seed. But watch the monitor.

Reading the NextLat model, a linear probe is right about 95 percent of the time across the whole story, and effectively 100 percent by the last move before the question. Reading the plain GPT model, the same probe is right closer to 60 percent overall, and by that last move it has collapsed toward chance, having quietly dropped the ball's location somewhere in the middle. It snaps back to the truth only at the instant the question is asked. Same story, same probe, same correct answer; one model is transparent throughout and the other is a black box until it acts.

The figure above is one frozen frame from the middle of a story. The ball is on shelf 3. The predictive model reads shelf 3, sharply. The plain model reads shelf 1, at 68 percent confidence, and it is simply wrong. If you were a monitor trying to catch the ball reaching a forbidden shelf, one of these models lets you and the other does not.

Zoom out from that single frame to the whole story and the problem takes on a shape:

![Monitor read-accuracy versus story position. Both models start pinned at 100 percent while the ball sits still, then as the ball moves the NextLat line stays flat near 100 percent while the GPT line decays toward chance, snapping back to 100 percent only at the question.](figures/storyworld-retention.svg)

The two models are identical until the ball starts moving. Then they split. The NextLat line stays flat along the top: the belief is legible at every step. The GPT line slides down toward chance, sentence by sentence, as it lets the running location decay out of its state. It recovers in one vertical jump, but only at the question, when the model finally needs the answer to act. Everywhere a monitor would actually want to look, the plain model has nothing to show it.

## The parts I want a skeptic to see

Three honest things, because the path here was not clean.

**A single seed fooled me.** My first eased run used one random seed, and the plain model happened to score badly on the task itself. I briefly wrote down the wrong conclusion: that *carrying is necessary to solve*. Three seeds killed that. The plain model solves the task perfectly and still defers. The finding is carry-versus-defer, not carry-or-fail. Both models know the answer; only one holds it where you can see it, and that is the whole point.

**My first story was too hard to mean anything.** Before the eased task I tried a harder one (six shelves, sixteen moves, pronouns throughout) and *no* model could answer the final question, predictive or plain. It is tempting to read a flat probe there as "the belief is hidden." It means nothing: you cannot monitor a belief a model never formed. A capability check (does the model even get the final answer right?) caught this, and I threw the run out. You have to confirm the model *can* do the task before you are allowed to say a word about whether you can read it.

**It is the short-horizon variant again.** Exactly as in the concealment post, the reliable carrier is the model predicting just *one* step ahead. The longer-horizon and multi-token variants were either seed-unstable or could not learn this particular task at all. More predict-ahead is not more carrying, and I still do not fully understand why the shortest horizon is the sweet spot.

## It degrades, on purpose

Push the story harder (add real coreference and a second distractor) and the gap frays but survives: the predictive model still carries the belief on two of three seeds, the plain model defers on all three. Push further, into the version no small model can track at all, and everyone goes dark, the carrier included. So this is not a law; it is an effect with a ceiling. It holds while the task sits inside what a small model can actually track, and thins as the language surface gets harder. That is the honest shape of it, and I would rather show you the fraying than a single cherry-picked story.

## Why this matters for safety

This is the same failure mode as before, one step closer to the thing we actually care about. A model can *use* a fact threaded through a passage of text, with distractors and pronouns in the way, without ever holding that fact where a probe can read it. And the things we would most want to monitor in a real language model are exactly this shape: did the user authorize this action earlier in the conversation, or did the instruction arrive inside a tool result; what does the agent currently believe about a value it has been updating; is the plan it is about to run the one it described three paragraphs ago. None of these are words in the latest token. All of them are beliefs carried across language.

The encouraging half stays encouraging on this surface: the training objective moves the needle. Predict-ahead makes "keep the belief legible" the cheap path, even when the belief is buried in sentences.

## Limits

The honest version. This is a templated, from-scratch, tiny model, not a real pretrained LLM, and the language is a thin surface (sentences, a distractor, pronouns) wrapped around an engineered running total, not genuine reasoning. The carry advantage degrades as the narrative gets harder and vanishes once the task outgrows what a small model can track at all. And it is seed-dependent for the less reliable variants, so "predict-ahead helps" is a lever, not a guarantee. But as a first move off the grid and onto language, it says the thing I wanted to know: the carry-versus-defer split is not an artifact of symbolic toys. It shows up in text too, and how a model was trained decides which side it lands on.

---

Play with it: [Demo 3 (Storyworld)](https://probe.hiskiasdingeto.com/#/storyworld). The results it builds on: [Demo 1 (Gridworld)](https://probe.hiskiasdingeto.com/) and [Demo 2 (Concealment)](https://probe.hiskiasdingeto.com/#/concealworld), and the earlier posts, [one](https://hisku.substack.com/p/monitorability-as-a-training-objective) and [two](https://hisku.substack.com/p/catching-a-model-that-hides-a-secret). Built on NextLat (Teoh et al., NeurIPS 2025).
