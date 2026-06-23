# concealworld writeups — the deception rung (incentive to conceal)

Design, results, and verdict for **concealworld** — the first experiment with an *incentive to
conceal*. Stage **2a** of the deception ladder: a model maintains a **running secret** (sum of
interleaved updates mod K) it must use to act but masks in its behavior, and we ask whether that
secret stays *linearly* decodable from hidden state mid-masking — and whether predict-ahead keeps it
decodable better than a vanilla GPT. The secret is a nonlinear running aggregate (the shell game),
which is what gives headroom where a single-token secret saturates (see [Exp-0](../gridworld/exp0.md)
for the precedent).

- **[scope.md](scope.md)** — the full scope (env, the single change vs Exp-0, masking-length sweep,
  leakage controls, probe, decision criteria, and the 2a/2b boundary). Read this first.

Code: [`../../../experiments/concealworld/`](../../../experiments/concealworld/). Raw results land
under `../../results/<date>/<tag>/` as in earlier rounds.

_Results empty until the experiment is built + run._
