# concealworld writeups — the deception rung (incentive to conceal)

Design, results, and verdict for **concealworld** — the first experiment with an *incentive to
conceal*. Stage **2a** of the deception ladder: clone a teacher that masks a secret it nonetheless
needs to act, and ask whether the secret stays linearly recoverable from hidden state during the
masking window — and whether predict-ahead keeps it recoverable longer. Built as a minimal
one-variable extension of the [Exp-0](../gridworld/exp0.md) decoupled-bit harness.

- **[scope.md](scope.md)** — the full scope (env, the single change vs Exp-0, masking-length sweep,
  leakage controls, probe, decision criteria, and the 2a/2b boundary). Read this first.

Code: [`../../../experiments/concealworld/`](../../../experiments/concealworld/). Raw results land
under `../../results/<date>/<tag>/` as in earlier rounds.

_Results empty until the experiment is built + run._
