# Candidate Safety-Monitoring Tasks (small-model, clean-label)

*As of 2026-06-19. A ranked menu of simple synthetic tasks for testing whether
predictive-representation objectives make *safety-relevant* belief state more monitorable, in the
same small-model / clean-label paradigm as [Experiment 0](experiments/gridworld/exp0.md). This is the
concrete task backlog behind [agentic-application.md](agentic-application.md); see
[research-overview.md](research-overview.md) for the program and where this sits.*

## The shared recipe

Every task reuses the discipline that made Exp 0 work:

- a tiny model (~6–33M params) trained **from scratch**, vanilla (GPT) vs a predictive objective
  (NextLat / MTP / JTP);
- the safety-relevant variable is **hidden behind history** (partial observability) so it must be
  *tracked*, not read off the current token — otherwise the baseline ceilings and the test can't
  discriminate (the v1/v2 saturation lesson);
- **exact per-token ground-truth labels** from the environment generator, with a leakage audit;
- freeze the model, then **linear-probe** the hidden states for the safety variable.

The question at each task: **is the safety-relevant state more linearly recoverable under a
predictive objective than under the vanilla baseline?** (Step 2 showed compactness itself buys no
monitor advantage, so we no longer expect a NextLat-*specific* edge — these tasks test the
*class-level* claim and its external validity.)

## The menu

**Operational / expressed — the regime the method fits:**

1. **Permission / authorization** — is an action allowed *now*, given scattered grant/revoke events?
2. **Instruction-binding / override** — was an earlier instruction superseded, or is it still in force?
3. **Trust / provenance (prompt-injection)** — did the instruction the agent is about to follow
   originate from untrusted content (a tool output) vs the user?
4. **Sensitive-data exposure** — a secret surfaced earlier; would the next action write/send/leak it?
5. **Irreversibility** — does the next action cross a point-of-no-return, given history?
6. **Sandbox / boundary** — is the agent inside or outside the allowed region after a sequence of moves/`cd`s?
7. **Budget / resource exhaustion** — hidden counter (quota, money, steps); would the next action overrun it?
8. **Stale state / consistency** — a fact was mutated later; is the agent acting on superseded info?
9. **Precondition / readiness** — are the required setup steps done, or is the agent about to skip one?
10. **Constraint / invariant** — a standing rule ("never A after B"); would the next action violate it?

**Hard / decoupled — stress the known weak spot:**

11. **Deception / hiding** — give the model an *incentive to conceal* a variable; is it still
    linearly recoverable? (the real test of whether this reaches deception)
12. **Theory-of-mind / who-knows-what** — track what *another* agent believes (false-belief); for
    multi-agent collusion/deception.

## Ranking (S × C × L)

`S` = safety value · `C` = cheapness to build+run · `L` = how cleanly a ~10M model learns it (clean
labels + headroom). Each 1–5; `score` = product.

| # | Task | S | C | L | score |
| --- | --- | --- | --- | --- | --- |
| 1 | **Permission / authorization** | 4 | 5 | 5 | **100** |
| 2 | **Instruction-binding / override** | 4 | 5 | 5 | **100** |
| 3 | **Trust / provenance (prompt-injection)** | 5 | 4 | 4 | **80** |
| 4 | Sandbox / boundary | 3 | 5 | 5 | 75 |
| 5 | Sensitive-data exposure | 4 | 4 | 4 | 64 |
| 6 | Irreversibility | 4 | 5 | 3 | 60 |
| 6 | Budget / resource exhaustion | 3 | 5 | 4 | 60 |
| 8 | Stale-state / consistency | 4 | 4 | 3 | 48 |
| 8 | Precondition / readiness | 3 | 4 | 4 | 48 |
| 10 | Constraint / invariant | 3 | 4 | 3 | 36 |
| 10 | Theory-of-mind | 4 | 3 | 3 | 36 |
| 12 | Deception / hiding | 5 | 2 | 2 | 20 |

## Chosen first two

1. **Permission / authorization** — cheapest, cleanest, highest-learnability; a single binary flag
   toggled over history. It doubles as the **external-validity gate** (does the carried-bit effect
   survive a tool-trace surface, off the gridworld?), so it's the foundation.
2. **Trust / provenance (prompt-injection)** — highest *safety* value and structurally *different*
   (source-binding, not a toggle), mapping onto the #1 real agentic threat.

Why not the other top scorers: **Instruction-binding** ties #1 but is structurally identical to
Permission (a flag set/overridden over history), so it adds little; **Sandbox/boundary** is
essentially gridworld-redux and tests almost nothing new. Pairing Permission with the
structurally-distinct Trust/provenance covers the most ground.

## First experiment sketch — the Permission "toolworld"

*(To be fully scoped before building, like the Exp-0 plan.)* A compact symbolic tool-use simulator:

- **Vocab (~30–50 tokens):** a few file IDs, tools (`ls read write delete grant revoke`), permission
  tokens, outcome tokens, separators. Traces ~100–300 tokens (within `block_size`).
- **Hidden state:** one (later: a few) permission bit(s), toggled by `grant`/`revoke` events
  scattered through the trace and **never restated** — so "is `delete` allowed now?" must be
  integrated from history (partial observability = the headroom).
- **Probe target (per token):** `is_delete_permitted` (binary). Later add `file_exists[f]`,
  `next_action_policy_compliant`, `irreversible`.
- **Controls:** permission ⟂ confounds (position, file identity, trace length); 50/50 class balance;
  leakage audit, as in Exp 0.
- **Arms:** GPT vs one predictive objective (start with MTP or NextLat) at the Exp-0 model size.
- **Read:** linear-probe accuracy/AUROC for `is_delete_permitted`, baseline vs predictive, with
  chance + shuffled floors. Confirm headroom (baseline well below ceiling) *first*; if it saturates,
  deepen the partial observability before trusting any gap.

Grow only after a clean single-variable read: add files/staleness, then `next-action-compliance`,
then a pre-action gate (Stage 3 of [agentic-application.md](agentic-application.md)).
