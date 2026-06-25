"""Adversarial probe-evasion objective for concealworld 2b.

A gradient-reversal linear probe is trained on the model's last hidden state to decode the
running secret S_t at wandering positions. The probe minimises its own cross-entropy (it
becomes a strong reader), but the gradient reversal pushes the *trunk* to make S_t HARDER to
read linearly — i.e. the model is rewarded for being unmonitorable, while still solving the
task (the next-token loss is untouched). `lambda_adv` is the adversarial strength (the GRL
scale); lambda_adv=0 disables it entirely (no head is built, training is identical to 2a).

The probe target is recomputed from the input tokens (the running secret is a deterministic
function of them), so nothing has to be threaded through the packed batch. Token convention
(from data/concealworld.py): 0=pad, 1=end, 2=wait, 3=act, 4..3+K = u0..u{K-1} (update by
d=id-4), 4+K.. = a{...}. The running sum resets at each `end`; wandering positions are exactly
the update/wait tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return _GradReverse.apply(x, lambd)


def running_secret_from_ids(input_ids, K):
    """Recompute per-position running secret S_t (mod K) + wandering mask from tokens.
    Returns (target [B,T] int64 with -100 outside the wandering window, in_window bool [B,T]).
    Validated to match data/concealworld.py exactly, including episode resets under packing."""
    ids = input_ids.long()
    is_update = (ids >= 4) & (ids < 4 + K)
    is_wait = ids == 2
    is_end = ids == 1
    d = torch.where(is_update, ids - 4, torch.zeros_like(ids))
    P = torch.cumsum(d, dim=1)                       # raw prefix sum of update deltas
    seg_start = torch.zeros_like(is_end)             # position starts a new episode if prev tok was 'end'
    seg_start[:, 1:] = is_end[:, :-1]
    seg_start[:, 0] = True
    P_prev = torch.zeros_like(P)
    P_prev[:, 1:] = P[:, :-1]
    start_val = torch.where(seg_start, P_prev, torch.zeros_like(P))
    offset = torch.cummax(start_val, dim=1).values   # P at current episode's start (P nondecreasing)
    R = (P - offset) % K
    in_window = is_update | is_wait
    target = torch.where(in_window, R, torch.full_like(R, -100))
    return target, in_window


def adversarial_probe_loss(head, hidden, inputs, K, lambd):
    """CE of the gradient-reversed linear probe predicting S_t over wandering positions.
    `hidden` [B,S,d] must be aligned to `inputs` [B,S] (same slice fed to the trunk)."""
    target, _ = running_secret_from_ids(inputs, K)        # [B,S], -100 outside window
    logits = head(grad_reverse(hidden, lambd))            # [B,S,K]
    B, S, Kc = logits.shape
    tgt = target.reshape(B * S)
    if (tgt != -100).any():
        return F.cross_entropy(logits.reshape(B * S, Kc), tgt, ignore_index=-100)
    return logits.sum() * 0.0
