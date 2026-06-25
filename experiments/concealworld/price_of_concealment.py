"""Offline price-of-concealment study (concealworld 2b, frozen-features version).

Question: can the running secret S_t be made LINEARLY un-decodable from a trained model's
hidden states WHILE preserving the information predict-ahead needs — and at what cost?

We learn an encoder phi: h -> z (+ decoder psi for reconstruction). Reconstruction error is
the predict-ahead-cost proxy: an INVERTIBLE phi preserves all of h's info, so predict-ahead is
satisfiable in z-space identically (D' = phi . D . phi^-1). An ensemble of linear probes is
trained (gradient-reversed into phi) to push S_t's LINEAR decodability down. We sweep the
adversary weight beta and, at each, evaluate with FRESH held-out linear + MLP probes:

  price curve = linear-decodability(S_t) vs reconstruction error (predict-ahead cost).
  linear falls at ~0 recon cost  -> exposure was incidental  -> cheap monitoring FRAGILE.
  linear can't fall without recon cost -> exposure load-bearing -> monitoring ROBUST.
  MLP still reads it while linear falls -> ENCRYPTED (present, non-linear), not destroyed.

Usage (venv):
    ./.venv/bin/python experiments/concealworld/price_of_concealment.py \
        docs/results/<date>/2b_l0.0_features --arm nextlat_h1 --seed 1234
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def grad_reverse(x, lambd):
    return _GradReverse.apply(x, lambd)


def mlp(din, dh, dout, depth=2):
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, dh), nn.GELU()]
        d = dh
    layers += [nn.Linear(d, dout)]
    return nn.Sequential(*layers)


def episode_split(ri, frac=0.8, seed=0):
    rng = np.random.default_rng(seed)
    eps = np.unique(ri)
    rng.shuffle(eps)
    tr = set(eps[: int(frac * len(eps))].tolist())
    mask = np.array([r in tr for r in ri])
    return mask


def fresh_probes(Ztr, ytr, Zva, yva):
    """Held-out linear + MLP decodability of S_t from z (independent of the training ensemble)."""
    lin = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, ytr)
    lin_acc = accuracy_score(yva, lin.predict(Zva))
    m = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, early_stopping=True).fit(Ztr, ytr)
    mlp_acc = accuracy_score(yva, m.predict(Zva))
    chance = np.bincount(yva).max() / len(yva)
    return float(lin_acc), float(mlp_acc), float(chance)


def _resid_net(d, dh, device):
    """Near-identity residual map: x + net(x), net's last layer zero-init so it starts = identity."""
    net = mlp(d, dh, d).to(device)
    last = [m for m in net if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
    return net


def run_beta(Xtr, ytr, Xva, yva, beta, K, d, steps=2500, bs=4096, n_probes=6, seed=0,
             device="cpu", lambda_rec=30.0):
    torch.manual_seed(seed)
    phi_net = _resid_net(d, 512, device)   # z = x + phi_net(x), starts as identity
    psi_net = _resid_net(d, 512, device)   # recon = z + psi_net(z)
    phi = lambda x: x + phi_net(x)
    psi = lambda z: z + psi_net(z)
    probes = nn.ModuleList([nn.Linear(d, K) for _ in range(n_probes)]).to(device)
    params = lambda: list(phi_net.parameters()) + list(psi_net.parameters()) + list(probes.parameters())
    opt = torch.optim.Adam(params(), lr=1e-3)
    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr, device=device)
    n = Xt.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        xb, yb = Xt[idx], yt[idx]
        z = phi(xb)
        rec = F.mse_loss(psi(z), xb)
        # ensemble decodes z; GRL pushes phi to hide S_t linearly (strength beta).
        # recon is weighted heavily so the encoder CANNOT collapse z to defeat the probe.
        adv = torch.stack([F.cross_entropy(p(grad_reverse(z, beta)), yb) for p in probes]).mean()
        if beta > 0 and step > 0 and step % 300 == 0:
            j = (step // 300) % n_probes
            probes[j] = nn.Linear(d, K).to(device)
            opt = torch.optim.Adam(params(), lr=1e-3)
        loss = lambda_rec * rec + adv
        opt.zero_grad(); loss.backward(); opt.step()
    # evaluate
    with torch.no_grad():
        Xva_t = torch.tensor(Xva, device=device)
        Zva = phi(Xva_t).cpu().numpy()
        Ztr = phi(Xt).cpu().numpy()
        rec_va = F.mse_loss(psi(phi(Xva_t)), Xva_t).item()  # reconstruction = invertibility/predict-ahead proxy
    var = float(np.var(Xva))
    rec_norm = rec_va / var  # normalized reconstruction error (0 = perfect, 1 = predict-the-mean)
    lin, mlpa, ch = fresh_probes(Ztr, ytr, Zva, yva)
    return {"beta": beta, "rec_norm": rec_norm, "lin": lin, "mlp": mlpa, "chance": ch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features_dir")
    ap.add_argument("--arm", default="nextlat_h1")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--late-off", type=int, default=16, help="restrict to wandering offset >= this")
    ap.add_argument("--betas", default="0.0,1.0,3.0,10.0,30.0")
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()

    f = os.path.join(args.features_dir, f"features_{args.arm}_seed{args.seed}.npz")
    dat = np.load(f)
    X, y, off, ri, K = dat["X"].astype(np.float32), dat["y"].astype(np.int64), dat["off"], dat["ri"], int(dat["n_states"])
    sel = off >= args.late_off
    X, y, ri = X[sel], y[sel], ri[sel]
    print(f"[price] {args.arm} seed{args.seed}: {X.shape[0]} late (off>={args.late_off}) positions, d={X.shape[1]}, K={K}")

    tr = episode_split(ri, 0.8, seed=0)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xva = sc.transform(X[tr]).astype(np.float32), sc.transform(X[~tr]).astype(np.float32)
    ytr, yva = y[tr], y[~tr]
    d = X.shape[1]

    print(f"\n{'beta':>6} | {'recon_err':>9} | {'lin-decode':>10} {'mlp-decode':>10} {'chance':>6} | reading")
    print("-" * 70)
    rows = []
    for beta in [float(b) for b in args.betas.split(",")]:
        r = run_beta(Xtr, ytr, Xva, yva, beta, K, d, steps=args.steps)
        rows.append(r)
        lin_ch, mlp_ch = r["lin"] - r["chance"], r["mlp"] - r["chance"]
        if lin_ch < 0.08 and mlp_ch > 0.25:
            read = "ENCRYPTED (linear gone, MLP holds)"
        elif lin_ch < 0.08 and mlp_ch < 0.12:
            read = "DESTROYED (both gone)"
        elif lin_ch > 0.25:
            read = "still linearly exposed"
        else:
            read = "partial"
        print(f"{r['beta']:>6.1f} | {r['rec_norm']:>9.3f} | {r['lin']:>10.2f} {r['mlp']:>10.2f} {r['chance']:>6.2f} | {read}")
    print("\nprice = how far linear-decode falls vs recon_err (predict-ahead cost).")
    print("low recon_err with linear->chance => exposure incidental => cheap monitoring FRAGILE.")


if __name__ == "__main__":
    main()
