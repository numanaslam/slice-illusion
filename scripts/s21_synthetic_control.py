"""Synthetic positive control for the slice-illusion mechanism (reviewer Experiment A).

The claim is causal: slice-wise faithfulness fails *because* a 3D model distributes evidence
across depth, so a one-slice perturbation removes too little of it. The clean test is a
synthetic dataset where we control where the evidence lives:

  CONCENTRATED regime : the class signal is a blob in ONE fixed axial slice; all other
                        slices are noise. Evidence is not distributed, so slice-wise
                        evaluation SHOULD work (signal ratio near 1).
  DISTRIBUTED regime  : the same total signal energy is spread across many slices. Evidence
                        is distributed, so the slice illusion SHOULD appear (ratio large).

If the ratio is ~1 in CONCENTRATED and large in DISTRIBUTED, the mechanism is established:
the illusion is caused by evidence distribution, not by slice-wise evaluation per se.

Self-contained: generates data, trains a small 3D CNN per regime, measures the volumetric
and slice-wise faithfulness signals with equal-volume deletion. Runs in minutes; CPU is fine.

    python -m scripts.s21_synthetic_control
Output: experiments/results/synthetic_control.csv
"""
import argparse
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 32                    # cube side
N_TRAIN, N_EVAL = 400, 120
STEPS = 20


def make_volume(label, regime, rng):
    x = rng.normal(0, 1, (D, D, D)).astype(np.float32)
    if label == 1:
        if regime == "concentrated":
            d = D // 2
            cy, cx = rng.integers(8, D - 8, 2)
            x[cy-3:cy+3, cx-3:cx+3, d] += 4.0          # bright blob in one slice
        else:                                           # distributed across depth
            for d in range(4, D - 4, 3):
                cy, cx = rng.integers(6, D - 6, 2)
                x[cy-2:cy+2, cx-2:cx+2, d] += 4.0 / np.sqrt(len(range(4, D - 4, 3)))
    return x


def make_set(n, regime, rng):
    X, y = [], []
    for i in range(n):
        lab = int(i % 2)
        X.append(make_volume(lab, regime, rng)); y.append(lab)
    X = torch.tensor(np.stack(X))[:, None]              # (N,1,D,D,D)
    return X, torch.tensor(y)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv3d(1, 8, 3, padding=1); self.c2 = nn.Conv3d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = F.max_pool3d(F.relu(self.c1(x)), 2)
        x = F.max_pool3d(F.relu(self.c2(x)), 2)
        x = F.adaptive_avg_pool3d(x, 1).flatten(1)      # GAP head (as in the paper)
        return self.fc(x)


def prob1(model, x, dev):
    with torch.no_grad():
        return float(torch.softmax(model(x[None, None].to(dev)), 1)[0, 1])


def voxel_importance(model, x, dev):
    """gradient-magnitude saliency per voxel."""
    xt = x[None, None].to(dev).requires_grad_(True)
    p = torch.softmax(model(xt), 1)[0, 1]
    p.backward()
    return xt.grad.abs()[0, 0].cpu().numpy()


def signal(model, Xev, dev, mode, per_volume=False):
    """faithfulness signal = AUC(random) - AUC(best-real), equal-volume deletion.
    mode 'vol' deletes 3D cubes by importance; 'slice' deletes within one axial slice.
    per_volume=True also returns the per-volume (random_auc - real_auc) gap array."""
    real_auc, rand_auc = [], []
    fill = float(Xev.mean())
    for x in Xev[:, 0]:
        imp = voxel_importance(model, x, dev)
        if mode == "slice":
            d = int(np.unravel_index(imp.sum((0, 1)).argmax(), imp.sum((0, 1)).shape)[0]) \
                if imp.ndim == 3 else 0
            d = int(imp.sum((0, 1)).argmax())           # most-important slice
            order_imp = imp[:, :, d].ravel()
            idx = np.argsort(-order_imp)
            def apply(xx, k):
                xx = xx.clone(); flat = xx[:, :, d].reshape(-1)
                flat[idx[:k]] = fill; xx[:, :, d] = flat.reshape(D, D); return xx
            total = D * D
        else:
            order_imp = imp.ravel(); idx = np.argsort(-order_imp)
            def apply(xx, k):
                xx = xx.clone().reshape(-1); xx[idx[:k]] = fill; return xx.reshape(D, D, D)
            total = D * D * D
        for imp_use, acc in [(imp, real_auc), (None, rand_auc)]:
            if imp_use is None:
                ridx = np.random.permutation(total)
                if mode == "slice":
                    def apply_r(xx, k):
                        xx = xx.clone(); flat = xx[:, :, d].reshape(-1)
                        flat[ridx[:k]] = fill; xx[:, :, d] = flat.reshape(D, D); return xx
                else:
                    def apply_r(xx, k):
                        xx = xx.clone().reshape(-1); xx[ridx[:k]] = fill; return xx.reshape(D, D, D)
                ap = apply_r
            else:
                ap = apply
            ps = []
            for s in range(STEPS + 1):
                k = int(total * s / STEPS)
                ps.append(prob1(model, ap(x, k), dev))
            acc.append(np.trapz(ps, np.linspace(0, 1, len(ps))))
    gap = np.array(rand_auc) - np.array(real_auc)          # per-volume signal
    if per_volume:
        return float(gap.mean()), gap
    return float(gap.mean())


def run(regime, dev):
    rng = np.random.default_rng(0)
    Xtr, ytr = make_set(N_TRAIN, regime, rng)
    Xev, yev = make_set(N_EVAL, regime, rng)
    model = TinyCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    model.train()
    for epoch in range(15):
        for i in range(0, len(Xtr), 32):
            xb, yb = Xtr[i:i+32].to(dev), ytr[i:i+32].to(dev)
            opt.zero_grad(); loss = F.cross_entropy(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xev.to(dev)).argmax(1).cpu() == yev).float().mean().item()
    conf = [i for i in range(len(Xev)) if yev[i] == 1 and prob1(model, Xev[i, 0], dev) > 0.6]
    Xc = Xev[conf]
    v, vg = signal(model, Xc, dev, "vol", per_volume=True)
    s, sg = signal(model, Xc, dev, "slice", per_volume=True)
    ratio = v / s if abs(s) > 1e-6 else float("inf")
    # paired Wilcoxon (per-volume volumetric vs slice-wise signal) + bootstrap CI on the ratio
    try:
        from scipy.stats import wilcoxon
        _, p = wilcoxon(vg, sg)
    except Exception:
        p = float("nan")
    rng = np.random.default_rng(0); n = len(Xc); rr = []
    for _ in range(2000):
        idx = rng.integers(0, n, n); den = sg[idx].mean()
        rr.append(vg[idx].mean() / den if abs(den) > 1e-9 else np.nan)
    rr = np.array(rr); rr = rr[np.isfinite(rr) & (rr > 0)]
    lo, hi = (np.percentile(rr, [2.5, 97.5]) if len(rr) else (np.nan, np.nan))
    print(f"[{regime}] acc={acc:.2f} n={n} vol_sig={v:+.4f} slice_sig={s:+.4f} "
          f"ratio={ratio:.1f}x (95% CI {lo:.1f}-{hi:.1f}) paired p={p:.2e}", flush=True)
    return acc, n, v, s, ratio, float(lo), float(hi), p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="experiments/results/synthetic_control.csv")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    rows = []
    for regime in ["concentrated", "distributed"]:
        rows.append((regime, *run(regime, dev)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["regime", "acc", "n", "vol_signal", "slice_signal", "ratio", "ci_lo", "ci_hi", "wilcoxon_p"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", r[2], f"{r[3]:.6f}", f"{r[4]:.6f}",
                        f"{r[5]:.4f}", f"{r[6]:.4f}", f"{r[7]:.4f}", f"{r[8]:.3e}"])
    print("\nExpected: concentrated ratio ~1 (slice-wise works), distributed ratio large "
          "(slice illusion). This isolates evidence distribution as the cause.")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
