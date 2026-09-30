"""Synthetic positive control for the slice-illusion mechanism (reviewer Experiment A).

Controls WHERE the discriminative evidence lives and asks whether per-slice faithfulness
recovers when the evidence is local. Crucially, the slice-wise arm here uses the SAME
aggregation as the main protocol (vcf3d/faithfulness/slice_illusion.py): the per-slice
deletion AUC is computed for every axial slice and AVERAGED over slices. The volumetric arm
deletes in 3D. Two regimes, with model accuracy matched so the comparison isolates evidence
layout rather than model quality:

  CONCENTRATED : the class signal is a blob in a few adjacent axial slices.
  DISTRIBUTED  : the same total signal energy (equal added L2 norm) is spread across depth.

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

D = 32
N_TRAIN, N_EVAL = 600, 160
STEPS = 20
SIG_SLICES_CONC = 4          # concentrated: blob spans this many adjacent slices
AMP_CONC = 3.0               # per-voxel amplitude (concentrated)
AMP_DIST = 3.0              # per-voxel amplitude (distributed); tuned for matched accuracy
EPOCHS = 30


def make_volume(label, regime, rng):
    x = rng.normal(0, 1, (D, D, D)).astype(np.float32)
    if label == 1:
        if regime == "concentrated":
            d0 = D // 2 - SIG_SLICES_CONC // 2
            for d in range(d0, d0 + SIG_SLICES_CONC):
                cy, cx = rng.integers(8, D - 8, 2)
                x[cy-3:cy+3, cx-3:cx+3, d] += AMP_CONC
        else:                                            # distributed across the depth
            ds = list(range(3, D - 3, 2))
            for d in ds:
                cy, cx = rng.integers(6, D - 6, 2)
                x[cy-2:cy+2, cx-2:cx+2, d] += AMP_DIST * np.sqrt(SIG_SLICES_CONC * 36.0 / (len(ds) * 16.0))
    return x


def make_set(n, regime, rng):
    X, y = [], []
    for i in range(n):
        lab = int(i % 2); X.append(make_volume(lab, regime, rng)); y.append(lab)
    return torch.tensor(np.stack(X))[:, None], torch.tensor(y)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv3d(1, 8, 3, padding=1); self.c2 = nn.Conv3d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = F.max_pool3d(F.relu(self.c1(x)), 2)
        x = F.max_pool3d(F.relu(self.c2(x)), 2)
        return self.fc(F.adaptive_avg_pool3d(x, 1).flatten(1))


def prob1(model, x, dev):
    with torch.no_grad():
        return float(torch.softmax(model(x[None, None].to(dev)), 1)[0, 1])


def grad_saliency(model, x, dev):
    xt = x[None, None].to(dev).requires_grad_(True)
    torch.softmax(model(xt), 1)[0, 1].backward()
    return xt.grad.abs()[0, 0].cpu().numpy()


def _auc_delete(model, x, dev, order_idx, apply_fn, total):
    ps = [prob1(model, x, dev)]
    for s in range(1, STEPS + 1):
        ps.append(prob1(model, apply_fn(x, int(total * s / STEPS), order_idx), dev))
    return float(np.trapz(ps, np.linspace(0, 1, len(ps))))


def signal(model, Xev, dev, mode, fill, per_volume=False):
    """Per-volume signal = AUC(random) - AUC(real). 'vol' deletes 3D by |grad|;
    'slice' averages the within-slice deletion AUC over ALL axial slices (matching the
    main protocol's np.mean over slices)."""
    real, rand = [], []
    for x in Xev[:, 0]:
        imp = grad_saliency(model, x, dev)
        if mode == "vol":
            idx = np.argsort(-imp.ravel())

            def ap(xx, k, idx):
                xx = xx.clone().reshape(-1); xx[idx[:k]] = fill; return xx.reshape(D, D, D)
            ridx = np.random.permutation(D * D * D)
            real.append(_auc_delete(model, x, dev, idx, ap, D * D * D))
            rand.append(_auc_delete(model, x, dev, ridx, ap, D * D * D))
        else:  # slice: mean over all D axial slices
            ra, rr = [], []
            for d in range(D):
                sidx = np.argsort(-imp[:, :, d].ravel())
                rp = np.random.permutation(D * D)

                def ap(xx, k, order, d=d):
                    xx = xx.clone(); fl = xx[:, :, d].reshape(-1)
                    fl[order[:k]] = fill; xx[:, :, d] = fl.reshape(D, D); return xx
                ra.append(_auc_delete(model, x, dev, sidx, ap, D * D))
                rr.append(_auc_delete(model, x, dev, rp, ap, D * D))
            real.append(float(np.mean(ra))); rand.append(float(np.mean(rr)))
    gap = np.array(rand) - np.array(real)
    return (float(gap.mean()), gap) if per_volume else float(gap.mean())


def run(regime, dev):
    rng = np.random.default_rng(0)
    Xtr, ytr = make_set(N_TRAIN, regime, rng)
    Xev, yev = make_set(N_EVAL, regime, rng)
    model = TinyCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    model.train()
    for _ in range(EPOCHS):
        for i in range(0, len(Xtr), 32):
            opt.zero_grad()
            F.cross_entropy(model(Xtr[i:i+32].to(dev)), ytr[i:i+32].to(dev)).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xev.to(dev)).argmax(1).cpu() == yev).float().mean().item()
    fill = float(Xev.mean())
    conf = [i for i in range(len(Xev)) if yev[i] == 1 and prob1(model, Xev[i, 0], dev) > 0.6]
    Xc = Xev[conf]
    v, vg = signal(model, Xc, dev, "vol", fill, per_volume=True)
    s, sg = signal(model, Xc, dev, "slice", fill, per_volume=True)
    ratio = v / s if abs(s) > 1e-6 else float("inf")
    try:
        from scipy.stats import wilcoxon
        _, p = wilcoxon(vg, sg)
    except Exception:
        p = float("nan")
    rng2 = np.random.default_rng(1); n = len(Xc); rr = []
    for _ in range(2000):
        idx = rng2.integers(0, n, n); den = sg[idx].mean()
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
    rows = [(r, *run(r, dev)) for r in ["concentrated", "distributed"]]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["regime", "acc", "n", "vol_signal", "slice_signal", "ratio", "ci_lo", "ci_hi", "wilcoxon_p"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", r[2], f"{r[3]:.6f}", f"{r[4]:.6f}",
                        f"{r[5]:.4f}", f"{r[6]:.4f}", f"{r[7]:.4f}", f"{r[8]:.3e}"])
    print("\nSlice-wise arm averages per-slice deletion AUC over ALL axial slices, matching the "
          "main protocol. Accuracy is matched across regimes to isolate evidence layout.")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
