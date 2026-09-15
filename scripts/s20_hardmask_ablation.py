"""Hard-mask ablation + effective-mask audit (reviewer's blocking control).

A reviewer worried that the soft Gaussian mask M in R(x,Omega)=x(1-M)+sM might, for the
one-voxel-thick K=1 region, never reach 1, so that K=1 only lightly attenuates a slice
rather than deleting it, which would inflate the slice illusion. Two diagnostics settle it:

  (A) EFFECTIVE MASK AUDIT. For each K, report the mean feathered-mask value inside the
      NOMINAL band region Omega (no model needed). If this is ~1, the region is genuinely
      deleted, not attenuated. (The operator dilates by max-pool then blurs then divides by
      the peak, so the peak is 1 by construction; this measures the interior mean.)

  (B) HARD-MASK RERUN. Repeat the K-sweep with a hard mask (M = the boolean region, no
      taper) and report the signal and ratio at each K. If the ratio is stable against the
      soft-mask result, the taper is not driving the effect.

    python -m scripts.s20_hardmask_ablation --config configs/brats_agg.yaml     # residual
    python -m scripts.s20_hardmask_ablation --config configs/rtx3070.yaml       # plain CNN
Output: experiments/results/hardmask_ablation_<config>.csv
"""
import argparse
import csv
import glob
import os
from collections import defaultdict
import numpy as np
import torch

from vcf3d.config import load_config
from vcf3d.models.backbone import load_backbone
from vcf3d.io.datasets import make_loader
from vcf3d.perturb.tissue_bank import build_tissue_bank
from vcf3d.perturb.anatomical_replace import anatomical_replace, _feather, _context_nn
from vcf3d.utils.supervoxels import supervoxels
from vcf3d.utils.gradcam import grad_cam_3d, supervoxel_importance
from vcf3d.concept.tcav import tcav_saliency
from vcf3d.faithfulness.slice_illusion import slice_illusion

KS = [1, 4, 16, 96]


def hard_replace(x, mask, bank, cfg):
    """R with a HARD mask: no Gaussian feather, no dilation. Same surrogate fill."""
    m = mask.to(x.device)
    s = _context_nn(x, m, bank, cfg) if cfg.faith.surrogate == "context-nn" else x.mean()
    Mf = m.to(x.dtype)
    return x * (1 - Mf) + s * Mf


def _panel(bb, x, L, cls, cfg, cavs):
    S = int(L.max())
    imp = {"gradcam": supervoxel_importance(grad_cam_3d(bb, x.unsqueeze(0), cls, cfg), L)}
    for cn, cv in cavs.items():
        imp[f"tcav_{cn}"] = supervoxel_importance(tcav_saliency(bb, x.unsqueeze(0), cv, cls, cfg), L)
    imp["random"] = np.random.rand(max(S, 1)).astype(np.float32)
    return imp


def _sig(rep):
    real = [m for m in rep["methods"] if m != "random"]
    idx = {m: i for i, m in enumerate(rep["methods"])}
    v = rep["vol_auc"][idx["random"]] - min(rep["vol_auc"][idx[m]] for m in real)
    s = rep["slice_auc"][idx["random"]] - min(rep["slice_auc"][idx[m]] for m in real)
    return v, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/brats_agg.yaml")
    ap.add_argument("--max_vols", type=int, default=10**9)
    a = ap.parse_args()
    cfg = load_config(a.config)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    if cfg.compute.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    dev = cfg.compute.device
    bb = load_backbone(cfg)
    sigma = float(cfg.faith.feather_sigma)

    held = [b["image"][0] for b in make_loader(
        [{"image": f} for f in glob.glob(os.path.join(cfg.paths.processed, "heldout", "*.nii*"))], cfg)]
    bank = build_tissue_bank(held, None, cfg) if held else None
    cav_path = os.path.join(cfg.paths.results, "cavs.npy")
    cavs = np.load(cav_path, allow_pickle=True).item() if os.path.exists(cav_path) else {}

    eff = defaultdict(list)                         # effective mask mean inside nominal Omega, per K
    soft = {K: defaultdict(list) for K in KS}       # signals under soft mask
    hard = {K: defaultdict(list) for K in KS}       # signals under hard mask
    files = sorted(glob.glob(os.path.join(cfg.paths.processed, "eval", "*.nii*")))[:a.max_vols]
    done = 0
    for i, f in enumerate(files):
        try:
            x = [b["image"][0] for b in make_loader([{"image": f}], cfg)][0].to(dev)
            with torch.no_grad():
                prob = torch.softmax(bb(x.unsqueeze(0)), 1)[0]
            cls, conf = int(prob.argmax()), float(prob.max())
            if conf < 0.6:
                continue
            L = supervoxels(x, cfg).to(dev)
            D = L.shape[-1]

            # (A) effective mask audit: middle band of width K, whole-slice region
            for K in KS:
                d0 = max(0, D // 2 - K // 2); d1 = min(D, d0 + K)
                nominal = torch.zeros_like(L, dtype=torch.bool)
                nominal[:, :, d0:d1] = (L[:, :, d0:d1] > 0)
                if nominal.any():
                    M = _feather(nominal, sigma)
                    eff[K].append(float(M[nominal].mean()))

            imp = _panel(bb, x, L, cls, cfg, cavs)
            for K in KS:
                cfg.faith.slice_band = K
                rs = slice_illusion(bb, x, imp, L, bank, cls, cfg, replace_fn=anatomical_replace)
                vh = slice_illusion(bb, x, imp, L, bank, cls, cfg, replace_fn=hard_replace)
                v_s, s_s = _sig(rs); v_h, s_h = _sig(vh)
                soft[K]["v"].append(v_s); soft[K]["s"].append(s_s)
                hard[K]["v"].append(v_h); hard[K]["s"].append(s_h)
            done += 1
            print(f"[{i+1}/{len(files)}] {os.path.basename(f):22s} conf={conf:.2f}", flush=True)
        except Exception as e:
            print(f"[skip] {os.path.basename(f)}: {e}", flush=True)
            continue

    out = os.path.join(cfg.paths.results,
                       f"hardmask_ablation_{os.path.splitext(os.path.basename(a.config))[0]}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["K", "n", "eff_mask_mean_in_Omega", "soft_vol_sig", "soft_slice_sig",
                    "soft_ratio", "hard_vol_sig", "hard_slice_sig", "hard_ratio"])
        print(f"\n== hard-mask ablation ({done} vols, feather_sigma={sigma}) ==")
        print(f"  {'K':>3} {'eff-mask':>9} {'soft ratio':>11} {'hard ratio':>11}")
        for K in KS:
            em = float(np.mean(eff[K]))
            sv, ss = float(np.mean(soft[K]["v"])), float(np.mean(soft[K]["s"]))
            hv, hs = float(np.mean(hard[K]["v"])), float(np.mean(hard[K]["s"]))
            sr = sv / ss if abs(ss) > 1e-9 else float("inf")
            hr = hv / hs if abs(hs) > 1e-9 else float("inf")
            w.writerow([K, done, f"{em:.4f}", f"{sv:.6f}", f"{ss:.6f}", f"{sr:.4f}",
                        f"{hv:.6f}", f"{hs:.6f}", f"{hr:.4f}"])
            print(f"  {K:>3} {em:>9.3f} {sr:>10.1f}x {hr:>10.1f}x")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
