"""Matched-absolute-budget deletion: isolate spatial support from intervention magnitude.

Reviewer's decisive control. The band-K sweep varies spatial support and absolute
extent together (a wider band exposes more tissue). This script holds the ABSOLUTE
number of deleted voxels fixed and varies only how those voxels are arranged in 3D:

  Arrangement COMPACT   : delete voxels within a SINGLE axial slice (the most-important
                          slice), equal-volume steps, up to budget B.
  Arrangement DISTRIBUTED: delete whole 3D supervoxels ranked by importance, equal-volume
                          steps, up to the SAME budget B (these span the depth).

B is the nonzero-supervoxel voxel count of the chosen slice, so both arrangements delete
the same absolute amount of tissue; only the spatial support differs. If the faithfulness
signal is larger for DISTRIBUTED than for COMPACT at equal B, spatial support drives
discriminability independently of intervention magnitude, and the band-K recovery is not
a mere consequence of removing more tissue.

    python -m scripts.s19_matched_budget --config configs/brats_agg.yaml      # plain CNN (headline model)
    python -m scripts.s19_matched_budget --config configs/rtx3070.yaml        # residual, as a check
Output: experiments/results/matched_budget_<config>.csv  (per-volume signals + paired test)
"""
import argparse
import csv
import glob
import os
from collections import defaultdict
import numpy as np
import torch
from scipy.stats import wilcoxon

from vcf3d.config import load_config
from vcf3d.models.backbone import load_backbone
from vcf3d.io.datasets import make_loader
from vcf3d.perturb.tissue_bank import build_tissue_bank
from vcf3d.utils.supervoxels import supervoxels
from vcf3d.utils.gradcam import grad_cam_3d, supervoxel_importance
from vcf3d.concept.tcav import tcav_saliency
from vcf3d.faithfulness.deletion_insertion import _class_prob, equal_volume_chunks
from vcf3d.perturb.anatomical_replace import anatomical_replace


def _budgeted_auc(bb, x, cls, cfg, bank, order_local, sizes_local, mask_builder, budget, steps):
    """Deletion AUC over a fixed voxel BUDGET, equal-volume steps, given an ordering.

    order_local indexes into the eligible unit list (descending importance); sizes_local
    are those units' voxel counts; mask_builder(unit_ids)->bool mask on the full grid.
    Units are consumed in importance order until cumulative volume reaches budget.
    """
    cum = np.cumsum(sizes_local[order_local])
    keep = order_local[cum <= budget]
    if len(keep) < 2:
        keep = order_local[:max(2, int(np.searchsorted(cum, budget)) + 1)]
    chunks = equal_volume_chunks(keep, sizes_local, steps)
    xd = x.clone()
    p = [_class_prob(bb, xd, cls, cfg)]
    for ch in chunks:
        xd = anatomical_replace(xd, mask_builder(ch), bank, cfg)
        p.append(_class_prob(bb, xd, cls, cfg))
    frac = np.linspace(0, 1, len(p))
    return float(np.trapz(p, frac))


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
    steps = int(cfg.faith.steps)

    held = [b["image"][0] for b in make_loader(
        [{"image": f} for f in glob.glob(os.path.join(cfg.paths.processed, "heldout", "*.nii*"))], cfg)]
    bank = build_tissue_bank(held, None, cfg) if held else None
    cav_path = os.path.join(cfg.paths.results, "cavs.npy")
    cavs = np.load(cav_path, allow_pickle=True).item() if os.path.exists(cav_path) else {}

    sig = {"compact": defaultdict(list), "distributed": defaultdict(list)}
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
            Ln = L.cpu().numpy()
            S = int(Ln.max())
            sizes3d = np.bincount(Ln.ravel(), minlength=S + 1)[1:S + 1].astype(float)

            imp = {"gradcam": supervoxel_importance(grad_cam_3d(bb, x.unsqueeze(0), cls, cfg), L)}
            for cn, cv in cavs.items():
                imp[f"tcav_{cn}"] = supervoxel_importance(tcav_saliency(bb, x.unsqueeze(0), cv, cls, cfg), L)
            imp["random"] = np.random.rand(max(S, 1)).astype(np.float32)
            real = [m for m in imp if m != "random"]

            # --- choose the most-important axial slice (generous to the compact arm) ---
            D = Ln.shape[-1]
            slice_imp = np.zeros(D)
            gc = imp["gradcam"]
            for d in range(D):
                labs = np.unique(Ln[:, :, d]); labs = labs[labs > 0]
                slice_imp[d] = gc[labs - 1].sum() if len(labs) else 0.0
            d0 = int(np.argmax(slice_imp))

            # COMPACT: units = per-slice supervoxel fragments in slice d0
            Lslice = Ln[:, :, d0]
            labs = np.unique(Lslice); labs = labs[labs > 0]
            if len(labs) < 3:
                continue
            sizes_c = np.array([(Lslice == l).sum() for l in labs], dtype=float)
            budget = float(sizes_c.sum())                       # one slice worth of voxels

            def mk_compact(unit_idx, labs=labs, d0=d0, L=L):
                sel = torch.as_tensor(labs[unit_idx], device=L.device)
                m = torch.zeros_like(L, dtype=torch.bool)
                m[:, :, d0] = torch.isin(L[:, :, d0], sel)
                return m

            # DISTRIBUTED: units = whole 3D supervoxels, capped at the same budget
            def mk_dist(unit_ids, L=L):
                return torch.isin(L, torch.as_tensor(unit_ids + 1, device=L.device))

            for m in imp:
                imp_c = imp[m][labs - 1]
                auc_c = _budgeted_auc(bb, x, cls, cfg, bank,
                                      np.argsort(-imp_c), sizes_c, mk_compact, budget, steps)
                sig["compact"][m].append(auc_c)
                auc_d = _budgeted_auc(bb, x, cls, cfg, bank,
                                      np.argsort(-imp[m]), sizes3d, mk_dist, budget, steps)
                sig["distributed"][m].append(auc_d)
            done += 1
            print(f"[{i+1}/{len(files)}] {os.path.basename(f):22s} conf={conf:.2f} budget={int(budget)} vox", flush=True)
        except Exception as e:
            print(f"[skip] {os.path.basename(f)}: {e}", flush=True)
            continue

    real = [m for m in sig["compact"] if m != "random"]
    out = os.path.join(cfg.paths.results, f"matched_budget_{os.path.splitext(os.path.basename(a.config))[0]}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arrangement", "n", "vol_random_auc", "best_real_auc", "signal"])
        rows = {}
        for arr in ("compact", "distributed"):
            rmean = float(np.mean(sig[arr]["random"]))
            bmean = min(float(np.mean(sig[arr][m])) for m in real)
            rows[arr] = rmean - bmean
            w.writerow([arr, done, f"{rmean:.6f}", f"{bmean:.6f}", f"{rmean - bmean:.6f}"])
    # paired per-volume signal test (distributed vs compact)
    rc = np.array(sig["compact"]["random"]); bc = np.min(np.stack([sig["compact"][m] for m in real]), 0)
    rd = np.array(sig["distributed"]["random"]); bd = np.min(np.stack([sig["distributed"][m] for m in real]), 0)
    sc, sd = rc - bc, rd - bd
    try:
        _, p = wilcoxon(sd, sc)
    except ValueError:
        p = float("nan")
    print(f"\n== matched-budget ({done} vols, budget = one axial slice) ==")
    print(f"  COMPACT     signal = {rows['compact']:+.4f}")
    print(f"  DISTRIBUTED signal = {rows['distributed']:+.4f}")
    print(f"  ratio distributed/compact = {rows['distributed']/rows['compact']:.2f}x"
          if abs(rows['compact']) > 1e-9 else "  compact signal ~0")
    print(f"  paired Wilcoxon (distributed vs compact signal, n={done}): p={p:.3e}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
