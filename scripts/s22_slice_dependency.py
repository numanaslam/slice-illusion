"""Slice-dependency analysis on the real models (reviewer Experiment B).

Directly measures whether a 3D classifier distributes evidence across depth, by comparing
the prediction drop from three interventions of matched or larger extent:

  dp_important : remove the single most-important axial slice (by Grad-CAM slice mass)
  dp_random    : remove a random axial slice
  dp_volumetric: remove a 3D region of the SAME voxel count as one slice, by importance

If dp_important and dp_random are both small while dp_volumetric is large, the model's
evidence is distributed: no single slice carries it, but a compact 3D region does. That is
the mechanism behind the slice illusion, shown on the prediction directly rather than
through the faithfulness signal.

    python -m scripts.s22_slice_dependency --config configs/brats_agg.yaml   # residual
    python -m scripts.s22_slice_dependency --config configs/rtx3070.yaml     # plain CNN
Output: experiments/results/slice_dependency_<config>.csv
"""
import argparse
import csv
import glob
import os
import numpy as np
import torch

from vcf3d.config import load_config
from vcf3d.models.backbone import load_backbone
from vcf3d.io.datasets import make_loader
from vcf3d.perturb.tissue_bank import build_tissue_bank
from vcf3d.perturb.anatomical_replace import anatomical_replace
from vcf3d.utils.supervoxels import supervoxels
from vcf3d.utils.gradcam import grad_cam_3d


def prob(bb, x, cls, cfg):
    with torch.no_grad():
        return float(torch.softmax(bb(x.unsqueeze(0)), 1)[0, cls])


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
    held = [b["image"][0] for b in make_loader(
        [{"image": f} for f in glob.glob(os.path.join(cfg.paths.processed, "heldout", "*.nii*"))], cfg)]
    bank = build_tissue_bank(held, None, cfg) if held else None

    rows = []
    files = sorted(glob.glob(os.path.join(cfg.paths.processed, "eval", "*.nii*")))[:a.max_vols]
    for i, f in enumerate(files):
        try:
            x = [b["image"][0] for b in make_loader([{"image": f}], cfg)][0].to(dev)
            p0 = torch.softmax(bb(x.unsqueeze(0)), 1)[0]
            cls, conf = int(p0.argmax()), float(p0.max())
            if conf < 0.6:
                continue
            L = supervoxels(x, cfg).to(dev)
            Dd = L.shape[-1]
            cam = grad_cam_3d(bb, x.unsqueeze(0), cls, cfg)
            cam = np.asarray(cam, dtype=np.float32)
            while cam.ndim > 3:
                cam = cam[0]
            # slice importance = Grad-CAM mass per axial slice (within brain)
            brain = (L > 0).cpu().numpy()
            slice_mass = np.array([cam[:, :, d][brain[:, :, d]].sum() if brain[:, :, d].any() else 0.0
                                   for d in range(Dd)])
            d_imp = int(slice_mass.argmax())
            d_rnd = int(np.random.default_rng(i).integers(0, Dd))

            def drop_slice(d):
                m = torch.zeros_like(L, dtype=torch.bool); m[:, :, d] = (L[:, :, d] > 0)
                return conf - prob(bb, anatomical_replace(x.clone(), m, bank, cfg), cls, cfg), int(m.sum())

            dp_imp, nvox = drop_slice(d_imp)
            dp_rnd, _ = drop_slice(d_rnd)
            # volumetric region of the SAME voxel count, chosen by supervoxel importance
            svids = torch.unique(L[L > 0])
            sv_imp = []
            camt = torch.as_tensor(cam, device=dev)
            for s in svids:
                mm = (L == s)
                sv_imp.append((float(camt[mm].mean()), int(s), int(mm.sum())))
            sv_imp.sort(reverse=True)
            picked, acc = [], 0
            for _, sid, sz in sv_imp:
                picked.append(sid); acc += sz
                if acc >= nvox:
                    break
            mvol = torch.isin(L, torch.as_tensor(picked, device=dev))
            dp_vol = conf - prob(bb, anatomical_replace(x.clone(), mvol, bank, cfg), cls, cfg)
            rows.append((os.path.basename(f), conf, dp_imp, dp_rnd, dp_vol, nvox, int(mvol.sum())))
            print(f"[{i+1}/{len(files)}] {os.path.basename(f):20s} dp_imp={dp_imp:+.3f} "
                  f"dp_rnd={dp_rnd:+.3f} dp_vol={dp_vol:+.3f}", flush=True)
        except Exception as e:
            print(f"[skip] {os.path.basename(f)}: {e}", flush=True)
            continue

    out = os.path.join(cfg.paths.results,
                       f"slice_dependency_{os.path.splitext(os.path.basename(a.config))[0]}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "conf", "dp_important_slice", "dp_random_slice", "dp_volumetric", "n_slice_vox", "n_vol_vox"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.6f}", f"{r[3]:.6f}", f"{r[4]:.6f}", r[5], r[6]])
    A = np.array([[r[2], r[3], r[4]] for r in rows])
    print(f"\n== slice dependency ({len(rows)} vols) ==")
    print(f"  mean dp important-slice = {A[:,0].mean():+.4f}")
    print(f"  mean dp random-slice    = {A[:,1].mean():+.4f}")
    print(f"  mean dp volumetric (=1 slice budget) = {A[:,2].mean():+.4f}")
    print(f"  volumetric/important-slice = {A[:,2].mean()/A[:,0].mean():.1f}x" if A[:,0].mean() else "")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
