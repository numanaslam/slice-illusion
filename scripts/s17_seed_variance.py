"""Across-seed variance of the random-control signal (reviewer item, round 3).

The slice-wise denominator of the signal ratio is comparable to the sampling noise of
the random baseline's mean, so point ratios are realisation-dependent. This script
quantifies that directly: it evaluates the single-dataset BraTS protocol (concept panel,
gate tau=0.6, context-nearest-neighbour) with R independent random-control draws per
volume and reports the per-seed volumetric and slice-wise signals, their across-seed
standard deviation, and the per-seed ratios.

Seed 0 uses the legacy np.random stream exactly as s04 does, so its numbers reproduce
the paper's headline; seeds 1..R-1 come from independent per-volume Generators and do
not perturb that stream. Writes results/seed_variance_<config>.csv and touches none of
the standard artifacts.

    python -m scripts.s17_seed_variance --config configs/rtx3070.yaml            # plain CNN
    python -m scripts.s17_seed_variance --config configs/brats_agg.yaml          # residual
"""
import argparse
import glob
import os
from collections import defaultdict
import numpy as np
import torch

from vcf3d.config import load_config
from vcf3d.models.backbone import load_backbone
from vcf3d.io.datasets import make_loader
from vcf3d.perturb.tissue_bank import build_tissue_bank
from vcf3d.utils.supervoxels import supervoxels
from vcf3d.utils.gradcam import grad_cam_3d, supervoxel_importance
from vcf3d.concept.tcav import tcav_saliency
from vcf3d.faithfulness.slice_illusion import slice_illusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rtx3070.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max_vols", type=int, default=10**9)
    a = ap.parse_args()
    cfg = load_config(a.config)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.compute.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    dev = cfg.compute.device
    bb = load_backbone(cfg)

    held = [b["image"][0] for b in make_loader(
        [{"image": f} for f in glob.glob(os.path.join(cfg.paths.processed, "heldout", "*.nii*"))], cfg)]
    bank = build_tissue_bank(held, None, cfg) if held else None
    cav_path = os.path.join(cfg.paths.results, "cavs.npy")
    cavs = np.load(cav_path, allow_pickle=True).item() if os.path.exists(cav_path) else {}

    files = sorted(glob.glob(os.path.join(cfg.paths.processed, "eval", "*.nii*")))[:a.max_vols]
    vacc, sacc = defaultdict(list), defaultdict(list)
    done = 0
    for i, f in enumerate(files):
        try:
            x = [b["image"][0] for b in make_loader([{"image": f}], cfg)][0].to(dev)
            with torch.no_grad():
                prob = torch.softmax(bb(x.unsqueeze(0)), 1)[0]
            cls_v, conf = int(prob.argmax()), float(prob.max())
            if conf < 0.6:
                print(f"[skip lowconf] {os.path.basename(f)} p={conf:.2f}", flush=True)
                continue
            L = supervoxels(x, cfg).to(dev)
            S = int(L.max())
            imp = {"gradcam": supervoxel_importance(grad_cam_3d(bb, x.unsqueeze(0), cls_v, cfg), L)}
            for cname, cv in cavs.items():
                imp[f"tcav_{cname}"] = supervoxel_importance(
                    tcav_saliency(bb, x.unsqueeze(0), cv, cls_v, cfg), L)
            imp["random"] = np.random.rand(max(S, 1)).astype(np.float32)   # legacy stream
            for si in range(1, a.seeds):                                   # independent draws
                g = np.random.default_rng(cfg.seed + 7919 * si + 31 * i)
                imp[f"random_s{si}"] = g.random(max(S, 1)).astype(np.float32)
            rep = slice_illusion(bb, x, imp, L, bank, cls_v, cfg)
        except Exception as e:
            print(f"[skip] {os.path.basename(f)}: {e}", flush=True)
            continue
        for m, va, sa in zip(rep["methods"], rep["vol_auc"], rep["slice_auc"]):
            vacc[m].append(float(va))
            sacc[m].append(float(sa))
        done += 1
        print(f"[{i+1}/{len(files)}] {os.path.basename(f):22s} conf={conf:.2f}", flush=True)

    real = [m for m in vacc if not m.startswith("random")]
    vbest = min(float(np.mean(vacc[m])) for m in real)
    sbest = min(float(np.mean(sacc[m])) for m in real)
    seeds = ["random"] + [f"random_s{si}" for si in range(1, a.seeds)]
    rows = []
    for s in seeds:
        vsig = float(np.mean(vacc[s])) - vbest
        ssig = float(np.mean(sacc[s])) - sbest
        rows.append((s, vsig, ssig, vsig / ssig if abs(ssig) > 1e-9 else float("inf")))
    tag = os.path.splitext(os.path.basename(a.config))[0]
    out = os.path.join(cfg.paths.results, f"seed_variance_{tag}.csv")
    with open(out, "w") as fh:
        fh.write("seed,vol_signal,slice_signal,ratio\n")
        for s, v, sl, r in rows:
            fh.write(f"{s},{v:.6f},{sl:.6f},{r:.4f}\n")
    vs = np.array([r[1] for r in rows]); ss = np.array([r[2] for r in rows])
    print(f"\n== seed variance ({done} vols, {a.seeds} random seeds) ==")
    for s, v, sl, r in rows:
        print(f"  {s:10s} vol {v:+.4f}  slice {sl:+.4f}  ratio {r:.1f}x")
    print(f"  vol signal  mean {vs.mean():.4f}  across-seed SD {vs.std(ddof=1):.4f}")
    print(f"  slice signal mean {ss.mean():.4f}  across-seed SD {ss.std(ddof=1):.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
