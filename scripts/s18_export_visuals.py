"""Export real BraTS slices + attribution heatmaps for the paper's visual figures.

Picks the first N confident eval volumes, computes Grad-CAM, one TCAV concept, and the
random-control heatmap (per-supervoxel importance mapped back to voxels), and saves
axial mid-slices to a small .npz. No standard artifact is touched. Runs in minutes.

    python -m scripts.s18_export_visuals --config configs/rtx3070.yaml
Output: experiments/results/visual_export.npz  (copy back to the laptop)
"""
import argparse
import glob
import os
import numpy as np
import torch

from vcf3d.config import load_config
from vcf3d.models.backbone import load_backbone
from vcf3d.io.datasets import make_loader
from vcf3d.utils.supervoxels import supervoxels
from vcf3d.utils.gradcam import grad_cam_3d, supervoxel_importance
from vcf3d.concept.tcav import tcav_saliency


def _vol(a):
    """(H,W,D) float numpy from a tensor or ndarray of shape (...,H,W,D)."""
    if hasattr(a, "detach"):
        a = a.detach().float().cpu().numpy()
    a = np.asarray(a, dtype=np.float32)
    while a.ndim > 3:
        a = a[0]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rtx3070.yaml")
    ap.add_argument("--n_vols", type=int, default=3)
    a = ap.parse_args()
    cfg = load_config(a.config)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = cfg.compute.device
    bb = load_backbone(cfg)

    cav_path = os.path.join(cfg.paths.results, "cavs.npy")
    cavs = np.load(cav_path, allow_pickle=True).item() if os.path.exists(cav_path) else {}
    cname = sorted(cavs)[0] if cavs else None

    out = {}
    kept = 0
    for f in sorted(glob.glob(os.path.join(cfg.paths.processed, "eval", "*.nii*"))):
        if kept >= a.n_vols:
            break
        x = [b["image"][0] for b in make_loader([{"image": f}], cfg)][0].to(dev)
        with torch.no_grad():
            prob = torch.softmax(bb(x.unsqueeze(0)), 1)[0]
        cls_v, conf = int(prob.argmax()), float(prob.max())
        if conf < 0.6:
            continue
        L = supervoxels(x, cfg).to(dev)
        Ln = L.cpu().numpy()
        S = int(Ln.max())

        cam = _vol(grad_cam_3d(bb, x.unsqueeze(0), cls_v, cfg))
        tcv = _vol(tcav_saliency(bb, x.unsqueeze(0), cavs[cname], cls_v, cfg)) if cname else None
        rnd_imp = np.random.rand(max(S, 1)).astype(np.float32)
        rnd = np.where(Ln > 0, rnd_imp[np.clip(Ln - 1, 0, S - 1)], 0.0)

        xin = x.cpu().numpy()                      # (C,H,W,D); FLAIR is the last channel
        anat = xin[-1] if xin.ndim == 4 else xin
        D = anat.shape[-1]
        sls = [D // 4, D // 2, 3 * D // 4]
        key = os.path.basename(f).split(".")[0]
        out[f"{key}_anat"] = np.stack([anat[:, :, d] for d in sls]).astype(np.float32)
        out[f"{key}_gradcam"] = np.stack([cam[:, :, d] for d in sls]).astype(np.float32)
        if tcv is not None:
            out[f"{key}_tcav"] = np.stack([tcv[:, :, d] for d in sls]).astype(np.float32)
        out[f"{key}_random"] = np.stack([rnd[:, :, d] for d in sls]).astype(np.float32)
        out[f"{key}_svox"] = np.stack([Ln[:, :, d] for d in sls]).astype(np.int32)
        out[f"{key}_meta"] = np.array([conf, cls_v, D], dtype=np.float32)
        kept += 1
        print(f"[export] {key} conf={conf:.2f} slices={sls} "
              f"tcav={'yes (' + cname + ')' if tcv is not None else 'no'}", flush=True)

    dst = os.path.join(cfg.paths.results, "visual_export.npz")
    np.savez_compressed(dst, **out)
    print(f"wrote {dst} ({os.path.getsize(dst)/1e6:.1f} MB, {kept} volumes)")


if __name__ == "__main__":
    main()
