"""Render a REAL Grad-CAM-on-BraTS overlay for the method figure (fig_method.svg).

Saves docs/fig_method_overlay.png — a clean square slice with the heatmap overlaid,
which fig_method.svg references. Run after a trained backbone exists.

    python -m scripts.s08_method_overlay --config configs\\rtx3070.yaml
"""
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="docs/fig_method_overlay.png")
    ap.add_argument("--channel", type=int, default=0, help="MRI channel to display")
    a, _ = ap.parse_known_args()

    import torch
    from vcf3d.config import load_config
    from vcf3d.models.backbone import load_backbone
    from vcf3d.io.datasets import make_loader
    from vcf3d.utils.gradcam import grad_cam_3d

    cfg = load_config(a.config)
    bb = load_backbone(cfg)
    files = sorted(glob.glob(os.path.join(cfg.paths.processed, "eval", "*.nii*")))
    if not files:
        raise SystemExit("no eval volumes — run s01/s02 first")
    x = next(iter(make_loader([{"image": files[0]}], cfg)))["image"][0].to(cfg.compute.device)
    cls = int(torch.softmax(bb(x.unsqueeze(0)), 1).argmax())

    cam = grad_cam_3d(bb, x.unsqueeze(0), cls, cfg)               # (H,W,D) in [0,1]
    fg = (x[a.channel] != 0).cpu().numpy()
    z = int(np.argmax(fg.reshape(-1, fg.shape[-1]).sum(0)))       # slice with most anatomy
    sl = np.rot90(x[a.channel, :, :, z].cpu().numpy())
    hm = np.rot90(cam[:, :, z])

    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    ax.imshow(sl, cmap="gray")
    ax.imshow(np.ma.masked_where(hm < 0.3, hm), cmap="inferno", alpha=0.6, vmin=0, vmax=1)
    ax.axis("off")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"wrote {a.out}  (slice z={z}, class {cls})")


if __name__ == "__main__":
    main()
