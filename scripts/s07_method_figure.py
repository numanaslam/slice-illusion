"""Realistic method figure built from REAL pipeline data (run on the server).

Panels use actual images from one eval volume: the input slice, Grad-CAM and
3D-TCAV concept overlays, 3D supervoxels, the anatomically-plausible perturbation
R (before/after), plus the real volumetric deletion curves and faithfulness-signal
bars. Run AFTER s03 + s04 (needs cavs.npy, deletion_curves_K1.npy, per_method_auc_K1.csv).

    python -m scripts.s07_method_figure --config configs\\rtx3070.yaml
Output: experiments/results/figures/fig_method_real.{pdf,png}
"""
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OKABE = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73", "grey": "#6b7280"}


def _style():
    plt.rcParams.update({
        "savefig.dpi": 300, "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10.5, "figure.constrained_layout.use": True})


def _img(ax, base, title, overlay=None, cmap_ov="inferno"):
    ax.imshow(base, cmap="gray", interpolation="nearest")
    if overlay is not None:
        m = np.ma.masked_where(overlay < 0.25, overlay)
        ax.imshow(m, cmap=cmap_ov, alpha=0.55, interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title, fontweight="600"); ax.set_xticks([]); ax.set_yticks([])


def compose(out, sl, gcam, tcav, svox, before, after, curves, permethod):
    _style()
    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
    ax = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]

    # Row 1 — model + volumetric explanation
    _img(ax[0][0], sl, "(a) Input axial slice")
    ax[0][0].set_ylabel("STEP 1  |  explain", fontsize=11, fontweight="700",
                        color="#2c3e50", labelpad=8)
    _img(ax[0][1], sl, "(b) Grad-CAM", overlay=gcam)
    _img(ax[0][2], sl, "(c) 3D-TCAV (concept)", overlay=tcav, cmap_ov="viridis")
    ax[0][3].imshow(sl, cmap="gray"); ax[0][3].imshow(svox, cmap="nipy_spectral", alpha=0.45)
    ax[0][3].set_title("(d) 3D supervoxels", fontweight="600")
    ax[0][3].set_xticks([]); ax[0][3].set_yticks([])

    # Row 2 — perturbation + faithfulness
    _img(ax[1][0], before, "(e) Original region")
    ax[1][0].set_ylabel("STEP 2  |  evaluate", fontsize=11, fontweight="700",
                        color="#2c3e50", labelpad=8)
    _img(ax[1][1], after, "(f) After R (context-NN)")

    axc = ax[1][2]
    if curves is not None:
        g = curves["grid"]
        order = [m for m in ["gradcam", "tcav_enhancing", "tcav_non_enhancing",
                             "tcav_edema", "tcav_dense", "tcav_core", "tcav_rim",
                             "random"] if m in curves["vol"]]
        tcav_cycle = [OKABE["green"], "#E69F00", "#56B4E9", "#CC79A7", "#999933"]; ti = 0
        for m in order:
            if m == "gradcam":
                c, ls = OKABE["blue"], "-"
            elif m == "random":
                c, ls = OKABE["grey"], (0, (4, 2))
            else:
                c, ls = tcav_cycle[ti % len(tcav_cycle)], "-"; ti += 1
            axc.plot(g, curves["vol"][m], ls=ls, color=c, lw=1.6,
                     label=m.replace("tcav_", "TCAV:").replace("gradcam", "Grad-CAM").replace("random", "Random"))
        axc.legend(fontsize=6.5, frameon=False, loc="upper right")
    axc.set_title("(g) Volumetric deletion curves", fontweight="600")
    axc.set_xlabel("fraction removed"); axc.set_ylabel("class prob"); axc.set_xlim(0, 1)
    for s in ("top", "right"): axc.spines[s].set_visible(False)

    axb = ax[1][3]
    if permethod is not None:
        real = [m for m in permethod if m != "random"]
        vsig = permethod["random"][0] - min(permethod[m][0] for m in real)
        ssig = permethod["random"][1] - min(permethod[m][1] for m in real)
        axb.barh([1, 0], [vsig, ssig], color=[OKABE["blue"], OKABE["vermillion"]], height=0.55)
        axb.set_yticks([1, 0]); axb.set_yticklabels(["volumetric", "slice-wise"])
        axb.set_xlabel("faithfulness signal")
        axb.text(vsig, 1, f" {vsig/max(ssig,1e-9):.0f}×", va="center", fontsize=9,
                 fontweight="700", color=OKABE["blue"])
    axb.set_title("(h) Faithfulness signal", fontweight="600")
    for s in ("top", "right"): axb.spines[s].set_visible(False)

    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}.pdf / .png")


def _best_slice(mask3d):
    return int(np.argmax(mask3d.reshape(-1, mask3d.shape[-1]).sum(0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args, _ = ap.parse_known_args()

    # heavy imports here so the plotting (compose) is testable without torch
    import torch
    from vcf3d.config import load_config
    from vcf3d.models.backbone import load_backbone
    from vcf3d.io.datasets import make_loader
    from vcf3d.utils.supervoxels import supervoxels
    from vcf3d.utils.gradcam import grad_cam_3d
    from vcf3d.concept.tcav import tcav_saliency
    from vcf3d.perturb.anatomical_replace import anatomical_replace

    cfg = load_config(args.config)
    bb = load_backbone(cfg)
    res = cfg.paths.results
    files = sorted(glob.glob(os.path.join(cfg.paths.processed, "eval", "*.nii*")))
    x = make_loader([{"image": files[0]}], cfg).__iter__().__next__()["image"][0]
    x = x.to(cfg.compute.device)
    cls = int(torch.softmax(bb(x.unsqueeze(0)), 1).argmax())

    fg = (x[0] != 0).cpu().numpy()
    z = _best_slice(fg)
    sl = x[0, :, :, z].cpu().numpy()

    gcam = grad_cam_3d(bb, x.unsqueeze(0), cls, cfg)[:, :, z]
    cav_path = os.path.join(res, "cavs.npy")
    if os.path.exists(cav_path):
        cav = next(iter(np.load(cav_path, allow_pickle=True).item().values()))
        tcav = tcav_saliency(bb, x.unsqueeze(0), cav, cls, cfg)[:, :, z]
    else:
        print("  [warn] cavs.npy missing (run s03) — TCAV panel blank")
        tcav = np.zeros_like(sl)

    L = supervoxels(x, cfg).to(cfg.compute.device)
    svox = L[:, :, z].cpu().numpy().astype(float); svox[svox == 0] = np.nan

    # perturb the most Grad-CAM-important supervoxel in this slice (nice demo)
    yy, xx = np.unravel_index(int(np.nanargmax(np.nan_to_num(gcam))), gcam.shape)
    lab = int(L[yy, xx, z].item())
    mask = (L == lab) if lab > 0 else (L == int(L[:, :, z].max()))
    xp = anatomical_replace(x.clone(), mask, None if cfg.faith.surrogate in
                            ("zero", "mean") else _bank(cfg), cfg)
    before = sl.copy(); after = xp[0, :, :, z].cpu().numpy()

    curves = None
    cp = os.path.join(res, "deletion_curves_K1.npy")
    if os.path.exists(cp):
        curves = np.load(cp, allow_pickle=True).item()
    permethod = None
    pp = os.path.join(res, "per_method_auc_K1.csv")
    if os.path.exists(pp):
        rows = [l.strip().split(",") for l in open(pp)][1:]
        permethod = {r[0]: (float(r[1]), float(r[2])) for r in rows}

    outdir = os.path.join(res, "figures"); os.makedirs(outdir, exist_ok=True)
    compose(os.path.join(outdir, "fig_method_real"), sl, gcam, tcav, svox,
            before, after, curves, permethod)


def _bank(cfg):
    from vcf3d.io.datasets import make_loader
    from vcf3d.perturb.tissue_bank import build_tissue_bank
    import glob as _g
    held = [b["image"][0] for b in make_loader(
        [{"image": f} for f in _g.glob(os.path.join(cfg.paths.processed, "heldout", "*.nii*"))[:8]], cfg)]
    return build_tissue_bank(held, None, cfg) if held else None


if __name__ == "__main__":
    main()
