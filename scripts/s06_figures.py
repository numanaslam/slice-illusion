"""Publication figures for the slice-illusion result.

Reads the artifacts s04 writes:
  experiments/results/summary.csv                (K-sweep -> Fig 1)
  experiments/results/per_method_auc_K1.csv      (per-method AUC -> Fig 2)
  experiments/results/deletion_curves_K1.npy     (deletion curves -> Fig 3)

Outputs vector PDF + 300-dpi PNG to experiments/results/figures/.

    python -m scripts.s06_figures --config configs\\rtx3070.yaml
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                              # headless server
import matplotlib.pyplot as plt

from vcf3d.config import load_config

# ---- house style: colorblind-safe (Okabe-Ito), serif, minimal ---------------
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "sky": "#56B4E9",
         "green": "#009E73", "vermillion": "#D55E00", "purple": "#CC79A7",
         "grey": "#444444"}
PROTO = {"vol": OKABE["blue"], "slice": OKABE["vermillion"]}
METHOD_COLOR = {"gradcam": OKABE["blue"], "tcav_enhancing": OKABE["orange"],
                "tcav_non_enhancing": OKABE["sky"], "tcav_edema": OKABE["green"],
                "random": OKABE["grey"]}
METHOD_NAME = {"gradcam": "Grad-CAM", "tcav_enhancing": "TCAV (enh. tumor)",
               "tcav_non_enhancing": "TCAV (non-enh.)", "tcav_edema": "TCAV (edema)",
               "random": "Random"}
ORDER = ["gradcam", "tcav_enhancing", "tcav_non_enhancing", "tcav_edema", "random"]


def set_style():
    plt.rcParams.update({
        "savefig.dpi": 300, "figure.dpi": 150,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 1.7, "lines.markersize": 5,
        "legend.frameon": False, "figure.constrained_layout.use": True,
    })


def _save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def _read_csv(path):
    if not os.path.exists(path):
        return None
    rows = [ln.strip().split(",") for ln in open(path) if ln.strip()]
    hdr, data = rows[0], rows[1:]
    return hdr, data


# ---- Figure 1: K-sweep (discriminability needs 3D context) -------------------
def fig_ksweep(res, outdir):
    parsed = _read_csv(os.path.join(res, "summary.csv"))
    if parsed is None:
        print("  [skip Fig1] summary.csv not found"); return
    hdr, data = parsed
    col = {c: i for i, c in enumerate(hdr)}
    by_k = {}                                       # dedup: last row per K wins
    for r in data:
        by_k[int(r[col["K"]])] = r
    ks = sorted(by_k)
    K = np.array(ks, float)
    sig_s = np.array([float(by_k[k][col["slice_signal"]]) for k in ks])
    drop_s = np.array([float(by_k[k][col["slice_drop"]]) for k in ks])
    sig_v = float(by_k[ks[0]][col["vol_signal"]])
    drop_v = float(by_k[ks[0]][col["vol_drop"]])

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for a, ys, yv, ylab, tag in (
            (ax[0], sig_s, sig_v, "Faithfulness signal", "(a)"),
            (ax[1], drop_s, drop_v, "Mean prediction drop", "(b)")):
        a.axhline(yv, ls=(0, (5, 2)), lw=1.4, color=PROTO["vol"],
                  label="volumetric (3D)")
        a.plot(K, ys, "-o", color=PROTO["slice"], mfc="white", mew=1.4,
               label="slice-wise (band $K$)")
        a.set_xscale("log", base=2); a.set_xticks(ks); a.set_xticklabels(ks)
        a.set_xlabel("slice-band width $K$ (axial slices)")
        a.set_ylabel(ylab); a.set_ylim(bottom=0)
        a.margins(x=0.06)
        a.text(-0.16, 1.02, tag, transform=a.transAxes, fontweight="bold", fontsize=11)
    ax[0].legend(loc="upper left")
    _save(fig, outdir, "fig1_ksweep")


# ---- Figure 2: per-method deletion AUC (Cleveland dot plot) ------------------
def fig_permethod(res, outdir):
    parsed = _read_csv(os.path.join(res, "per_method_auc_K1.csv")) \
        or _read_csv(os.path.join(res, "per_method_auc.csv"))
    if parsed is None:
        print("  [skip Fig2] per_method_auc_K1.csv not found"); return
    hdr, data = parsed
    d = {r[0]: (float(r[1]), float(r[2])) for r in data}
    methods = [m for m in ORDER if m in d] + [m for m in d if m not in ORDER]
    methods = sorted(methods, key=lambda m: d[m][0])         # by volumetric AUC
    y = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for yi, m in zip(y, methods):
        v, s = d[m]
        ax.plot([v, s], [yi, yi], color="0.75", lw=1.0, zorder=1)
    ax.scatter([d[m][0] for m in methods], y, s=42, color=PROTO["vol"],
               zorder=3, label="volumetric (3D)")
    ax.scatter([d[m][1] for m in methods], y, s=42, color=PROTO["slice"],
               marker="s", zorder=3, label="slice-wise (2D)")
    ax.set_yticks(y); ax.set_yticklabels([METHOD_NAME.get(m, m) for m in methods])
    ax.set_xlabel("deletion AUC  (lower = more faithful)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    # annotate the discriminability gap under each protocol
    va = [d[m][0] for m in methods]; sa = [d[m][1] for m in methods]
    ax.annotate("", xy=(min(va), -0.6), xytext=(max(va), -0.6),
                arrowprops=dict(arrowstyle="<->", color=PROTO["vol"], lw=1.1))
    ax.text(np.mean(va), -0.95, f"spread {max(va)-min(va):.2f}",
            color=PROTO["vol"], ha="center", va="top", fontsize=7.5)
    ax.set_ylim(-1.3, len(methods) - 0.4)
    _save(fig, outdir, "fig2_permethod_auc")


# ---- Figure 3: mean deletion curves (the fan-out) ---------------------------
def fig_curves(res, outdir):
    path = os.path.join(res, "deletion_curves_K1.npy")
    if not os.path.exists(path):
        path = os.path.join(res, "deletion_curves.npy")
    if not os.path.exists(path):
        print("  [skip Fig3] deletion_curves_K1.npy not found"); return
    dc = np.load(path, allow_pickle=True).item()
    grid = dc["grid"]
    methods = [m for m in ORDER if m in dc["vol"]] + \
              [m for m in dc["vol"] if m not in ORDER]

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for a, key, title, tag in ((ax[0], "vol", "Volumetric (3D) deletion", "(a)"),
                               (ax[1], "slice", "Slice-wise (2D) deletion", "(b)")):
        for m in methods:
            ls = (0, (4, 2)) if m == "random" else "-"
            a.plot(grid, dc[key][m], ls=ls, color=METHOD_COLOR.get(m, "0.3"),
                   label=METHOD_NAME.get(m, m))
        a.set_xlabel("fraction of volume removed")
        a.set_title(title, fontsize=9.5)
        a.set_xlim(0, 1)
        a.text(-0.02, 1.04, tag, transform=a.transAxes, fontweight="bold", fontsize=11)
    ax[0].set_ylabel("predicted class probability")
    ax[1].legend(loc="upper right", ncol=1)
    _save(fig, outdir, "fig3_deletion_curves")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(ap.parse_known_args()[0].config)
    res = cfg.paths.results
    outdir = os.path.join(res, "figures"); os.makedirs(outdir, exist_ok=True)
    set_style()
    print(f"[s06] writing figures to {outdir}")
    fig_ksweep(res, outdir)
    fig_permethod(res, outdir)
    fig_curves(res, outdir)
    print("[s06] done.")


if __name__ == "__main__":
    main()
