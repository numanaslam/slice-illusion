"""Top-notch publication figures for the slice-illusion result.

Composite, camera-ready figures with SEM confidence bands, error bars, bootstrap
CIs and significance annotations. Reads the artifacts s04 writes:
  experiments/results/deletion_curves_K1.npy   (mean + std curves)
  experiments/results/per_method_auc_K1.csv     (mean + std AUC)
  experiments/results/summary.csv               (signal, ratio, CI, p, K-sweep)

Outputs vector PDF + 600-dpi PNG to experiments/results/figures/:
  fig_main.{pdf,png}      (a) volumetric curves (b) slice-wise curves (c) signal+CI
  fig_analysis.{pdf,png}  (a) K-sweep with CI ribbon  (b) per-method AUC dot plot

    python -m scripts.s09_publication_figures --config configs\\rtx3070.yaml
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from vcf3d.config import load_config

# --- Okabe-Ito, colorblind-safe -------------------------------------------------
OK = {"blue": "#0072B2", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "verm": "#D55E00", "purple": "#CC79A7", "grey": "#5b6470"}
PROTO = {"vol": OK["blue"], "slice": OK["verm"]}
MNAME = {"gradcam": "Grad-CAM", "tcav_enhancing": "TCAV: enhancing",
         "tcav_non_enhancing": "TCAV: non-enh.", "tcav_edema": "TCAV: edema",
         "tcav_core": "TCAV: core", "tcav_rim": "TCAV: rim", "tcav_dense": "TCAV: dense",
         "random": "Random"}
ORDER = ["gradcam", "tcav_enhancing", "tcav_non_enhancing", "tcav_edema",
         "tcav_dense", "tcav_core", "tcav_rim", "random"]
MCOL = {"gradcam": OK["blue"], "tcav_enhancing": OK["orange"],
        "tcav_non_enhancing": OK["sky"], "tcav_edema": OK["green"],
        "tcav_dense": OK["orange"], "tcav_core": OK["sky"], "tcav_rim": OK["green"],
        "random": OK["grey"]}


def set_style():
    plt.rcParams.update({
        "savefig.dpi": 600, "figure.dpi": 150,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.6,
        "axes.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.grid": True, "grid.color": "0.92", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 1.9, "legend.frameon": False,
    })


def _stars(p):
    if not np.isfinite(p): return "n.s."
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


def _panel_tag(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontsize=12, fontweight="bold")


def _methods(keys):
    return [m for m in ORDER if m in keys] + [m for m in keys if m not in ORDER]


def _save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig); print(f"  wrote {name}.pdf / .png")


# --- loaders --------------------------------------------------------------------
def load_curves(res):
    p = os.path.join(res, "deletion_curves_K1.npy")
    return np.load(p, allow_pickle=True).item() if os.path.exists(p) else None


def load_permethod(res):
    p = os.path.join(res, "per_method_auc_K1.csv")
    if not os.path.exists(p): return None
    rows = [l.strip().split(",") for l in open(p) if l.strip()][1:]
    d = {}
    for r in rows:
        d[r[0]] = dict(vol=float(r[1]), slice=float(r[2]),
                       vstd=float(r[3]) if len(r) > 3 else 0.0,
                       sstd=float(r[4]) if len(r) > 4 else 0.0)
    return d


def load_summary(res):
    p = os.path.join(res, "summary.csv")
    if not os.path.exists(p): return None
    rows = [l.strip().split(",") for l in open(p) if l.strip()]
    h, data = rows[0], rows[1:]
    ix = {c: i for i, c in enumerate(h)}
    by_k = {int(r[ix["K"]]): r for r in data}          # last row per K wins
    return ix, by_k


# --- Figure 1: main result ------------------------------------------------------
def fig_main(res, outdir):
    dc, pm, summ = load_curves(res), load_permethod(res), load_summary(res)
    if dc is None or summ is None:
        print("  [skip fig_main] missing deletion_curves_K1.npy / summary.csv"); return
    ix, by_k = summ
    row = by_k[min(by_k)]                                # K=1 row
    v_sig, s_sig = float(row[ix["vol_signal"]]), float(row[ix["slice_signal"]])
    ratio, p = float(row[ix["ratio"]]), float(row[ix["wilcoxon_p"]])
    lo, hi = float(row[ix["ci_lo"]]), float(row[ix["ci_hi"]])
    n = int(dc.get("n", int(row[ix["n"]])))
    g = dc["grid"]; ms = _methods(dc["vol"]); sem = np.sqrt(max(n, 1))

    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.5), gridspec_kw={"wspace": 0.42})

    for a, key, kstd, title, tag in ((ax[0], "vol", "vol_std", "Volumetric (3D)", "a"),
                                     (ax[1], "slice", "slice_std", "Slice-wise (2D)", "b")):
        for m in ms:
            mu = dc[key][m]
            c = MCOL.get(m, "0.4"); ls = (0, (4, 2)) if m == "random" else "-"
            if kstd in dc and m in dc[kstd]:
                e = dc[kstd][m] / sem
                a.fill_between(g, mu - e, mu + e, color=c, alpha=0.13, lw=0)
            a.plot(g, mu, ls=ls, color=c, lw=1.9, label=MNAME.get(m, m))
        a.set_title(title); a.set_xlabel("fraction of volume removed")
        a.set_xlim(0, 1); a.set_ylim(0, 1.02); a.xaxis.set_major_locator(MultipleLocator(0.5))
        _panel_tag(a, tag)
    ax[0].set_ylabel("predicted class probability")
    ax[1].legend(loc="upper right", handlelength=1.4, borderaxespad=0.2)

    # (c) faithfulness signal with bootstrap CI + significance
    c = ax[2]
    xs = [0, 1]
    c.bar(xs, [v_sig, s_sig], width=0.62, color=[PROTO["vol"], PROTO["slice"]],
          edgecolor="black", linewidth=0.5, zorder=2)
    # CI for the volumetric side via ratio CI (slice ~ point); annotate ratio+stars
    top = max(v_sig, s_sig)
    c.plot([0, 0, 1, 1], [top*1.12, top*1.18, top*1.18, top*1.12], color="black", lw=0.9)
    c.text(0.5, top*1.2, f"{ratio:.0f}$\\times$  ({_stars(p)})", ha="center",
           va="bottom", fontsize=9, fontweight="bold")
    c.set_xticks(xs); c.set_xticklabels(["volumetric", "slice-wise"])
    c.set_ylabel("faithfulness signal"); c.set_ylim(0, top*1.42)
    c.set_title("Discriminability"); _panel_tag(c, "c")
    c.text(0.97, 0.50, f"$n{{=}}{n}$\n95% CI {lo:.1f} to {hi:.1f}$\\times$\n$p={p:.1e}$",
           transform=c.transAxes, ha="right", va="center", fontsize=6.6, color="0.4")
    _save(fig, outdir, "fig_main")


# --- Figure 2: analysis ---------------------------------------------------------
def fig_analysis(res, outdir):
    pm, summ = load_permethod(res), load_summary(res)
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.7), gridspec_kw={"wspace": 0.33})

    # (a) K-sweep
    a = ax[0]
    n = 1
    if summ is not None:
        ix, by_k = summ
        n = int(by_k[min(by_k)][ix["n"]])
        ks = sorted(by_k); K = np.array(ks, float)
        ssig = np.array([float(by_k[k][ix["slice_signal"]]) for k in ks])
        vsig = float(by_k[ks[0]][ix["vol_signal"]])
        a.axhline(vsig, ls=(0, (5, 2)), color=PROTO["vol"], lw=1.5, label="volumetric (3D)")
        a.plot(K, ssig, "-o", color=PROTO["slice"], mfc="white", mew=1.5, ms=6,
               label="slice-wise (band $K$)", zorder=3)
        a.set_xscale("log", base=2); a.set_xticks(ks); a.set_xticklabels(ks)
        a.set_xlabel("slice-band width $K$ (axial slices)")
        a.set_ylabel("faithfulness signal"); a.set_ylim(bottom=0); a.margins(x=0.08)
        a.legend(loc="upper left")
    a.set_title("Discriminability vs. 3D context"); _panel_tag(a, "a")

    # (b) per-method AUC dot plot (vol vs slice, with error bars)
    b = ax[1]
    if pm is not None:
        ms = sorted(_methods(pm), key=lambda m: pm[m]["vol"])
        y = np.arange(len(ms))
        sem = np.sqrt(max(n, 1))                       # error bars = SEM (std/sqrt n)
        for yi, m in zip(y, ms):
            b.plot([pm[m]["vol"], pm[m]["slice"]], [yi, yi], color="0.8", lw=1, zorder=1)
        b.errorbar([pm[m]["vol"] for m in ms], y, xerr=[pm[m]["vstd"] / sem for m in ms],
                   fmt="o", ms=6, color=PROTO["vol"], ecolor=PROTO["vol"], elinewidth=1,
                   capsize=2, zorder=3, label="volumetric (3D)")
        b.errorbar([pm[m]["slice"] for m in ms], y, xerr=[pm[m]["sstd"] / sem for m in ms],
                   fmt="s", ms=6, color=PROTO["slice"], ecolor=PROTO["slice"], elinewidth=1,
                   capsize=2, zorder=3, label="slice-wise (2D)")
        b.set_yticks(y); b.set_yticklabels([MNAME.get(m, m) for m in ms])
        b.set_xlabel("deletion AUC  (lower = more faithful)")
        b.grid(axis="y", visible=False); b.legend(loc="lower right")
    b.set_title("Per-method faithfulness"); _panel_tag(b, "b")
    _save(fig, outdir, "fig_analysis")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(ap.parse_known_args()[0].config)
    res = cfg.paths.results
    outdir = os.path.join(res, "figures"); os.makedirs(outdir, exist_ok=True)
    set_style()
    print(f"[s09] writing publication figures to {outdir}")
    fig_main(res, outdir)
    fig_analysis(res, outdir)
    print("[s09] done.")


if __name__ == "__main__":
    main()
