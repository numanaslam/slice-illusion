"""Random-effects meta-analysis of the slice-illusion effect across datasets.

Reads experiments/results/multidataset.csv (from s11), pools log10(signal ratio) with
DerSimonian-Laird random effects, reports the pooled ratio, its 95% CI, and I^2
heterogeneity, and draws a forest plot (experiments/results/figures/forest.pdf). The
forest plot is the headline multi-dataset figure for a MedIA/TMI submission: it shows
the effect is large and consistent across datasets, architectures and operators.

    python -m scripts.s12_meta_analysis --csv experiments/results/multidataset.csv
"""
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    ix = {c: i for i, c in enumerate(rows[0])}
    out = []
    for r in rows[1:]:
        try:
            ratio = float(r[ix["ratio"]]); lo = float(r[ix["ci_lo"]]); hi = float(r[ix["ci_hi"]])
        except (ValueError, KeyError, IndexError):
            continue
        if not np.isfinite(ratio) or ratio <= 0:
            continue
        out.append(dict(label=f"{r[ix['dataset']]} | {r[ix['arch']]} | {r[ix['operator']]}",
                        ratio=ratio, lo=lo, hi=hi, n=int(r[ix["n"]])))
    return out


def meta(rows):
    """DerSimonian-Laird random effects on y = log10(ratio); se inferred from each CI."""
    y = np.array([np.log10(r["ratio"]) for r in rows])
    se = []
    for r in rows:
        if np.isfinite(r["lo"]) and np.isfinite(r["hi"]) and r["lo"] > 0 and r["hi"] > r["lo"]:
            se.append((np.log10(r["hi"]) - np.log10(r["lo"])) / (2 * 1.96))
        else:
            se.append(y.std() if len(y) > 1 else 0.3)              # fallback when CI missing
    v = np.array(se) ** 2
    w = 1.0 / v
    mu_fixed = (w * y).sum() / w.sum()
    Q = float((w * (y - mu_fixed) ** 2).sum())
    df = len(y) - 1
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0               # between-study variance
    wr = 1.0 / (v + tau2)
    mu = (wr * y).sum() / wr.sum()
    se_mu = np.sqrt(1.0 / wr.sum())
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    # Hartung-Knapp-Sidik-Jonkman: robust variance + t_{k-1} critical value
    q_hk = float((wr * (y - mu) ** 2).sum()) / (df * wr.sum()) if df > 0 else se_mu ** 2
    se_hk = np.sqrt(q_hk)
    try:
        from scipy import stats as _st
        tcrit = float(_st.t.ppf(0.975, df))
    except Exception:
        tcrit = 2.131                                             # t_{15, 0.975}
    return dict(ratio=10 ** mu, lo=10 ** (mu - 1.96 * se_mu),
                hi=10 ** (mu + 1.96 * se_mu),
                hk_lo=10 ** (mu - tcrit * se_hk), hk_hi=10 ** (mu + tcrit * se_hk),
                I2=I2, k=len(rows))


NICE = {"nodulemnist3d": "NoduleMNIST3D", "organmnist3d": "OrganMNIST3D",
        "fracturemnist3d": "FractureMNIST3D", "vesselmnist3d": "VesselMNIST3D",
        "adrenalmnist3d": "AdrenalMNIST3D", "synapsemnist3d": "SynapseMNIST3D",
        "lidc": "LIDC-IDRI", "brats": "BraTS"}


ARCH = {"resnet": "Residual", "reference": "Plain CNN"}
OP = {"road": "ROAD", "context-nn": "context-NN"}


def _pretty(label):
    p = label.split(" | ")
    p[0] = NICE.get(p[0], p[0])
    if len(p) > 1: p[1] = ARCH.get(p[1], p[1])
    if len(p) > 2: p[2] = OP.get(p[2], p[2])
    return " | ".join(p)


def forest(rows, pooled, out):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 600, "axes.linewidth": 0.9,
    })
    rows = sorted(rows, key=lambda r: r["ratio"])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.7))
    for yi, r in zip(y, rows):
        lo = r["lo"] if (np.isfinite(r["lo"]) and r["lo"] > 0) else r["ratio"]
        hi = r["hi"] if (np.isfinite(r["hi"]) and r["hi"] > 0) else r["ratio"]
        ax.plot([lo, hi], [yi, yi], color="0.45", lw=1.2, zorder=1)
        ax.scatter([r["ratio"]], [yi], s=16 + r["n"], color="#0072B2", zorder=2)
    ax.axvline(1, ls=":", color="0.6")                            # ratio = 1: no effect
    ax.axvspan(pooled["hk_lo"], pooled["hk_hi"], color="#D55E00", alpha=0.10, zorder=0)
    ax.axvline(pooled["ratio"], ls="--", color="#D55E00", lw=1.4)
    ax.set_yticks(y); ax.set_yticklabels([_pretty(r["label"]) for r in rows], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("volumetric / slice-wise faithfulness signal  (log scale)")
    ax.set_title(f"Slice illusion across datasets: pooled {pooled['ratio']:.1f}x  "
                 f"(95% HKSJ CI {pooled['hk_lo']:.1f} to {pooled['hk_hi']:.1f}; "
                 f"I2={pooled['I2']:.0f}%, k={pooled['k']})", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=300)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=["experiments/results/multidataset.csv"],
                    help="one or more multidataset.csv files to pool (e.g. MedMNIST + LIDC)")
    a = ap.parse_args()
    rows = []
    for c in a.csv:
        rows += load(c)
    if not rows:
        print("no usable rows in", a.csv, "(run s11 first)"); return
    pooled = meta(rows)
    print(f"pooled ratio {pooled['ratio']:.1f}x  95% CI {pooled['lo']:.1f} to "
          f"{pooled['hi']:.1f}  I2={pooled['I2']:.0f}%  (k={pooled['k']} cells)")
    forest(rows, pooled, os.path.join(os.path.dirname(a.csv[0]), "figures", "forest.pdf"))


if __name__ == "__main__":
    main()
