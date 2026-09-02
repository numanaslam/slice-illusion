"""Confidence-threshold (tau) sensitivity from per-volume dumps. CPU-only.

Reads one or more per-volume record files (from s14, or from s11 run with
--dump_pervol) with columns  file,conf,method,vol_auc,slice_auc  (s11 dumps carry
extra dataset/arch/operator columns and are grouped by them), and reports the
faithfulness signal, ratio, Wilcoxon p, and log-normal CI at each tau. The panel is
every real method present in the file unless --methods restricts it.

    python -m scripts.s16_tau_sensitivity --pervol experiments/results/panel_pervol_resnet_context-nn.csv
    python -m scripts.s16_tau_sensitivity --pervol experiments/results/pervol_*.csv --taus 0.4 0.5 0.6 0.7
"""
import argparse
import csv
import glob
import os
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon


def stats_at(recs, panel, tau):
    byfile, conf = defaultdict(dict), {}
    for r in recs:
        byfile[r["file"]][r["method"]] = (float(r["vol_auc"]), float(r["slice_auc"]))
        conf[r["file"]] = float(r["conf"])
    need = set(panel) | {"random"}
    files = [f for f in byfile if conf[f] >= tau and need <= set(byfile[f])]
    n = len(files)
    if n < 6:
        return None
    vr = np.array([byfile[f]["random"][0] for f in files])
    sr = np.array([byfile[f]["random"][1] for f in files])
    vb = np.array([min(byfile[f][m][0] for m in panel) for f in files])
    sb = np.array([min(byfile[f][m][1] for m in panel) for f in files])
    vs, ss = vr - vb, sr - sb
    v_sig, s_sig = float(vs.mean()), float(ss.mean())
    ratio = v_sig / s_sig if abs(s_sig) > 1e-6 else float("inf")
    try:
        _, p = wilcoxon(vs, ss)
    except ValueError:
        p = float("nan")
    rng = np.random.default_rng(0)
    rr = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        den = ss[idx].mean()
        rr.append(vs[idx].mean() / den if abs(den) > 1e-9 else np.nan)
    rr = np.array(rr)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if len(rr) > 2 and np.isfinite(ratio) and ratio > 0:
        se = float(np.std(np.log(rr)))
        lo, hi = ratio * np.exp(-1.96 * se), ratio * np.exp(1.96 * se)
    else:
        lo, hi = float("nan"), float("nan")
    return dict(n=n, vol_signal=round(v_sig, 6), slice_signal=round(s_sig, 6),
                ratio=round(ratio, 4), wilcoxon_p=f"{p:.3e}",
                ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pervol", nargs="+", required=True)
    ap.add_argument("--taus", nargs="+", type=float,
                    default=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ap.add_argument("--methods", nargs="+", default=None,
                    help="restrict the real panel (default: all non-random methods present)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    paths = []
    for p in a.pervol:
        paths += sorted(glob.glob(p))
    groups = defaultdict(list)                     # (source, dataset, arch, operator) -> recs
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (os.path.basename(p), r.get("dataset", "-"),
                       r.get("arch", "-"), r.get("operator", "-"))
                groups[key].append(r)

    out = a.out or os.path.join(os.path.dirname(paths[0]) or ".", "tau_sensitivity.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "dataset", "arch", "operator", "tau", "n", "vol_signal",
                    "slice_signal", "ratio", "wilcoxon_p", "ci_lo", "ci_hi"])
        for (src, ds, arch, op), recs in sorted(groups.items()):
            methods = sorted({r["method"] for r in recs if r["method"] != "random"})
            panel = a.methods or methods
            for tau in a.taus:
                st = stats_at(recs, panel, tau)
                if st is None:
                    continue
                w.writerow([src, ds, arch, op, tau, st["n"], st["vol_signal"],
                            st["slice_signal"], st["ratio"], st["wilcoxon_p"],
                            st["ci_lo"], st["ci_hi"]])
                print(f"{src} {ds}/{arch}/{op}  tau={tau:.2f}  n={st['n']:3d}  "
                      f"ratio={st['ratio']}x", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
