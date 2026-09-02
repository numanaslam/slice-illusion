"""Panel-composition x confidence-threshold sensitivity on BraTS (reviewer item).

One GPU pass computes EVERY attribution (Grad-CAM, IG, occlusion, all TCAV concepts,
random) for every eval volume with NO confidence gate, and stores per-volume records.
Panels and tau thresholds are then assembled post hoc for free, so a single run answers:
  (a) how much does the signal ratio depend on which explainers are in the panel?
  (b) how much does it depend on the confidence threshold tau?
  (c) how much of the 46.9x (concept panel, tau=0.6) vs 20.8x (common panel, tau=0.5)
      difference is panel vs cohort?

Outputs (cfg.paths.results):
  panel_pervol_<arch>_<op>.csv   file,conf,method,vol_auc,slice_auc   (crash-safe, incremental)
  panel_sensitivity.csv          panel,tau,arch,operator,n,vol_signal,slice_signal,
                                 ratio,wilcoxon_p,ci_lo,ci_hi        (accumulates rows)

Run (GPU box; occlusion makes this an overnight-class job on the full cohort):
    python -m scripts.s14_panel_sensitivity --config configs/brats_agg.yaml
    python -m scripts.s14_panel_sensitivity --config configs/brats_agg.yaml --operator road
Re-assemble without the GPU pass (after editing PANELS/taus):
    python -m scripts.s14_panel_sensitivity --config configs/brats_agg.yaml --assemble_only
Isolate the checkpoint factor (grid backbone instead of the s04 one):
    python -m scripts.s14_panel_sensitivity --config configs/brats_agg.yaml \
        --ckpt <processed>/backbone_resnet_brats.pt --tag gridckpt
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
from vcf3d.perturb.operators import OPERATORS
from vcf3d.utils.supervoxels import supervoxels
from vcf3d.utils.gradcam import grad_cam_3d, supervoxel_importance
from vcf3d.utils.attributions import ig_supervoxel_importance, occlusion_supervoxel_importance
from vcf3d.concept.tcav import tcav_saliency
from vcf3d.faithfulness.slice_illusion import slice_illusion

TAUS = [0.50, 0.55, 0.60, 0.65, 0.70]


def _panels(methods):
    """Panel definitions over whatever real methods exist in the records."""
    tcav = sorted(m for m in methods if m.startswith("tcav_"))
    have = set(methods)
    P = {}
    if {"gradcam", "ig", "occlusion"} <= have:
        P["common"] = ["gradcam", "ig", "occlusion"]                 # the s11 grid panel
    if tcav and "gradcam" in have:
        P["concept"] = ["gradcam"] + tcav                            # the s04 main panel
    real = sorted(m for m in have if m != "random")
    if len(real) > 1:
        P["union"] = real
    for m in ("gradcam", "ig", "occlusion"):                         # single-explainer extremes
        if m in have:
            P[f"only-{m}"] = [m]
    return P


def _signal_stats(recs, panel, tau):
    """Assemble signal, ratio, Wilcoxon p, log-normal CI for one panel x tau."""
    byfile = defaultdict(dict)
    conf = {}
    for r in recs:
        byfile[r["file"]][r["method"]] = (float(r["vol_auc"]), float(r["slice_auc"]))
        conf[r["file"]] = float(r["conf"])
    need = set(panel) | {"random"}
    files = [f for f in byfile
             if conf[f] >= tau and need <= set(byfile[f])]
    n = len(files)
    if n < 6:
        return None
    vr = np.array([byfile[f]["random"][0] for f in files])
    sr = np.array([byfile[f]["random"][1] for f in files])
    vb = np.array([min(byfile[f][m][0] for m in panel) for f in files])
    sb = np.array([min(byfile[f][m][1] for m in panel) for f in files])
    v_sig, s_sig = float((vr - vb).mean()), float((sr - sb).mean())
    ratio = v_sig / s_sig if abs(s_sig) > 1e-6 else float("inf")
    vs, ss = vr - vb, sr - sb
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


def collect(cfg, args, pervol_csv):
    """The single GPU pass: every attribution for every eval volume, no tau gate."""
    dev = cfg.compute.device
    if args.ckpt:
        cfg.model.ckpt = args.ckpt
        cfg.model.source = "checkpoint"
    bb = load_backbone(cfg)
    replace_fn = OPERATORS[args.operator]

    held = [b["image"][0] for b in make_loader(
        [{"image": f} for f in glob.glob(os.path.join(cfg.paths.processed, "heldout", "*.nii*"))], cfg)]
    bank = build_tissue_bank(held, None, cfg) if held else None

    cav_path = os.path.join(cfg.paths.results, "cavs.npy")
    cavs = np.load(cav_path, allow_pickle=True).item() if os.path.exists(cav_path) else {}
    if not cavs:
        print("[s14] WARNING: no cavs.npy — concept panel unavailable this run", flush=True)

    files = sorted(glob.glob(os.path.join(cfg.paths.processed, "eval", "*.nii*")))[:args.max_vols]
    done = set()
    if os.path.exists(pervol_csv):                       # resume support
        with open(pervol_csv, newline="") as fh:
            done = {r["file"] for r in csv.DictReader(fh)}
        print(f"[s14] resuming: {len(done)} volumes already recorded", flush=True)
    new_file = not os.path.exists(pervol_csv)

    with open(pervol_csv, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["file", "conf", "method", "vol_auc", "slice_auc"])
        for i, f in enumerate(files):
            name = os.path.basename(f)
            if name in done:
                continue
            try:
                x = [b["image"][0] for b in make_loader([{"image": f}], cfg)][0].to(dev)
                with torch.no_grad():
                    prob = torch.softmax(bb(x.unsqueeze(0)), 1)[0]
                cls_v, conf = int(prob.argmax()), float(prob.max())
                L = supervoxels(x, cfg).to(dev)
                S = int(L.max())
                imp = {"gradcam": supervoxel_importance(grad_cam_3d(bb, x.unsqueeze(0), cls_v, cfg), L),
                       "ig": ig_supervoxel_importance(bb, x.unsqueeze(0), cls_v, cfg, L),
                       "occlusion": occlusion_supervoxel_importance(bb, x, L, cls_v, bank, cfg)}
                for cname, cv in cavs.items():
                    sal = tcav_saliency(bb, x.unsqueeze(0), cv, cls_v, cfg)
                    imp[f"tcav_{cname}"] = supervoxel_importance(sal, L)
                imp["random"] = np.random.rand(max(S, 1)).astype(np.float32)
                rep = slice_illusion(bb, x, imp, L, bank, cls_v, cfg, replace_fn=replace_fn)
            except Exception as e:                       # keep the overnight run alive
                print(f"[skip] {name}: {e}", flush=True)
                continue
            for m, va, sa in zip(rep["methods"], rep["vol_auc"], rep["slice_auc"]):
                w.writerow([name, f"{conf:.4f}", m, f"{float(va):.6f}", f"{float(sa):.6f}"])
            fh.flush()
            print(f"[{i+1}/{len(files)}] {name:22s} conf={conf:.2f} "
                  f"methods={len(rep['methods'])}", flush=True)


def assemble(cfg, args, pervol_csv):
    with open(pervol_csv, newline="") as fh:
        recs = list(csv.DictReader(fh))
    methods = sorted({r["method"] for r in recs})
    print(f"[s14] {len({r['file'] for r in recs})} volumes, methods: {methods}", flush=True)
    out = os.path.join(cfg.paths.results, "panel_sensitivity.csv")
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["panel", "tau", "arch", "operator", "n", "vol_signal",
                        "slice_signal", "ratio", "wilcoxon_p", "ci_lo", "ci_hi"])
        for pname, panel in _panels(methods).items():
            for tau in TAUS:
                st = _signal_stats(recs, panel, tau)
                if st is None:
                    continue
                w.writerow([pname, tau, args.arch, args.operator, st["n"],
                            st["vol_signal"], st["slice_signal"], st["ratio"],
                            st["wilcoxon_p"], st["ci_lo"], st["ci_hi"]])
                print(f"  {pname:14s} tau={tau:.2f}  n={st['n']:3d}  "
                      f"ratio={st['ratio']}x  p={st['wilcoxon_p']}", flush=True)
    print(f"[s14] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/brats_agg.yaml")
    ap.add_argument("--arch", default="resnet")
    ap.add_argument("--operator", default="context-nn", choices=list(OPERATORS))
    ap.add_argument("--ckpt", default=None, help="override checkpoint (e.g. the s10 grid backbone)")
    ap.add_argument("--tag", default=None, help="suffix for the per-volume dump filename")
    ap.add_argument("--max_vols", type=int, default=10**9)
    ap.add_argument("--assemble_only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.compute.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")   # numerics parity with s04
    os.makedirs(cfg.paths.results, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    pervol_csv = os.path.join(cfg.paths.results,
                              f"panel_pervol_{args.arch}_{args.operator}{tag}.csv")
    if not args.assemble_only:
        collect(cfg, args, pervol_csv)
    assemble(cfg, args, pervol_csv)


if __name__ == "__main__":
    main()
