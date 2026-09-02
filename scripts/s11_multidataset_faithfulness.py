"""Slice illusion across many genuine-label datasets x architectures x operators.

Writes experiments/results/multidataset.csv with one row per (dataset, arch, operator):
faithfulness signal (volumetric vs slice-wise), their ratio, a per-volume paired
Wilcoxon test, and a bootstrap 95% CI. This is the breadth + robustness evidence a
MedIA/TMI submission needs. It reuses the *exact* validated slice_illusion, swapping
only the importance panel (Grad-CAM, IG, occlusion, random) and the deletion operator.

    python -m scripts.s11_multidataset_faithfulness --config configs/medmnist.yaml \
        --datasets nodulemnist3d organmnist3d vesselmnist3d fracturemnist3d \
        --archs resnet reference --operators context-nn road --max_eval 60
"""
import argparse
import csv
import os
from collections import defaultdict
import numpy as np
import torch
from scipy.stats import wilcoxon

from vcf3d.config import load_config
from vcf3d.models.backbone import load_backbone
from vcf3d.io.registry import load_split, info      # routes --dataset lidc | *mnist3d
from vcf3d.perturb.tissue_bank import build_tissue_bank
from vcf3d.perturb.operators import OPERATORS
from vcf3d.utils.supervoxels import supervoxels
from vcf3d.utils.gradcam import grad_cam_3d, supervoxel_importance
from vcf3d.utils.attributions import ig_supervoxel_importance, occlusion_supervoxel_importance
from vcf3d.faithfulness.slice_illusion import slice_illusion

COLS = ["dataset", "modality", "arch", "operator", "n", "vol_signal", "slice_signal",
        "ratio", "wilcoxon_p", "ci_lo", "ci_hi"]


def _read_existing(path):
    """Prior rows keyed by (dataset, arch, operator) so repeated runs accumulate.
    Uses csv so a comma inside a field (e.g. a modality string) never misaligns."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if not rows:
        return {}
    ix = {c: i for i, c in enumerate(rows[0])}
    out = {}
    for r in rows[1:]:
        try:
            d = {c: r[ix[c]] for c in COLS}
        except (KeyError, IndexError):
            continue
        out[(d["dataset"], d["arch"], d["operator"])] = d
    return out


def _write(path, merged):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)                                # quotes fields containing commas
        w.writerow(COLS)
        for r in sorted(merged.values(), key=lambda x: str(x["dataset"])):
            w.writerow([r[c] for c in COLS])


def _panel(bb, x, L, cls_v, bank, cfg):
    """The importance panel: 3 diverse real explainers + a random control."""
    S = int(L.max())
    return {
        "gradcam":   supervoxel_importance(grad_cam_3d(bb, x.unsqueeze(0), cls_v, cfg), L),
        "ig":        ig_supervoxel_importance(bb, x.unsqueeze(0), cls_v, cfg, L),
        "occlusion": occlusion_supervoxel_importance(bb, x, L, cls_v, bank, cfg),
        "random":    np.random.rand(max(S, 1)).astype(np.float32),
    }


def _looks_binary(X, thresh=12):
    """True for near-binary / shape-mask volumes (e.g. Vessel/AdrenalMNIST3D), where
    the intensity in-distribution deletion operator has nothing to delete. Such
    datasets are out of scope: the method targets grayscale CT/MRI, not voxelised
    shapes. Detected by very few distinct intensities per volume."""
    nd = [len(np.unique(np.round(X[i].reshape(-1).cpu().numpy(), 2)))
          for i in range(min(8, X.shape[0]))]
    return float(np.median(nd)) < thresh


def run_one(cfg, flag, arch, op, max_eval, size, conf_min, dump_pervol=False):
    dev = cfg.compute.device
    cfg.model.arch = arch
    cfg.model.source = "checkpoint"
    cfg.data.num_channels = info(flag).get("num_channels", 1)   # BraTS=4, MedMNIST/LIDC=1
    cfg.model.num_classes = info(flag)["num_classes"]
    cfg.model.ckpt = os.path.join(cfg.paths.processed, f"backbone_{arch}_{flag}.pt")
    if not os.path.exists(cfg.model.ckpt):
        print(f"[skip] no checkpoint {cfg.model.ckpt} (run s10 first)", flush=True)
        return None
    bb = load_backbone(cfg)
    replace_fn = OPERATORS[op]

    Xev, _ = load_split(flag, "test", cfg, size=size, max_n=max_eval)
    if _looks_binary(Xev):
        print(f"[excluded] {flag}: near-binary/shape volumes — the intensity "
              f"in-distribution deletion operator is out of scope (grayscale CT/MRI "
              f"only). Skipped, no row written.", flush=True)
        return None
    Xhold, _ = load_split(flag, "train", cfg, size=size, max_n=12)     # tissue bank source
    bank = build_tissue_bank([h for h in Xhold], None, cfg)

    vacc, sacc = defaultdict(list), defaultdict(list)                  # per-method AUCs
    pervol = []                                                        # optional dump rows
    done = 0
    for i in range(Xev.shape[0]):
        try:
            x = Xev[i].to(dev)
            with torch.no_grad():
                prob = torch.softmax(bb(x.unsqueeze(0)), 1)[0]
            cls_v, conf = int(prob.argmax()), float(prob.max())
            if conf < conf_min:                                       # explain confident preds
                continue
            L = supervoxels(x, cfg).to(dev)
            imp = _panel(bb, x, L, cls_v, bank, cfg)
            rep = slice_illusion(bb, x, imp, L, bank, cls_v, cfg, replace_fn=replace_fn)
        except Exception as e:
            print(f"  [skip vol {i}] {e}", flush=True); continue
        for m, va, sa in zip(rep["methods"], rep["vol_auc"], rep["slice_auc"]):
            vacc[m].append(float(va)); sacc[m].append(float(sa))
            pervol.append([flag, arch, op, f"vol{i:04d}", f"{conf:.4f}", m,
                           f"{float(va):.6f}", f"{float(sa):.6f}"])
        done += 1

    if dump_pervol and pervol:
        dp = os.path.join(cfg.paths.results, f"pervol_{flag}_{arch}_{op}.csv")
        with open(dp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "arch", "operator", "file", "conf", "method",
                        "vol_auc", "slice_auc"])
            w.writerows(pervol)
        print(f"  [dump] {dp} ({done} volumes)", flush=True)

    if done < 6:
        print(f"[{flag}/{arch}/{op}] too few confident volumes ({done}); skipped", flush=True)
        return None

    real = [m for m in vacc if m != "random"]
    vmean = {m: float(np.mean(vacc[m])) for m in vacc}
    smean = {m: float(np.mean(sacc[m])) for m in sacc}
    v_sig = vmean["random"] - min(vmean[m] for m in real)
    s_sig = smean["random"] - min(smean[m] for m in real)
    ratio = v_sig / s_sig if abs(s_sig) > 1e-6 else float("inf")

    vbest = np.min(np.stack([vacc[m] for m in real]), 0)
    sbest = np.min(np.stack([sacc[m] for m in real]), 0)
    vs, ss = np.array(vacc["random"]) - vbest, np.array(sacc["random"]) - sbest
    try:
        _, p = wilcoxon(vs, ss)
    except ValueError:
        p = float("nan")
    rng = np.random.default_rng(0); rr = []
    for _ in range(2000):
        idx = rng.integers(0, done, done); den = ss[idx].mean()
        rr.append(vs[idx].mean() / den if abs(den) > 1e-9 else np.nan)
    rr = np.array(rr); rr = rr[np.isfinite(rr) & (rr > 0)]
    # Log-normal CI centred on the point estimate: the ratio-of-means point always
    # lies inside. (A percentile bootstrap is upward-biased for a ratio estimator and
    # can exclude the point.) The SE on log(ratio) comes from the bootstrap spread,
    # which also gives s12 a clean per-study weight = 1/Var(log10 ratio).
    if len(rr) > 2 and np.isfinite(ratio) and ratio > 0:
        se_log = float(np.std(np.log(rr)))
        lo, hi = ratio * np.exp(-1.96 * se_log), ratio * np.exp(1.96 * se_log)
    else:
        lo, hi = float("nan"), float("nan")

    return dict(dataset=flag, modality=info(flag)["modality"], arch=arch, operator=op,
                n=done, vol_signal=round(v_sig, 6), slice_signal=round(s_sig, 6),
                ratio=round(ratio, 4), wilcoxon_p=f"{p:.3e}",
                ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/medmnist.yaml")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--archs", nargs="+", default=["resnet"])
    ap.add_argument("--operators", nargs="+", default=["context-nn"])
    ap.add_argument("--max_eval", type=int, default=60)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--conf_min", type=float, default=0.5)
    ap.add_argument("--dump_pervol", action="store_true",
                    help="also write per-volume records for tau/panel sensitivity (s16)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    if cfg.compute.device == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(cfg.paths.results, exist_ok=True)
    out = os.path.join(cfg.paths.results, "multidataset.csv")
    merged = _read_existing(out)                          # accumulate across runs
    for flag in a.datasets:
        for arch in a.archs:
            for op in a.operators:
                r = run_one(cfg, flag, arch, op, a.max_eval, a.size, a.conf_min,
                            dump_pervol=a.dump_pervol)
                if r:
                    merged[(flag, arch, op)] = r
                    print(f"[done] {flag}/{arch}/{op}: ratio={r['ratio']}x  "
                          f"n={r['n']}  p={r['wilcoxon_p']}", flush=True)
                    _write(out, merged)                   # save after each cell
    print(f"\nwrote {out}  ({len(merged)} rows)")


if __name__ == "__main__":
    main()
