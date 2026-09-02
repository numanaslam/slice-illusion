"""Piece 2: 3D faithfulness + slice-illusion experiment across the eval cohort."""
import argparse
import glob
import os
from collections import defaultdict
import numpy as np
import torch
from scipy.stats import wilcoxon

from vcf3d.config import parse_config_arg
from vcf3d.models.backbone import load_backbone
from vcf3d.io.datasets import make_loader
from vcf3d.perturb.tissue_bank import build_tissue_bank
from vcf3d.utils.supervoxels import supervoxels
from vcf3d.utils.gradcam import grad_cam_3d, supervoxel_importance
from vcf3d.concept.tcav import tcav_saliency
from vcf3d.faithfulness.slice_illusion import slice_illusion


def _load_all(files, cfg):
    return [b["image"][0] for b in make_loader([{"image": f} for f in files], cfg)]


def parse_run_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max_vols", type=int, default=10**9,
                    help="cap eval volumes (e.g. 3 for a quick check)")
    ap.add_argument("--slice_band", type=int, default=None,
                    help="override faith.slice_band K without editing the config")
    return ap.parse_known_args()[0]


def main():
    cfg = parse_config_arg()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    if cfg.compute.device == "cuda":
        torch.backends.cudnn.benchmark = True         # autotune convs for fixed sizes
        torch.set_float32_matmul_precision("high")
    bb = load_backbone(cfg)

    held = _load_all(glob.glob(os.path.join(cfg.paths.processed, "heldout", "*.nii*")), cfg)
    bank = build_tissue_bank(held, None, cfg) if held else None

    # real, diverse importance methods for the slice-illusion (>= 3 needed so
    # Kendall-tau / flip-rate are not degenerate)
    cav_path = os.path.join(cfg.paths.results, "cavs.npy")
    cavs = np.load(cav_path, allow_pickle=True).item() if os.path.exists(cav_path) else {}
    if not cavs:
        print("[s04] WARNING: no cavs.npy found (run s03 first) — TCAV methods "
              "skipped; slice-illusion will be under-powered.")

    eval_files = glob.glob(os.path.join(cfg.paths.processed, "eval", "*.nii*"))
    taus, flips, done = [], [], []
    vdrops, sdrops = [], []                                            # pred-drop per protocol
    vol_auc_acc, slice_auc_acc = defaultdict(list), defaultdict(list)  # per-method
    vcurve_acc, scurve_acc = defaultdict(list), defaultdict(list)      # deletion curves
    cls = cfg.model.num_classes - 1
    _args = parse_run_args()
    max_vols = _args.max_vols
    if _args.slice_band is not None:
        cfg.faith.slice_band = _args.slice_band
    print(f"[s04] slice_band K={getattr(cfg.faith, 'slice_band', 1)}", flush=True)
    subset = eval_files[:max_vols]

    os.makedirs(cfg.paths.results, exist_ok=True)
    out_npy = os.path.join(cfg.paths.results, "slice_illusion.npy")
    out_csv = os.path.join(cfg.paths.results, "slice_illusion.csv")

    for i, f in enumerate(subset):
        try:
            # keep the whole volume + label map GPU-resident for the loops
            x = _load_all([f], cfg)[0].to(cfg.compute.device)
            # Explain the model's PREDICTED class (not a hardcoded class), and skip
            # low-confidence volumes: measuring class-1 faithfulness on a class-0
            # volume gives a flat curve and ties all methods together.
            with torch.no_grad():
                prob = torch.softmax(bb(x.unsqueeze(0)), dim=1)[0]
            cls_v, conf = int(prob.argmax()), float(prob.max())
            if conf < 0.6:
                print(f"[skip lowconf] {os.path.basename(f)} p={conf:.2f}", flush=True)
                continue
            L = supervoxels(x, cfg).to(cfg.compute.device)
            S = int(L.max())

            imp_by_method = {}
            cam = grad_cam_3d(bb, x.unsqueeze(0), cls_v, cfg)
            imp_by_method["gradcam"] = supervoxel_importance(cam, L)
            for cname, cv in cavs.items():
                sal = tcav_saliency(bb, x.unsqueeze(0), cv, cls_v, cfg)
                imp_by_method[f"tcav_{cname}"] = supervoxel_importance(sal, L)
            imp_by_method["random"] = np.random.rand(max(S, 1)).astype(np.float32)

            rep = slice_illusion(bb, x, imp_by_method, L, bank, cls_v, cfg)
        except Exception as e:                      # never let one bad volume kill an overnight run
            print(f"[skip] {os.path.basename(f)}: {e}", flush=True)
            continue

        taus.append(rep["kendall_tau"]); flips.append(rep["flip_rate"])
        done.append(os.path.basename(f))
        for m, va, sa in zip(rep["methods"], rep["vol_auc"], rep["slice_auc"]):
            vol_auc_acc[m].append(float(va)); slice_auc_acc[m].append(float(sa))
            vcurve_acc[m].append(rep["vol_curves"][m]); scurve_acc[m].append(rep["slice_curves"][m])
        vdrops.append(rep["vol_drop"]); sdrops.append(rep["slice_drop"])

        # incremental save AFTER EACH volume -> an interrupted run still yields data
        np.save(out_npy, dict(tau=taus, flip=flips, files=done))
        with open(out_csv, "w") as fh:
            fh.write("file,tau,flip\n")
            for nm, t, fl in zip(done, taus, flips):
                fh.write(f"{nm},{t:.4f},{fl:.4f}\n")

        print(f"[{i+1}/{len(subset)}] {os.path.basename(f):22s} "
              f"tau={rep['kendall_tau']:+.2f} flip={rep['flip_rate']:.2f} | "
              f"running mean tau={np.mean(taus):+.3f} flip={np.mean(flips):.3f}",
              flush=True)

    print(f"\n== Slice illusion ({len(taus)} vols) == "
          f"mean tau={np.mean(taus):+.3f} mean flip-rate={np.mean(flips):.3f}")

    # ---- validation: are methods distinguishable, does random rank worst, and
    #      do 2D vs 3D pick DIFFERENT most-faithful methods? ----
    if vol_auc_acc:
        methods = list(vol_auc_acc)
        vmean = {m: float(np.mean(vol_auc_acc[m])) for m in methods}
        smean = {m: float(np.mean(slice_auc_acc[m])) for m in methods}
        print("\nPer-method mean deletion AUC (LOWER = more faithful; "
              "'random' should be HIGHEST):")
        print(f"  {'method':16s} {'volumetric':>11s} {'slice-wise':>11s}")
        for m in sorted(methods, key=lambda k: vmean[k]):
            print(f"  {m:16s} {vmean[m]:11.4f} {smean[m]:11.4f}")
        best_v = min(vmean, key=vmean.get); best_s = min(smean, key=smean.get)
        worst_v = max(vmean, key=vmean.get); worst_s = max(smean, key=smean.get)
        print(f"\n  most-faithful : volumetric -> {best_v} | slice-wise -> {best_s}")
        print(f"  least-faithful: volumetric -> {worst_v} | slice-wise -> {worst_s}"
              f"   {'(random worst = protocol SANE)' if 'random' in (worst_v, worst_s) else '(random NOT worst - CHECK)'}")

        # THE headline metric: 'faithfulness signal' = how much more faithful the
        # best real method is than random. A protocol that can't separate them
        # (signal ~ 0) is uninformative. Big volumetric signal + ~0 slice-wise
        # signal == the slice illusion, quantified.
        if "random" in methods:
            real = [m for m in methods if m != "random"]
            v_sig = vmean["random"] - min(vmean[m] for m in real)
            s_sig = smean["random"] - min(smean[m] for m in real)
            degen = abs(s_sig) < 1e-6            # slice-wise signal ~0 => not a real ratio
            ratio = float("inf") if degen else v_sig / s_sig
            vdrop_m, sdrop_m = float(np.mean(vdrops)), float(np.mean(sdrops))
            print(f"\n  faithfulness signal (random - best real explanation):")
            rtxt = ("n/a  [slice signal ~0: saturated/untrained model - check val_acc]"
                    if degen else f"{ratio:.1f}x")
            print(f"    volumetric = {v_sig:+.4f}   slice-wise = {s_sig:+.4f}   ratio = {rtxt}")

            # #1 transparency: how much the prediction actually drops per protocol
            print(f"  mean prediction drop:  volumetric = {vdrop_m:.3f}   "
                  f"slice-wise = {sdrop_m:.3f}  (band K={getattr(cfg.faith,'slice_band',1)})")

            # #2 rigor: per-volume paired test + bootstrap CI on the signal ratio
            n = len(done); pval, lo, hi = float("nan"), float("nan"), float("nan")
            if n >= 6 and not degen:
                vr, sr = np.array(vol_auc_acc["random"]), np.array(slice_auc_acc["random"])
                vbest = np.min(np.stack([vol_auc_acc[m] for m in real]), axis=0)
                sbest = np.min(np.stack([slice_auc_acc[m] for m in real]), axis=0)
                vsig, ssig = vr - vbest, sr - sbest        # per-volume signals
                try:
                    _, pval = wilcoxon(vsig, ssig)
                except ValueError:
                    pval = float("nan")
                rng = np.random.default_rng(0); rr = []
                for _ in range(2000):
                    idx = rng.integers(0, n, n)
                    den = ssig[idx].mean()
                    rr.append(vsig[idx].mean() / den if abs(den) > 1e-9 else np.nan)
                rr = np.array(rr); rr = rr[np.isfinite(rr)]
                lo, hi = (np.percentile(rr, [2.5, 97.5]) if len(rr) else (np.nan, np.nan))
                print(f"  paired Wilcoxon (vol vs slice signal, n={n}): p={pval:.2e}")
                print(f"  bootstrap 95% CI on ratio: {lo:.1f}x - {hi:.1f}x")

            # append one summary row per run (keyed by K) -> figures read this
            K = int(getattr(cfg.faith, "slice_band", 1))
            summ = os.path.join(cfg.paths.results, "summary.csv")
            new = not os.path.exists(summ)
            with open(summ, "a") as fh:
                if new:
                    fh.write("K,n,vol_signal,slice_signal,ratio,vol_drop,slice_drop,"
                             "wilcoxon_p,ci_lo,ci_hi\n")
                fh.write(f"{K},{n},{v_sig:.6f},{s_sig:.6f},{ratio:.4f},{vdrop_m:.4f},"
                         f"{sdrop_m:.4f},{pval:.3e},{lo:.4f},{hi:.4f}\n")

        # suffix per-K so a K-sweep doesn't clobber the K=1 data the figures use
        Ktag = f"_K{int(getattr(cfg.faith, 'slice_band', 1))}"
        # per-method AUC with std (for error bars / dot plots)
        with open(os.path.join(cfg.paths.results, f"per_method_auc{Ktag}.csv"), "w") as fh:
            fh.write("method,vol_auc,slice_auc,vol_std,slice_std\n")
            for m in methods:
                fh.write(f"{m},{vmean[m]:.4f},{smean[m]:.4f},"
                         f"{np.std(vol_auc_acc[m]):.4f},{np.std(slice_auc_acc[m]):.4f}\n")

        # deletion curves per method: mean AND std (for confidence bands)
        grid = np.linspace(0, 1, 21)
        dc = {"grid": grid, "n": len(done), "vol": {}, "slice": {},
              "vol_std": {}, "slice_std": {}}
        for m in methods:
            V, S = np.stack(vcurve_acc[m]), np.stack(scurve_acc[m])
            dc["vol"][m], dc["vol_std"][m] = V.mean(0), V.std(0)
            dc["slice"][m], dc["slice_std"][m] = S.mean(0), S.std(0)
        np.save(os.path.join(cfg.paths.results, f"deletion_curves{Ktag}.npy"), dc)


if __name__ == "__main__":
    main()
