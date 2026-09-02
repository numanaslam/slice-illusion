"""Slice-illusion diagnosis (METHOD Piece 2.3) — the headline experiment.

For each explanation method, compute faithfulness two ways: volumetric (3D
supervoxels) vs slice-wise (2D deletion per axial slice, aggregated). Report the
Kendall-tau between the induced method rankings and the pairwise verdict-flip
rate. Low tau / high flip rate == 2D faithfulness tools mislead on 3D models.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
import torch
from scipy.stats import kendalltau

from .deletion_insertion import deletion_insertion_3d, _class_prob, equal_volume_chunks
from ..perturb.anatomical_replace import anatomical_replace


def _slicewise_deletion(backbone, x, importance, L, bank, class_idx, cfg,
                        replace_fn=anatomical_replace):
    """2D-style deletion over axial BANDS of width K = cfg.faith.slice_band.

    K=1 is the strict per-slice (2D) protocol; K -> depth recovers the volumetric
    protocol. Sweeping K shows discriminability emerges only as the perturbation
    gains 3D context (the robustness ablation answering the 'strawman' critique).
    Uses the SAME equal-volume stepping as the volumetric arm, so the arms differ
    only in how much 3D context each perturbation spans.

    Returns (mean_auc, mean_pred_drop) — the drop makes the mechanism explicit:
    a per-slice protocol barely moves a 3D model, so it cannot rank explanations.
    """
    D = x.shape[-1]
    K = max(1, int(getattr(cfg.faith, "slice_band", 1)))
    grid = np.linspace(0, 1, 21)
    aucs, drops, curves = [], [], []
    for d0 in range(0, D, K):
        d1 = min(d0 + K, D)
        Lb = L[:, :, d0:d1]
        sv = torch.unique(Lb[Lb > 0])
        if sv.numel() == 0:
            continue
        sv_np = sv.cpu().numpy()
        imp = importance[sv_np - 1]
        order_local = np.argsort(-imp)
        vol = np.bincount(Lb.cpu().numpy().ravel(),
                          minlength=int(sv_np.max()) + 1)[sv_np]   # per-svox voxels in band
        steps = min(cfg.faith.steps, len(sv_np))
        chunks = equal_volume_chunks(order_local, vol, steps)
        xslc = x.clone(); p = [_class_prob(backbone, xslc, class_idx, cfg)]
        for ch in chunks:
            labels = torch.as_tensor(sv_np[ch], device=L.device)
            mask = torch.zeros_like(L, dtype=torch.bool)
            mask[:, :, d0:d1] = torch.isin(Lb, labels)
            xslc = replace_fn(xslc, mask, bank, cfg)
            p.append(_class_prob(backbone, xslc, class_idx, cfg))
        fr = np.linspace(0, 1, len(p))
        aucs.append(float(np.trapz(p, fr)))
        drops.append(float(p[0] - p[-1]))
        curves.append(np.interp(grid, fr, p))
    mean_curve = np.mean(np.stack(curves), axis=0) if curves else np.full(21, np.nan)
    return (float(np.mean(aucs)) if aucs else 0.0,
            float(np.mean(drops)) if drops else 0.0, mean_curve)


def slice_illusion(backbone, x, imp_by_method, L, bank, class_idx, cfg,
                   replace_fn=anatomical_replace):
    methods = list(imp_by_method)
    vol_auc, slice_auc, vol_drop, slice_drop = [], [], [], []
    vol_curves, slice_curves = {}, {}
    for m in methods:
        di = deletion_insertion_3d(backbone, x, imp_by_method[m], L, bank, class_idx, cfg,
                                   replace_fn=replace_fn)
        vol_auc.append(di["del_auc"])
        vol_drop.append(di["p_del"][0] - di["p_del"][-1])
        vol_curves[m] = di["curve"]
        sauc, sdrop, scurve = _slicewise_deletion(
            backbone, x, imp_by_method[m], L, bank, class_idx, cfg, replace_fn=replace_fn)
        slice_auc.append(sauc); slice_drop.append(sdrop); slice_curves[m] = scurve
    vol_auc, slice_auc = np.array(vol_auc), np.array(slice_auc)

    # Rank correlation between the two protocols' faithfulness orderings.
    # Pass the AUC VALUES directly (both use the same deletion-AUC convention);
    # feeding argsort() indices instead is wrong and flips the sign.
    tau, _ = kendalltau(vol_auc, slice_auc)
    flips = tot = 0
    for i, j in combinations(range(len(methods)), 2):
        si, sj = np.sign(vol_auc[i] - vol_auc[j]), np.sign(slice_auc[i] - slice_auc[j])
        if si != 0 and sj != 0:
            tot += 1; flips += int(si != sj)
    return {"methods": methods, "vol_auc": vol_auc, "slice_auc": slice_auc,
            "kendall_tau": float(tau) if tau == tau else 0.0,
            "flip_rate": flips / max(tot, 1),
            "vol_drop": float(np.mean(vol_drop)), "slice_drop": float(np.mean(slice_drop)),
            "vol_curves": vol_curves, "slice_curves": slice_curves}
