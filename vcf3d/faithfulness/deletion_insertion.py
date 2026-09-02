"""Volumetric deletion / insertion faithfulness (METHOD Piece 2.2)."""
from __future__ import annotations
import numpy as np
import torch

from ..perturb.anatomical_replace import anatomical_replace


@torch.no_grad()
def _class_prob(backbone, x, class_idx, cfg):
    """x:(C,H,W,D) -> softmax prob of class_idx. Uses autocast for GPU speed."""
    x = x.unsqueeze(0).to(cfg.compute.device)
    use_amp = getattr(cfg.compute, "amp", False) and cfg.compute.device == "cuda"
    with torch.autocast("cuda", enabled=use_amp):
        p = torch.softmax(backbone(x), dim=1)[0, class_idx]
    return float(p.float().cpu())


def equal_volume_chunks(order, sizes, steps):
    """Split `order` (indices into `sizes`) into <=steps groups of ~equal total size.

    Stepping by a fixed NUMBER of regions is confounded because regions vary in
    size; grouping by equal cumulative volume makes every method remove the same
    amount of tissue per step, which is required for a fair faithfulness curve.
    """
    order = np.asarray(order)
    n = len(order)
    if n == 0:
        return []
    cum = np.cumsum(sizes[order].astype(np.int64))
    total = int(cum[-1])
    if total == 0:
        return [order]
    chunks, start = [], 0
    for e in np.linspace(0, total, steps + 1)[1:]:
        end = max(int(np.searchsorted(cum, e, side="right")), start + 1)
        chunks.append(order[start:min(end, n)]); start = end
        if start >= n:
            break
    return [c for c in chunks if len(c)]


def deletion_insertion_3d(backbone, x, importance, L, bank, class_idx, cfg,
                          replace_fn=anatomical_replace):
    """x:(C,H,W,D), importance:(S,) per-supervoxel score, L:(H,W,D) labels 1..S.

    Deletion: replace top-importance supervoxels with R(); prob should fall fast
    (low AUC = faithful). Insertion: start fully replaced, restore top-first;
    prob should rise fast (high AUC = faithful). Steps remove ~equal voxel volume.

    `replace_fn` is the perturbation operator R (default anatomical_replace,
    context-nn). Pass an alternative from perturb.operators (blur|road) to show the
    slice illusion is operator-independent — same signature (x, mask, bank, cfg).
    """
    S = len(importance)
    order = np.argsort(-importance)                       # descending importance
    sizes = np.bincount(L.cpu().numpy().ravel(), minlength=S + 1)[1:S + 1]
    steps = min(cfg.faith.steps, max(S, 1))
    chunks = equal_volume_chunks(order, sizes, steps)

    def mask_of(ch):
        return torch.isin(L, torch.as_tensor(ch + 1, device=L.device))  # labels 1-based

    # ---- Deletion ----
    xdel = x.clone(); p_del = [_class_prob(backbone, xdel, class_idx, cfg)]
    for ch in chunks:
        xdel = replace_fn(xdel, mask_of(ch), bank, cfg)
        p_del.append(_class_prob(backbone, xdel, class_idx, cfg))

    # ---- Insertion (start fully replaced) ----
    xins = replace_fn(x.clone(), L > 0, bank, cfg)
    p_ins = [_class_prob(backbone, xins, class_idx, cfg)]
    for ch in chunks:
        m = mask_of(ch)
        xins[:, m] = x[:, m]
        p_ins.append(_class_prob(backbone, xins, class_idx, cfg))

    frac = np.linspace(0, 1, len(p_del))
    grid = np.linspace(0, 1, 21)
    return {"del_auc": float(np.trapz(p_del, frac)),
            "ins_auc": float(np.trapz(p_ins, frac)),
            "p_del": p_del, "p_ins": p_ins, "frac": frac,
            "curve": np.interp(grid, frac, p_del)}   # fixed-grid curve for figures
