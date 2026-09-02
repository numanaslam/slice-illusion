"""Extra 3D explainers for the faithfulness panel: Integrated Gradients + Occlusion.

Both return per-supervoxel importance (S,), matching utils.gradcam.supervoxel_importance,
so slice_illusion can rank them alongside Grad-CAM, TCAV and random. A larger, more
diverse method panel strengthens the discriminability claim for a journal submission:
it makes "the slice-wise protocol cannot separate ANY of these from random" a stronger
statement than with two methods.
"""
from __future__ import annotations
import numpy as np
import torch

from .gradcam import supervoxel_importance
from ..faithfulness.deletion_insertion import _class_prob
from ..perturb.anatomical_replace import anatomical_replace


def integrated_gradients_3d(backbone, x, class_idx, cfg, steps: int = 32):
    """x:(1,C,H,W,D) -> (H,W,D) attribution (|IG| summed over channels, scaled to [0,1])."""
    x = x.to(cfg.compute.device)
    base = torch.zeros_like(x)                                 # zero (post-z-score) baseline
    total = torch.zeros_like(x)
    for a in torch.linspace(0, 1, steps, device=x.device):
        xi = (base + a * (x - base)).clone().detach().requires_grad_(True)
        with torch.enable_grad():
            logit = backbone(xi)[0, class_idx]
            total = total + torch.autograd.grad(logit, xi)[0]
    ig = ((x - base) * total / steps)[0]                       # (C,H,W,D)
    a = ig.abs().sum(0)                                        # (H,W,D)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return a.detach().cpu().numpy()


def ig_supervoxel_importance(backbone, x, class_idx, cfg, L, steps: int = 32):
    """x:(1,C,H,W,D), L:(H,W,D) -> (S,) mean |IG| per supervoxel."""
    return supervoxel_importance(integrated_gradients_3d(backbone, x, class_idx, cfg, steps), L)


@torch.no_grad()
def occlusion_supervoxel_importance(backbone, x, L, class_idx, bank, cfg):
    """Per-supervoxel occlusion: drop in predicted-class prob when each supervoxel is
    replaced by the in-distribution operator.

    x:(C,H,W,D), L:(H,W,D) labels 1..S -> (S,) importance (larger = more important).
    Uses the same operator as deletion so attribution and evaluation share a
    perturbation model. Costs S forward passes per volume (the run's main compute
    driver; lower svox.num_supervoxels or --max_eval if too slow).
    """
    S = int(L.max())
    p0 = _class_prob(backbone, x, class_idx, cfg)
    imp = np.zeros(S, np.float32)
    for s in range(1, S + 1):
        m = L == s
        if not bool(m.any()):
            continue
        xo = anatomical_replace(x.clone(), m, bank, cfg)
        imp[s - 1] = p0 - _class_prob(backbone, xo, class_idx, cfg)
    return imp
