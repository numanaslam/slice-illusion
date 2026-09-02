"""Volumetric TCAV score (METHOD Piece 1.1).

Directional derivative of the class logit w.r.t. the *bottleneck activation
tensor*, projected onto the CAV at every voxel, then aggregated over the full 3D
grid (mean|sum). Aggregating in 3D — not per-slice — is the volumetric nuance the
paper hinges on.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def volumetric_sensitivity(backbone, x, cav_v, class_idx, cfg):
    """x: (1,C,H,W,D) single volume. Returns scalar aggregated sensitivity S_C,k(x)."""
    device = cfg.compute.device
    # Backbone params are frozen (requires_grad=False); make the INPUT require grad
    # so the bottleneck activation lands in the autograd graph and we can take
    # d(logit)/d(activation).
    x = x.to(device).clone().detach().requires_grad_(True)
    v = torch.as_tensor(cav_v, device=device, dtype=x.dtype).view(1, -1, 1, 1, 1)  # (1,K,1,1,1)

    with torch.enable_grad():
        logits, act = backbone(x, need_activation=True)   # act: (1,K,h,w,d), in-graph
        logit_k = logits[0, class_idx]
        grad = torch.autograd.grad(logit_k, act, retain_graph=False)[0]  # d logit / d act

    per_voxel = (grad * v).sum(dim=1)                 # (1,h,w,d) dot with CAV per voxel
    if cfg.tcav.aggregate == "mean":
        s = per_voxel.mean()
    elif cfg.tcav.aggregate == "sum":
        s = per_voxel.sum()
    else:
        raise ValueError("tcav.aggregate must be mean|sum")
    return float(s.detach().cpu())


def tcav_score(backbone, x_class, cav_v, class_idx, cfg):
    """x_class: (B,C,H,W,D) inputs of the target class.

    Returns (tcav, S): fraction with positive sensitivity, and the raw S values.
    """
    S = np.array([
        volumetric_sensitivity(backbone, x_class[i:i + 1], cav_v, class_idx, cfg)
        for i in range(x_class.shape[0])
    ])
    return float((S > 0).mean()), S


def tcav_saliency(backbone, x, cav_v, class_idx, cfg):
    """Per-voxel concept sensitivity map (Piece 1, for supervoxel localisation).

    x:(1,C,H,W,D) -> (H,W,D) map = |<d logit / d act, CAV>| upsampled to input grid
    and min-max normalised. Averaging this within a supervoxel gives that
    supervoxel's concept-importance for the slice-illusion experiment.
    """
    device = cfg.compute.device
    x = x.to(device).clone().detach().requires_grad_(True)
    v = torch.as_tensor(cav_v, device=device, dtype=x.dtype).view(1, -1, 1, 1, 1)
    with torch.enable_grad():
        logits, act = backbone(x, need_activation=True)     # act:(1,K,h,w,d)
        grad = torch.autograd.grad(logits[0, class_idx], act)[0]
    sal = (grad * v).sum(dim=1, keepdim=True)               # (1,1,h,w,d) per-voxel S
    sal = F.interpolate(sal, size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0]
    sal = sal.abs()                                         # magnitude of influence
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal.detach().cpu().numpy()
