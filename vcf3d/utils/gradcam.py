"""3D Grad-CAM saliency baseline + per-supervoxel importance helpers."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def grad_cam_3d(backbone, x, class_idx, cfg):
    """x:(1,C,H,W,D) -> (H,W,D) CAM in [0,1] upsampled to input grid."""
    x = x.to(cfg.compute.device).clone().detach().requires_grad_(True)
    with torch.enable_grad():
        logits, act = backbone(x, need_activation=True)   # act:(1,K,h,w,d), in-graph
        grad = torch.autograd.grad(logits[0, class_idx], act)[0]
    w = grad.mean(dim=(2, 3, 4), keepdim=True)        # (1,K,1,1,1)
    cam = F.relu((w * act).sum(1, keepdim=True))      # (1,1,h,w,d)
    cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
    cam = cam[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy()


def supervoxel_importance(cam, L):
    """Mean CAM per supervoxel. cam:(H,W,D), L:(H,W,D) labels 1..S -> (S,)."""
    Ln = L.cpu().numpy()
    S = int(Ln.max())
    imp = np.zeros(S, np.float32)
    for s in range(1, S + 1):
        m = Ln == s
        if m.any():
            imp[s - 1] = cam[m].mean()
    return imp
