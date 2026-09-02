"""Pooled-feature extraction (no-grad) for CAV training / concept discovery."""
from __future__ import annotations
import torch


@torch.no_grad()
def pooled_features(backbone, x, device, amp=True):
    """x: (B,C,H,W,D) -> phi: (B,K) global-average-pooled bottleneck features."""
    x = x.to(device)
    with torch.autocast(device_type="cuda", enabled=amp and device == "cuda"):
        _, act = backbone(x, need_activation=True)     # (B,K,h,w,d)
    phi = act.float().mean(dim=(2, 3, 4))              # (B,K)
    return phi
