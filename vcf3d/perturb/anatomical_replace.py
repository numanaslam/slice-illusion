"""Anatomically-plausible 3D perturbation operator R (METHOD Piece 2.1) — GPU-native.

R(x, Omega) = x*(1-M) + s*M, with M a Gaussian-feathered soft mask and s an
anatomically-consistent surrogate. Every op runs on x.device in pure torch (no
scipy/CPU round-trips), so the deletion/insertion loops keep the GPU busy.
Modes: zero|mean (OOD baselines) and tissue-mean|context-nn (in-distribution).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def _gaussian1d(sigma: float, device, dtype):
    r = max(1, int(round(3.0 * sigma)))
    xs = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (xs / sigma) ** 2)
    return k / k.sum(), r


def _blur3d(vol, sigma):
    """Separable 3D Gaussian blur. vol: (1,1,H,W,D)."""
    k, r = _gaussian1d(sigma, vol.device, vol.dtype)
    pad_slot = {2: 4, 3: 2, 4: 0}          # which F.pad slots correspond to each spatial dim
    for dim in (2, 3, 4):
        shape = [1, 1, 1, 1, 1]; shape[dim] = k.numel()
        pad = [0, 0, 0, 0, 0, 0]
        pad[pad_slot[dim]] = r; pad[pad_slot[dim] + 1] = r
        vol = F.conv3d(F.pad(vol, pad, mode="replicate"), k.view(shape))
    return vol


def _feather(mask, sigma):
    """mask:(H,W,D) bool -> (H,W,D) soft mask in [0,1] on the same device."""
    m = mask.to(torch.float32).view(1, 1, *mask.shape)
    kd = max(1, int(round(sigma)))
    m = F.max_pool3d(m, kernel_size=2 * kd + 1, stride=1, padding=kd)   # dilation
    m = _blur3d(m, sigma)
    return (m / (m.amax() + 1e-8))[0, 0]


def anatomical_replace(x, mask, bank, cfg):
    """x:(C,H,W,D), mask:(H,W,D) bool. Returns perturbed (C,H,W,D) on x.device."""
    mask = mask.to(x.device)
    M = _feather(mask, cfg.faith.feather_sigma)          # (H,W,D)
    C = x.shape[0]
    mode = cfg.faith.surrogate

    if mode == "zero":
        s = torch.zeros_like(x)
    elif mode == "mean":
        s = torch.empty_like(x)
        for c in range(C):
            v = x[c]
            s[c] = v[v != 0].mean() if (v != 0).any() else x.new_zeros(())
    elif mode == "tissue-mean":
        s = _tissue_mean(x, bank)
    elif mode == "context-nn":
        s = _context_nn(x, mask, bank, cfg)
    else:
        raise ValueError("faith.surrogate must be zero|mean|tissue-mean|context-nn")

    return x * (1 - M) + s * M


def _tissue_mean(x, bank):
    dev = x.device
    t = torch.as_tensor(bank["tissue_label"], device=dev)
    s = torch.zeros_like(x)
    for c in range(x.shape[0]):
        sc = torch.zeros(x.shape[1:], device=dev, dtype=x.dtype)
        for i, lbl in enumerate(bank["classes"]):
            sc[t == lbl] = float(bank["mean"][c, i])
        s[c] = sc
    return s


def _context_nn(x, mask, bank, cfg):
    dev = x.device
    C = x.shape[0]
    mf = mask.to(torch.float32).view(1, 1, *mask.shape)
    ring = (F.max_pool3d(mf, 7, 1, 3) > 0).view(mask.shape) & (~mask)   # context shell
    desc = []
    for c in range(C):
        v = x[c][ring]
        desc.append(float(v.mean()) if v.numel() > 0 else 0.0)
        desc.append(float(v.std()) if v.numel() > 1 else 0.0)
    desc = np.asarray(desc, np.float32)
    j = int(np.argmin(((bank["ring_desc"] - desc) ** 2).sum(1)))
    s = torch.as_tensor(bank["patches"][j], device=dev, dtype=x.dtype)
    if tuple(s.shape) != tuple(x.shape):
        s = F.interpolate(s.unsqueeze(0), size=tuple(x.shape[1:]),
                          mode="trilinear", align_corners=False)[0]
    return s
