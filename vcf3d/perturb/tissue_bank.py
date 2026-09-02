"""Surrogate bank for anatomical_replace (METHOD Piece 2.1).

Built from held-out volumes (+ optional organ labels, e.g. TotalSegmentator).
Provides per-tissue intensity means (tissue-mean mode) and exemplar patches with
context-ring descriptors (context-nn mode).
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from sklearn.mixture import GaussianMixture


def build_tissue_bank(volumes, labels, cfg, n_gmm=4, n_exemplars=30):
    """volumes: list of (C,H,W,D) tensors. labels: list of (H,W,D) or None."""
    all_means, all_cls = [], []
    patches, ring_desc = [], []
    tissue_label = None

    for n, vol in enumerate(volumes):
        v = vol.cpu().numpy()
        if labels is not None:
            t = labels[n].cpu().numpy()
        else:
            t = _intensity_gmm(v[0], n_gmm)
        if n == 0:
            tissue_label = t

        for lbl in np.unique(t[t > 0]):
            m = t == lbl
            all_means.append(v[:, m].mean(1)); all_cls.append(int(lbl))

        P, R = _sample_exemplars(v, n_exemplars)
        patches += P; ring_desc += R

    classes = sorted(set(all_cls))
    mean = np.zeros((volumes[0].shape[0], len(classes)), np.float32)
    all_means = np.stack(all_means); all_cls = np.asarray(all_cls)
    for i, cl in enumerate(classes):
        mean[:, i] = all_means[all_cls == cl].mean(0)

    return {"classes": classes, "mean": mean, "tissue_label": tissue_label,
            "patches": patches, "ring_desc": np.stack(ring_desc)}


def _intensity_gmm(vol0, k):
    x = vol0[vol0 != 0].reshape(-1, 1)
    gm = GaussianMixture(k, reg_covar=1e-3, n_init=2).fit(x)
    t = np.zeros(vol0.shape, np.int32)
    t[vol0 != 0] = gm.predict(x) + 1
    return t


def _sample_exemplars(v, n):
    C, H, W, D = v.shape
    ps = max(4, min(H, W, D) // 6)
    P, R = [], []
    rng = np.random.default_rng(0)
    for _ in range(n):
        a, b, d = rng.integers(0, H - ps), rng.integers(0, W - ps), rng.integers(0, D - ps)
        patch = v[:, a:a + ps, b:b + ps, d:d + ps]
        P.append(torch.tensor(patch))
        core = np.zeros((H, W, D), bool); core[a:a + ps, b:b + ps, d:d + ps] = True
        ring = binary_dilation(core, iterations=3) & ~core
        desc = []
        for c in range(C):
            vv = v[c][ring]; desc += [vv.mean(), vv.std()]
        R.append(np.asarray(desc, np.float32))
    return P, R
