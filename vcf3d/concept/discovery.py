"""Unsupervised volumetric concept discovery — 3D-ACE (METHOD Piece 1.3).

Over-segment each volume into 3D supervoxels, embed each supervoxel by a masked
forward pass, then cluster embeddings across the corpus. Each cluster is a
candidate concept whose members seed the positive set for a CAV.
"""
from __future__ import annotations
import numpy as np
import torch
from sklearn.cluster import KMeans

from ..utils.supervoxels import supervoxels
from ..models.activations import pooled_features


def discover_concepts(backbone, volumes, cfg, n_concepts=None):
    """volumes: iterable of (C,H,W,D) tensors. Returns list of concept dicts."""
    feats, meta = [], []
    for n, vol in enumerate(volumes):
        L = supervoxels(vol, cfg)                     # (H,W,D) int labels
        for s in np.unique(L[L > 0]):
            mask = torch.as_tensor(L == s)
            patch = vol * mask                        # isolate supervoxel
            phi = pooled_features(backbone, patch.unsqueeze(0),
                                  cfg.compute.device, cfg.compute.amp)
            feats.append(phi[0].cpu().numpy()); meta.append((n, int(s)))

    feats = np.stack(feats)
    k = n_concepts or max(2, len(feats) // 20)
    km = KMeans(n_clusters=k, n_init=5, random_state=cfg.seed).fit(feats)
    concepts = []
    for c in range(k):
        sel = km.labels_ == c
        concepts.append({"members": feats[sel], "centroid": km.cluster_centers_[c],
                         "meta": [meta[i] for i in np.where(sel)[0]]})
    return concepts
