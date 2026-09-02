"""MedMNIST v2 3D loaders — genuine-label volumetric classification across modalities.

Adds real classification tasks (not the BraTS enhancing-volume median-split proxy) so
the slice illusion can be demonstrated on many datasets with true labels — the breadth
a journal (MedIA/TMI) requires. All are public, no data-use agreement, and download on
first use.

    pip install medmnist

Genuine-label 3D sets used here (all single-channel volumes):
    nodulemnist3d   CT   lung-nodule malignancy      (2 classes)   <- from LIDC-IDRI
    organmnist3d    CT   abdominal organ             (11 classes)
    adrenalmnist3d  CT   adrenal shape (normal/hyper)(2 classes)
    fracturemnist3d CT   rib-fracture type           (3 classes)
    vesselmnist3d   MRA  brain-vessel/aneurysm       (2 classes)
    synapsemnist3d  EM   synapse polarity            (2 classes)
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn.functional as F

# flag -> (modality string for tables, num_classes)
MED3D = {
    "nodulemnist3d":   ("CT (lung nodule)",      2),
    "organmnist3d":    ("CT (abdominal organ)",  11),
    "adrenalmnist3d":  ("CT (adrenal)",          2),
    "fracturemnist3d": ("CT (rib fracture)",     3),
    "vesselmnist3d":   ("MRA (brain vessel)",    2),
    "synapsemnist3d":  ("EM (synapse)",          2),
}


def info(flag: str) -> dict:
    modality, k = MED3D[flag]
    return {"modality": modality, "num_classes": k, "num_channels": 1}


def _load_npz(flag: str, split: str, size: int, root: str | None = None):
    """Return (imgs (N,S,S,S) uint8, labels (N,)) via the medmnist package."""
    import medmnist
    from medmnist import INFO
    cls = getattr(medmnist, INFO[flag]["python_class"])
    kw = dict(split=split, download=True, size=size)
    if root is not None:
        os.makedirs(root, exist_ok=True)      # medmnist requires the download dir to exist
        kw["root"] = root
    ds = cls(**kw)
    x = np.asarray(ds.imgs)
    y = np.asarray(ds.labels).reshape(-1).astype(np.int64)
    return x, y


def load_split(flag: str, split: str, cfg, size: int = 64, max_n: int | None = None):
    """Return (X, y): X (N,1,H,W,D) float, per-volume z-scored, resized to cfg.data.vol_size.

    Matches the NIfTI pipeline's NormalizeIntensityd(nonzero=True) so a backbone and its
    faithfulness run see the same normalisation.
    """
    root = getattr(getattr(cfg, "data", object()), "root", None)
    x, y = _load_npz(flag, split, size, root)
    if max_n is not None:
        x, y = x[:max_n], y[:max_n]
    X = torch.from_numpy(x).float().unsqueeze(1)                 # (N,1,S,S,S)
    tgt = tuple(cfg.data.vol_size)
    if tuple(X.shape[2:]) != tgt:
        X = F.interpolate(X, size=tgt, mode="trilinear", align_corners=False)
    for i in range(X.shape[0]):                                  # per-volume z-score on nonzero
        v = X[i]; nz = v != 0
        if bool(nz.any()):
            X[i] = (v - v[nz].mean()) / (v[nz].std() + 1e-6)
    return X, torch.from_numpy(y)
