"""BraTS loader (4-channel brain MRI) exposing the same (X, y) interface as
io.medmnist3d / io.lidc, so the paper's primary dataset appears as a homogeneous
point in the multi-dataset forest (s11/s12) with the identical explainer panel and
operator, instead of living only in the main table.

Reads the preprocessed BraTS volumes the paper pipeline already produced under
cfg.paths.processed (e.g. data/processed/{eval,heldout}/*.nii.gz), 4-channel NIfTI,
the same files scripts/s04 globs. Faithfulness explains the model's PREDICTED class
and does not use y, so a label file is optional: y defaults to zeros, and if
class_labels.json is present it is used.
"""
from __future__ import annotations
import glob
import json
import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

# flag -> (modality, num_classes, num_channels). No commas in the modality string
# (the results CSV is comma-separated; s11/s12 also quote, but keep it clean).
MED = {"brats": ("MRI brain tumour (BraTS)", 2, 4)}
_SPLIT = {"train": "train", "val": "eval", "test": "eval", "heldout": "heldout"}


def info(flag: str) -> dict:
    modality, k, ch = MED[flag]
    return {"modality": modality, "num_classes": k, "num_channels": ch}


def load_split(flag: str, split: str, cfg, size=None, max_n=None):
    """Return (X, y): X (N,4,H,W,D) float, per-channel z-scored to cfg.data.vol_size.
    `size` is ignored (BraTS uses cfg.data.vol_size); kept for a uniform signature."""
    sub = _SPLIT.get(split, split)
    d = os.path.join(cfg.paths.processed, sub)
    files = sorted(glob.glob(os.path.join(d, "*.nii*")))
    if max_n is not None:
        files = files[:max_n]
    lab_path = os.path.join(cfg.paths.processed, "class_labels.json")
    labels = json.load(open(lab_path)) if os.path.exists(lab_path) else {}
    tgt = tuple(cfg.data.vol_size)
    xs, ys = [], []
    for f in files:
        arr = nib.load(f).get_fdata().astype(np.float32)                       # (H,W,D,C) or (H,W,D)
        arr = np.transpose(arr, (3, 0, 1, 2)) if arr.ndim == 4 else arr[None]   # (C,H,W,D)
        t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)            # (1,C,H,W,D)
        if tuple(t.shape[2:]) != tgt:
            t = F.interpolate(t, size=tgt, mode="trilinear", align_corners=False)
        v = t[0]
        for c in range(v.shape[0]):                       # per-channel z-score on nonzero
            nz = v[c] != 0
            if bool(nz.any()):
                v[c] = (v[c] - v[c][nz].mean()) / (v[c][nz].std() + 1e-6)
        xs.append(v)
        ys.append(int(labels.get(os.path.join(sub, os.path.basename(f)), 0)))
    if not xs:
        raise RuntimeError(f"no BraTS volumes for split={split!r} under {d} "
                           f"(expected the same eval/heldout NIfTI scripts/s04 uses)")
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)
