"""Full-resolution LIDC-IDRI loader (chest CT, nodule-crop malignancy) exposing the
same (X, y) interface as io.medmnist3d, so the multi-dataset faithfulness run (s11)
and meta-analysis (s12) treat LIDC identically to the MedMNIST breadth sets. LIDC is
the clinically-serious, full-res CT anchor for the journal breadth argument.

Prerequisite: run scripts/s02_prepare_lidc.py first (needs pylidc + the LIDC DICOM).
It writes, under cfg.paths.processed:
    {train,eval,heldout}/LIDC_*.nii.gz     nodule crops resized to cfg.data.vol_size
    class_labels.json                       "{split}/{basename}.nii.gz" -> 0|1 malignancy

Train the backbone for the aggregation path with s10 (registry-aware) so training and
s11 evaluation share this module's normalisation:
    python -m scripts.s10_train_medmnist --config configs/lidc.yaml --dataset lidc --arch resnet --batch 2
"""
from __future__ import annotations
import glob
import json
import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

MED = {"lidc": ("CT (lung nodule, LIDC-IDRI)", 2)}

# medmnist-style split name -> the LIDC folder s02_prepare_lidc produces.
# LIDC has no separate val set; training validates on eval, faithfulness runs on eval,
# and heldout seeds the tissue bank.
_SPLIT = {"train": "train", "val": "eval", "test": "eval", "heldout": "heldout"}


def info(flag: str) -> dict:
    modality, k = MED[flag]
    return {"modality": modality, "num_classes": k, "num_channels": 1}


def _labels(cfg) -> dict:
    p = os.path.join(cfg.paths.processed, "class_labels.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found — run scripts/s02_prepare_lidc first")
    with open(p) as f:
        return json.load(f)


def load_split(flag: str, split: str, cfg, size=None, max_n: int | None = None):
    """Return (X, y): X (N,1,H,W,D) float, per-volume z-scored to cfg.data.vol_size,
    y (N,) long. `size` is ignored (LIDC uses cfg.data.vol_size); kept for a uniform
    signature with io.medmnist3d.load_split."""
    sub = _SPLIT.get(split, split)
    labels = _labels(cfg)
    files = sorted(glob.glob(os.path.join(cfg.paths.processed, sub, "*.nii.gz")))
    files = [f for f in files if os.path.join(sub, os.path.basename(f)) in labels]
    if max_n is not None:
        files = files[:max_n]
    tgt = tuple(cfg.data.vol_size)
    xs, ys = [], []
    for f in files:
        arr = nib.load(f).get_fdata().astype(np.float32)          # (H,W,D,1) or (H,W,D)
        arr = np.transpose(arr, (3, 0, 1, 2)) if arr.ndim == 4 else arr[None]   # (1,H,W,D)
        t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)            # (1,1,H,W,D)
        if tuple(t.shape[2:]) != tgt:
            t = F.interpolate(t, size=tgt, mode="trilinear", align_corners=False)
        v = t[0]; nz = v != 0
        if bool(nz.any()):
            v = (v - v[nz].mean()) / (v[nz].std() + 1e-6)          # match NormalizeIntensity(nonzero)
        xs.append(v)
        ys.append(int(labels[os.path.join(sub, os.path.basename(f))]))
    if not xs:
        raise RuntimeError(f"no LIDC volumes for split={split!r} under "
                           f"{cfg.paths.processed}/{sub} — run s02_prepare_lidc")
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)
