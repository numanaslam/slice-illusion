"""MONAI-based loading for 3D medical volumes (BraTS / MSD / generic NIfTI)."""
from __future__ import annotations
import glob
import os
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, ResizeD, ConcatItemsd, ToTensord,
)
from monai.data import Dataset, DataLoader


def build_transforms(cfg, keys):
    """Load -> RAS -> resize to cfg.data.vol_size -> per-channel z-score."""
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        ResizeD(keys=keys, spatial_size=tuple(cfg.data.vol_size), mode="trilinear"),
        NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        ConcatItemsd(keys=keys, name="image", dim=0) if len(keys) > 1 else ToTensord(keys=keys),
        ToTensord(keys=["image"] if len(keys) > 1 else keys),
    ])


def msd_brats_items(root):
    """MSD Task01_BrainTumour stores 4-channel images already; return file dicts."""
    imgs = sorted(glob.glob(os.path.join(root, "imagesTr", "*.nii.gz")))
    return [{"image": p} for p in imgs if not os.path.basename(p).startswith(".")]


def make_loader(items, cfg, keys=("image",), batch_size=1, shuffle=False):
    ds = Dataset(items, transform=build_transforms(cfg, list(keys)))
    # Windows uses multiprocessing 'spawn'; DataLoader workers > 0 can hang/duplicate
    # unless every entrypoint is guarded by `if __name__ == '__main__'`. Default to 0
    # for portability; override via cfg.data.num_workers on Linux/WSL for speed.
    num_workers = int(getattr(getattr(cfg, "data", object()), "num_workers", 0))
    persistent = num_workers > 0
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, persistent_workers=persistent)
