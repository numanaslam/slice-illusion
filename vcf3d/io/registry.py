"""Dataset resolver so s10/s11 handle any registered 3D dataset uniformly.

`--dataset lidc` routes to io.lidc; every MedMNIST flag routes to io.medmnist3d. Both
expose the same info(flag) and load_split(flag, split, cfg, **kw) -> (X, y) contract,
so the training and faithfulness drivers stay dataset-agnostic.
"""
from __future__ import annotations
from . import medmnist3d, lidc, brats


def _mod(flag: str):
    if flag in lidc.MED:
        return lidc
    if flag in brats.MED:
        return brats
    return medmnist3d


def info(flag: str) -> dict:
    return _mod(flag).info(flag)


def load_split(flag: str, split: str, cfg, **kw):
    return _mod(flag).load_split(flag, split, cfg, **kw)
