"""Preprocess raw NIfTI into resampled/normalised tensors + build eval/heldout splits.

Minimal: for the MSD BraTS tarball, this just verifies loading and writes an
index. Extend per dataset as needed.
"""
import os
import numpy as np

from vcf3d.config import parse_config_arg
from vcf3d.io.datasets import msd_brats_items, make_loader


def main():
    cfg = parse_config_arg()
    items = msd_brats_items(cfg.data.root)
    print(f"found {len(items)} volumes under {cfg.data.root}")
    loader = make_loader(items[:2], cfg)                # sanity-load two
    for b in loader:
        print("loaded image tensor:", tuple(b["image"].shape))
        break
    os.makedirs(cfg.paths.processed, exist_ok=True)
    np.save(os.path.join(cfg.paths.processed, "index.npy"), items)
    print("wrote index.npy — create eval/ and heldout/ splits as symlinks or copies")


if __name__ == "__main__":
    main()
