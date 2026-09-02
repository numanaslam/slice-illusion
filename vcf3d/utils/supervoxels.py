"""3D supervoxel over-segmentation (SLIC on the volume)."""
from __future__ import annotations
import numpy as np
import torch
from skimage.segmentation import slic


def supervoxels(vol: torch.Tensor, cfg) -> torch.Tensor:
    """vol:(C,H,W,D) -> (H,W,D) int64 labels (0 = background)."""
    g = vol[0].cpu().numpy()
    gmin, gmax = g.min(), g.max()
    gn = (g - gmin) / (gmax - gmin + 1e-8)
    L = slic(gn, n_segments=cfg.svox.num_supervoxels,
             compactness=cfg.svox.compactness, channel_axis=None, start_label=1)
    L[g == 0] = 0
    return torch.as_tensor(L.astype(np.int64))
