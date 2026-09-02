"""Alternative deletion operators (blur / ROAD) sharing the anatomical_replace
signature, so slice_illusion(..., replace_fn=op) shows the slice illusion is not an
artefact of the context-nn surrogate. A journal reviewer will ask exactly this: "is
the effect a property of the model, or of your one masking operator?"

Each operator: (x:(C,H,W,D), mask:(H,W,D) bool, bank, cfg) -> perturbed (C,H,W,D),
on x.device, feathered like the in-distribution operator so only the fill differs.
"""
from __future__ import annotations
import torch

from .anatomical_replace import _feather, _blur3d, anatomical_replace


def _blur_channels(x, sigma):
    return torch.stack([_blur3d(x[c].view(1, 1, *x.shape[1:]), sigma)[0, 0]
                        for c in range(x.shape[0])])


def blur_replace(x, mask, bank, cfg):
    """Fill the masked region with a heavily blurred (locally-averaged) copy."""
    mask = mask.to(x.device)
    M = _feather(mask, cfg.faith.feather_sigma)
    s = _blur_channels(x, float(getattr(cfg.faith, "blur_sigma", 6.0)))
    return x * (1 - M) + s * M


def road_replace(x, mask, bank, cfg):
    """ROAD-style noisy linear imputation.

    Approximates ROAD's noisy linear imputation without a per-mask linear solve: a
    large-sigma blur supplies a smooth in-fill from surrounding context, plus small
    Gaussian noise so the operator is not a deterministic low-pass artefact. Removes
    the class-out-of-distribution confound that plain zero/mean masking introduces.
    """
    mask = mask.to(x.device)
    M = _feather(mask, cfg.faith.feather_sigma)
    s = _blur_channels(x, float(getattr(cfg.faith, "road_sigma", 8.0)))
    noise = float(getattr(cfg.faith, "road_noise", 0.03)) * torch.randn_like(x)
    return x * (1 - M) + (s + noise) * M


# name -> operator, for CLI selection in s11. context-nn is the paper's default.
OPERATORS = {"context-nn": anatomical_replace, "blur": blur_replace, "road": road_replace}
