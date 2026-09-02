"""Inference-time concept intervention for V-CBM faithfulness (METHOD Piece 2.2)."""
from __future__ import annotations
import numpy as np


def inference_time_intervention(vcbm, phi, concept_idx, class_idx):
    """phi:(K,) single pooled feature. Force concept c->0, measure logit change.

    A faithful concept gives observed change ~ W[c,k]*c_z[c]  (ratio ~ 1).
    """
    c = phi @ vcbm.cav_bank                            # (m,)
    cz = (c - vcbm.mu) / vcbm.sd
    logits0 = cz @ vcbm.W + vcbm.bias
    cz_abl = cz.copy(); cz_abl[concept_idx] = 0.0
    logits1 = cz_abl @ vcbm.W + vcbm.bias

    observed = float(logits0[class_idx] - logits1[class_idx])
    predicted = float(vcbm.W[concept_idx, class_idx] * cz[concept_idx])
    return {"observed": observed, "predicted": predicted,
            "ratio": observed / (predicted + 1e-8)}
