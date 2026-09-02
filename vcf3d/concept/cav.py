"""Concept Activation Vectors from pooled 3D features (METHOD Piece 1.1)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


@dataclass
class CAV:
    v: np.ndarray      # (K,) unit normal = concept direction
    bias: float
    acc: float         # concept separability (train accuracy)


def train_cav(pos_feat: np.ndarray, neg_feat: np.ndarray, cfg) -> CAV:
    """pos_feat/neg_feat: (Np,K)/(Nn,K). CAV = unit normal of the linear boundary.

    A concept that is not linearly separable from random (low acc) is not a valid
    direction — check `.acc` before trusting its TCAV score.
    """
    X = np.concatenate([pos_feat, neg_feat], axis=0)
    y = np.concatenate([np.ones(len(pos_feat)), np.zeros(len(neg_feat))])

    # Fit the concept boundary DIRECTLY in raw activation space, because tcav.py
    # projects raw-activation gradients onto the CAV. (Standardising then mapping
    # back to raw distorts the direction and collapses TCAV.) A moderate C keeps
    # the lbfgs solve well-conditioned enough to converge without a round-trip.
    if cfg.tcav.classifier == "logistic":
        clf = LogisticRegression(C=1.0, max_iter=5000).fit(X, y)
        w, b = clf.coef_[0], float(clf.intercept_[0])
    elif cfg.tcav.classifier == "svm":
        clf = LinearSVC(C=1.0, max_iter=5000).fit(X, y)
        w, b = clf.coef_[0], float(clf.intercept_[0])
    else:
        raise ValueError("tcav.classifier must be logistic|svm")

    v = w / (np.linalg.norm(w) + 1e-12)
    acc = float((clf.predict(X) == y).mean())
    return CAV(v=v.astype(np.float32), bias=b, acc=acc)
