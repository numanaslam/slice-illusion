"""Post-hoc Volumetric Concept Bottleneck (METHOD Piece 1.2).

Project frozen features onto a CAV bank -> concept activations c(x); fit a sparse
linear head by KL-distillation of the frozen model. Faithfulness is probed by
inference-time intervention (see faithfulness/intervention.py).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class VCBM:
    W: np.ndarray       # (m, nc) concept -> logit
    bias: np.ndarray    # (nc,)
    cav_bank: np.ndarray  # (K, m)
    mu: np.ndarray      # (m,) concept-activation mean (for intervention)
    sd: np.ndarray      # (m,)


def fit_vcbm(phi: np.ndarray, cav_bank: np.ndarray, soft_targets: np.ndarray, cfg) -> VCBM:
    """phi:(N,K)  cav_bank:(K,m)  soft_targets:(N,nc) frozen-model softmax."""
    C = phi @ cav_bank                                # (N,m) concept activations
    mu, sd = C.mean(0), C.std(0) + 1e-6
    Cz = torch.tensor((C - mu) / sd, dtype=torch.float32)
    T = torch.tensor(soft_targets, dtype=torch.float32)

    m, nc = cav_bank.shape[1], soft_targets.shape[1]
    W = torch.zeros(m, nc, requires_grad=True)
    b = torch.zeros(nc, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=cfg.cbm.lr)

    for _ in range(cfg.cbm.max_epochs):
        opt.zero_grad()
        logits = Cz @ W + b
        logp = F.log_softmax(logits, dim=1)
        kl = F.kl_div(logp, T, reduction="batchmean")
        loss = kl + cfg.cbm.l1 * W.abs().mean()       # L1 sparsity
        loss.backward()
        opt.step()

    return VCBM(W=W.detach().numpy(), bias=b.detach().numpy(),
                cav_bank=cav_bank, mu=mu, sd=sd)
