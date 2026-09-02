"""Random-concept sanity check (reproducibility contract).

A metric is only trustworthy if RANDOM concepts score at chance: TCAV ~ 0.5 and
not separable. If a random concept 'passes', the pipeline is invalid and no
positive result should be reported.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import ttest_1samp

from ..concept.cav import train_cav
from ..concept.tcav import tcav_score


def random_concept_sanity(backbone, x_class, phi_pool, class_idx, cfg):
    """phi_pool:(Npool,K). Returns (passed, mean_tcav, pvalue)."""
    rng = np.random.default_rng(cfg.seed)
    n = phi_pool.shape[0]
    half = min(cfg.tcav.neg_per_set, n // 2)
    tcavs = []
    for _ in range(cfg.tcav.num_random_sets):
        idx = rng.permutation(n)
        a, b = idx[:half], idx[half:2 * half]
        cav = train_cav(phi_pool[a], phi_pool[b], cfg)  # random split -> no concept
        t, _ = tcav_score(backbone, x_class, cav.v, class_idx, cfg)
        tcavs.append(t)
    tcavs = np.array(tcavs)
    _, p = ttest_1samp(tcavs, 0.5)
    passed = bool(p > cfg.tcav.alpha_sig)               # fail to reject 0.5 == good
    print(f"[sanity] mean random TCAV={tcavs.mean():.3f} (p={p:.3g}) -> "
          f"{'PASS' if passed else 'FAIL'}")
    return passed, float(tcavs.mean()), float(p)
