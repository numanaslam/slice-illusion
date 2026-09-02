"""End-to-end smoke test on synthetic 3D data (no dataset, runs on CPU or GPU).

    pytest -q tests/test_smoke.py
"""
import numpy as np
import torch
from types import SimpleNamespace

from vcf3d.models.backbone import load_backbone
from vcf3d.models.activations import pooled_features
from vcf3d.concept.cav import train_cav
from vcf3d.concept.tcav import tcav_score
from vcf3d.perturb.anatomical_replace import anatomical_replace
from vcf3d.faithfulness.deletion_insertion import deletion_insertion_3d
from vcf3d.faithfulness.slice_illusion import slice_illusion
from vcf3d.utils.supervoxels import supervoxels


def _cfg():
    return SimpleNamespace(
        seed=0,
        data=SimpleNamespace(num_channels=2, vol_size=[32, 32, 32]),
        model=SimpleNamespace(source="reference", bottleneck="enc4", num_classes=2),
        tcav=SimpleNamespace(num_random_sets=3, neg_per_set=20, classifier="logistic",
                             aggregate="mean", alpha_sig=0.01),
        svox=SimpleNamespace(num_supervoxels=60, compactness=0.1),
        faith=SimpleNamespace(surrogate="mean", feather_sigma=1.5, steps=5),
        compute=SimpleNamespace(device="cuda" if torch.cuda.is_available() else "cpu", amp=False),
    )


def test_end_to_end():
    cfg = _cfg()
    bb = load_backbone(cfg)

    # Piece 1: CAV + volumetric TCAV
    K = 128
    pos = np.random.randn(40, K) + 0.6
    neg = np.random.randn(40, K)
    cav = train_cav(pos, neg, cfg)
    assert abs(np.linalg.norm(cav.v) - 1) < 1e-4

    x_class = torch.randn(3, cfg.data.num_channels, *cfg.data.vol_size)
    tcav, S = tcav_score(bb, x_class, cav.v, 1, cfg)
    assert 0.0 <= tcav <= 1.0 and len(S) == 3

    # Piece 2: perturbation + faithfulness + slice illusion
    x = torch.randn(cfg.data.num_channels, *cfg.data.vol_size)
    L = supervoxels(x, cfg)
    Smax = int(L.max())
    xp = anatomical_replace(x, L == 1, None, cfg)
    assert xp.shape == x.shape

    imp = np.random.rand(Smax)
    out = deletion_insertion_3d(bb, x, imp, L, None, 1, cfg)
    assert np.isfinite(out["del_auc"]) and np.isfinite(out["ins_auc"])

    rep = slice_illusion(bb, x, {"a": imp, "b": np.random.rand(Smax)}, L, None, 1, cfg)
    assert -1.0 <= rep["kendall_tau"] <= 1.0 and 0.0 <= rep["flip_rate"] <= 1.0
    print("OK", {k: rep[k] for k in ("kendall_tau", "flip_rate")})


if __name__ == "__main__":
    test_end_to_end()
    print("ALL SMOKE TESTS PASSED")
