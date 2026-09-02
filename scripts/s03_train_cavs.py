"""Piece 1: learn volumetric CAVs + TCAV scores.

Concept folders under cfg.paths.concepts:
    concepts/<name>/pos/*.nii.gz     positive concept sub-volumes
    concepts/random/*.nii.gz         random-negative pool
    concepts/_classbatch/*.nii.gz    target-class volumes for the TCAV score
"""
import glob
import os
import numpy as np
import torch

from vcf3d.config import parse_config_arg
from vcf3d.models.backbone import load_backbone
from vcf3d.models.activations import pooled_features
from vcf3d.concept.cav import train_cav
from vcf3d.concept.tcav import tcav_score
from vcf3d.io.datasets import make_loader


def _phis(backbone, files, cfg):
    loader = make_loader([{"image": f} for f in files], cfg)
    out = []
    for b in loader:
        out.append(pooled_features(backbone, b["image"], cfg.compute.device, cfg.compute.amp))
    return torch.cat(out).cpu().numpy()


def main():
    cfg = parse_config_arg()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    bb = load_backbone(cfg)
    croot = cfg.paths.concepts

    neg = _phis(bb, glob.glob(os.path.join(croot, "random", "*.nii*")), cfg)
    class_files = glob.glob(os.path.join(croot, "_classbatch", "*.nii*"))
    x_class = torch.cat([b["image"] for b in make_loader(
        [{"image": f} for f in class_files], cfg)])

    results = {}
    cav_vectors = {}                       # mean CAV per concept, for s04 saliency
    for cdir in sorted(glob.glob(os.path.join(croot, "*"))):
        name = os.path.basename(cdir)
        if name in ("random", "_classbatch") or not os.path.isdir(cdir):
            continue
        pos = _phis(bb, glob.glob(os.path.join(cdir, "pos", "*.nii*")), cfg)
        tcavs, accs, vs = [], [], []
        for _ in range(cfg.tcav.num_random_sets):
            sel = np.random.choice(len(neg), min(cfg.tcav.neg_per_set, len(neg)), replace=False)
            cav = train_cav(pos, neg[sel], cfg)
            t, _ = tcav_score(bb, x_class, cav.v, cfg.model.num_classes - 1, cfg)
            tcavs.append(t); accs.append(cav.acc); vs.append(cav.v)
        results[name] = dict(tcav=float(np.mean(tcavs)),
                             ci=float(1.96 * np.std(tcavs) / np.sqrt(len(tcavs))),
                             separability=float(np.mean(accs)))
        v_mean = np.mean(np.stack(vs), axis=0)
        cav_vectors[name] = (v_mean / (np.linalg.norm(v_mean) + 1e-12)).astype(np.float32)
        print(f"{name:18s} TCAV={results[name]['tcav']:.3f}"
              f"±{results[name]['ci']:.3f} sep={results[name]['separability']:.2f}")

    os.makedirs(cfg.paths.results, exist_ok=True)
    np.save(os.path.join(cfg.paths.results, "cav_results.npy"), results)
    np.save(os.path.join(cfg.paths.results, "cavs.npy"), cav_vectors)   # used by s04


if __name__ == "__main__":
    main()
