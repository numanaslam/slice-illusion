"""Prepare LIDC-IDRI (chest CT) into the folders the pipeline expects — generality
axis #2 (different modality + anatomy vs BraTS MRI brain).

Unit = nodule-centred crop (the standard LIDC classification unit). Binary task:
malignant (median malignancy >= 4) vs benign (<= 2); ambiguous (3) skipped.
Concepts = LIDC semantic attributes (spiculation, lobulation, margin, sphericity,
texture, subtlety): concept positives are crops of nodules scoring high (>=4) on
that attribute; random negatives are non-nodule lung patches.

REQUIRES (not tested in this repo — expect to iterate):
  pip install pylidc pynrrd
  a ~/.pylidcrc pointing at the LIDC DICOM directory, e.g.:
      [dicom]
      path = C:\\path\\to\\LIDC-IDRI

Run:
  python -m scripts.s02_prepare_lidc --config configs\\lidc.yaml --max_scans 200
"""
import argparse
import glob
import json
import os
import shutil
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from vcf3d.config import load_config

ATTRS = ["spiculation", "lobulation", "margin", "sphericity", "texture", "subtlety"]
HU_MIN, HU_MAX = -1000.0, 400.0          # lung window


def _norm_ct(hu):
    """Clip to lung window, map air->0 (background), rest -> (0,1]."""
    x = np.clip(hu, HU_MIN, HU_MAX)
    x = (x - HU_MIN) / (HU_MAX - HU_MIN)     # [0,1], air ~ 0
    x[hu < -900] = 0.0                        # treat air as background
    return x.astype(np.float32)


def _resize(vol, size):
    t = torch.from_numpy(vol)[None, None]    # (1,1,H,W,D)
    t = F.interpolate(t, size=tuple(size), mode="trilinear", align_corners=False)
    return t[0, 0].numpy()


def _save(vol, path):
    nib.save(nib.Nifti1Image(vol[..., None].astype(np.float32), np.eye(4)), path)


def _crop(vol, center, half):
    sl = []
    for c, s in zip(center, vol.shape):
        a = int(np.clip(round(c) - half, 0, max(s - 2 * half, 0)))
        sl.append((a, a + 2 * half))
    (a0, a1), (b0, b1), (c0, c1) = sl
    return vol[a0:a1, b0:b1, c0:c1]


def main():
    import pylidc as pl                       # imported here so the repo loads without it
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lidc.yaml")
    ap.add_argument("--max_scans", type=int, default=300)
    ap.add_argument("--crop", type=int, default=64, help="nodule crop cube (voxels)")
    ap.add_argument("--patch", type=int, default=32, help="concept patch cube")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed)
    half, phalf = args.crop // 2, args.patch // 2

    proc, conc = cfg.paths.processed, cfg.paths.concepts
    for sub in ["train", "eval", "heldout"]:
        shutil.rmtree(os.path.join(proc, sub), ignore_errors=True)
        os.makedirs(os.path.join(proc, sub), exist_ok=True)
    for name in ATTRS + ["random", "_classbatch"]:
        shutil.rmtree(os.path.join(conc, name), ignore_errors=True)
        os.makedirs(os.path.join(conc, name, "pos") if name in ATTRS
                    else os.path.join(conc, name), exist_ok=True)

    scans = pl.query(pl.Scan).all()[:args.max_scans]
    print(f"[lidc] {len(scans)} scans")
    class_labels = {}
    n_nod = 0

    for si, scan in enumerate(scans):
        try:
            vol = _norm_ct(scan.to_volume().astype(np.float32))    # (H,W,D)
        except Exception as e:
            print(f"[skip scan {scan.id}] {e}"); continue
        split = "train" if si % 100 < 70 else ("eval" if si % 100 < 85 else "heldout")

        for anns in scan.cluster_annotations():
            mal = float(np.median([a.malignancy for a in anns]))
            if mal == 3:
                continue                                          # drop ambiguous
            y = int(mal >= 4)
            ctr = np.mean([a.centroid for a in anns], axis=0)     # (i,j,k) voxels
            crop = _crop(vol, ctr, half)
            if min(crop.shape) < args.crop:                       # near-edge nodule
                continue
            crop = _resize(crop, cfg.data.vol_size)
            base = f"LIDC_{scan.id}_{n_nod}.nii.gz"; n_nod += 1
            _save(crop, os.path.join(proc, split, base))
            class_labels[os.path.join(split, base)] = y
            if split == "eval" and y == 1:
                _save(crop, os.path.join(conc, "_classbatch", base))

            # concept positives: nodule scores high on an attribute
            cc = np.array(cfg.data.vol_size) // 2
            patch = _crop(crop, cc, phalf)
            if min(patch.shape) == args.patch:
                for at in ATTRS:
                    score = float(np.median([getattr(a, at) for a in anns]))
                    if score >= 4:
                        _save(patch, os.path.join(conc, at, "pos", base))
            # random negative: off-nodule lung patch
            cand = np.argwhere(crop > 0.2)
            if len(cand):
                p = _crop(crop, cand[rng.integers(len(cand))], phalf)
                if min(p.shape) == args.patch:
                    _save(p, os.path.join(conc, "random", base))

        if (si + 1) % 25 == 0:
            print(f"[lidc] {si+1}/{len(scans)} scans, {n_nod} nodules")

    with open(os.path.join(proc, "class_labels.json"), "w") as f:
        json.dump(class_labels, f, indent=2)
    npos = {a: len(glob.glob(os.path.join(conc, a, "pos", "*.nii.gz"))) for a in ATTRS}
    print(f"[lidc] done. {n_nod} nodules | positive class "
          f"{sum(class_labels.values())}/{len(class_labels)} | concept pos {npos} | "
          f"classbatch {len(glob.glob(os.path.join(conc,'_classbatch','*.nii.gz')))}")


if __name__ == "__main__":
    main()
