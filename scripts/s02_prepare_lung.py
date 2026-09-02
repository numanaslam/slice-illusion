"""Prepare MSD Task06_Lung (chest CT) -> pipeline folders. Dataset generality axis
(CT / lung vs BraTS MRI / brain), reusing the tested NIfTI machinery.

Binary task: tumour volume > dataset median (large vs small tumour). Concepts are
morphology / intensity regions WITHIN the tumour:
  core  = eroded tumour interior
  rim   = tumour boundary shell
  dense = tumour voxels with above-median HU
Random negatives = non-tumour lung patches.

Download (no DUA):
  curl.exe -L -o data\\raw\\Task06_Lung.tar https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar
  tar -xf data\\raw\\Task06_Lung.tar -C data\\raw

Run:
  python -m scripts.s02_prepare_lung --config configs\\lung.yaml --max_vols 200
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
from scipy.ndimage import binary_erosion

from vcf3d.config import load_config

HU_MIN, HU_MAX = -1000.0, 400.0            # lung window
CONCEPTS = ["core", "rim", "dense"]


def norm_ct(hu):
    x = np.clip(hu, HU_MIN, HU_MAX)
    x = (x - HU_MIN) / (HU_MAX - HU_MIN)   # [0,1], air ~ 0
    x[hu < -900] = 0.0                      # air -> background
    return x.astype(np.float32)


def resize_vol(v, size):
    t = torch.from_numpy(v)[None, None]
    return F.interpolate(t, size=tuple(size), mode="trilinear", align_corners=False)[0, 0].numpy()


def save_nii(v, path):
    nib.save(nib.Nifti1Image(v[..., None].astype(np.float32), np.eye(4)), path)


def crop(v, center, P):
    half = P // 2
    a = [int(np.clip(round(c) - half, 0, s - P)) for c, s in zip(center, v.shape)]
    return v[a[0]:a[0]+P, a[1]:a[1]+P, a[2]:a[2]+P]


def concept_masks(hu, tumor):
    core = binary_erosion(tumor, iterations=2) if tumor.any() else np.zeros_like(tumor)
    rim = tumor & ~core
    dense = (tumor & (hu > np.median(hu[tumor]))) if tumor.any() else np.zeros_like(tumor)
    return {"core": core, "rim": rim, "dense": dense}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lung.yaml")
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--pos_per_vol", type=int, default=3)
    ap.add_argument("--neg_per_vol", type=int, default=3)
    ap.add_argument("--max_vols", type=int, default=300)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed)
    P = args.patch
    lesion = int(getattr(cfg.data, "lesion_label", 1))   # 1 for Lung; 2 for organ+tumour (Pancreas/Liver)

    imgs = sorted(glob.glob(os.path.join(cfg.data.root, "imagesTr", "*.nii.gz")))
    imgs = [p for p in imgs if not os.path.basename(p).startswith(".")][:args.max_vols]
    lbls = [p.replace("imagesTr", "labelsTr") for p in imgs]
    assert imgs, f"no images under {cfg.data.root}\\imagesTr"

    proc, conc = cfg.paths.processed, cfg.paths.concepts
    for sub in ["train", "eval", "heldout"]:
        shutil.rmtree(os.path.join(proc, sub), ignore_errors=True)
        os.makedirs(os.path.join(proc, sub), exist_ok=True)
    for name in CONCEPTS + ["random", "_classbatch"]:
        shutil.rmtree(os.path.join(conc, name), ignore_errors=True)
        os.makedirs(os.path.join(conc, name, "pos") if name in CONCEPTS
                    else os.path.join(conc, name), exist_ok=True)

    tvol = [int((nib.load(lp).get_fdata() == lesion).sum()) for lp in lbls]
    thr = float(np.median(tvol))
    print(f"[prep] lesion label={lesion} | tumour-volume median threshold = {thr:.0f} voxels")

    idx = rng.permutation(len(imgs)); n = len(idx); tr, ev = int(0.7*n), int(0.85*n)
    split = {i: ("train" if k < tr else "eval" if k < ev else "heldout")
             for k, i in enumerate(idx)}

    class_labels = {}
    for i, (ip, lp) in enumerate(zip(imgs, lbls)):
        hu = nib.load(ip).get_fdata().astype(np.float32)
        lbl = nib.load(lp).get_fdata().astype(np.int16)
        ct = norm_ct(hu); tumor = (lbl == lesion)
        y = int(tvol[i] > thr); base = os.path.basename(ip)

        vol = resize_vol(ct, cfg.data.vol_size)
        save_nii(vol, os.path.join(proc, split[i], base))
        class_labels[os.path.join(split[i], base)] = y
        if split[i] == "eval" and y == 1:
            save_nii(vol, os.path.join(conc, "_classbatch", base))

        masks = concept_masks(hu, tumor)
        for name in CONCEPTS:
            vox = np.argwhere(masks[name])
            for k in range(args.pos_per_vol):
                if len(vox) == 0:
                    break
                patch = crop(ct, vox[rng.integers(len(vox))], P)
                if patch.shape == (P, P, P):
                    save_nii(patch, os.path.join(conc, name, "pos", f"{base[:-7]}_{k}.nii.gz"))
        lung = (ct > 0) & (~tumor); nv = np.argwhere(lung)
        for k in range(args.neg_per_vol):
            if len(nv) == 0:
                break
            patch = crop(ct, nv[rng.integers(len(nv))], P)
            if patch.shape == (P, P, P):
                save_nii(patch, os.path.join(conc, "random", f"{base[:-7]}_{k}.nii.gz"))
        if (i + 1) % 20 == 0:
            print(f"[lung] {i+1}/{len(imgs)} volumes")

    with open(os.path.join(proc, "class_labels.json"), "w") as f:
        json.dump(class_labels, f, indent=2)
    npos = {c: len(glob.glob(os.path.join(conc, c, "pos", "*.nii.gz"))) for c in CONCEPTS}
    print(f"[lung] done. class {sum(class_labels.values())}/{len(class_labels)} pos | "
          f"concepts {npos} | random {len(glob.glob(os.path.join(conc,'random','*.nii.gz')))} | "
          f"classbatch {len(glob.glob(os.path.join(conc,'_classbatch','*.nii.gz')))}")


if __name__ == "__main__":
    main()
