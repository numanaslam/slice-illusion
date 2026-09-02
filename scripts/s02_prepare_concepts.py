"""Prepare BraTS (MSD Task01_BrainTumour) into the folders the pipeline expects.

Produces, under cfg.paths.processed / cfg.paths.concepts:
  processed/train|eval|heldout/*.nii.gz    resized 4-channel volumes (splits)
  processed/class_labels.json              filename -> 0/1 binary class
  concepts/<edema|non_enhancing|enhancing>/pos/*.nii.gz   concept patches
  concepts/random/*.nii.gz                 random-negative patches
  concepts/_classbatch/*.nii.gz            class-1 volumes for the TCAV score

Binary task (proxy): class 1 = enhancing-tumour volume above the dataset median,
else class 0 — a clinically-flavoured label so TCAV/faithfulness have a class to
explain. Swap in tumour grade if you have grade labels.

Run:
    python -m scripts.s02_prepare_concepts --config configs\\rtx3070.yaml
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

SUBSTRUCTURES = {"edema": 1, "non_enhancing": 2, "enhancing": 3}  # MSD BraTS labels


def load_msd(img_path, lbl_path):
    img = nib.load(img_path).get_fdata().astype(np.float32)      # (H,W,D,4)
    lbl = nib.load(lbl_path).get_fdata().astype(np.int16)        # (H,W,D)
    img = np.transpose(img, (3, 0, 1, 2)) if img.ndim == 4 else img[None]
    return img, lbl                                              # (4,H,W,D),(H,W,D)


def znorm(img):
    out = np.zeros_like(img)
    for c in range(img.shape[0]):
        v = img[c]; m = v > 0
        if m.sum() > 0:
            out[c][m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-6)
    return out


def resize_vol(img, size):
    t = torch.from_numpy(img)[None]                             # (1,C,H,W,D)
    t = F.interpolate(t, size=tuple(size), mode="trilinear", align_corners=False)
    return t[0].numpy()


def save_nii(arr_cxyz, path):
    arr = np.transpose(arr_cxyz, (1, 2, 3, 0))                  # (H,W,D,C)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), path)


def crop_patch(img, center, P):
    C, H, W, D = img.shape
    half = P // 2
    a = [int(np.clip(c - half, 0, s - P)) for c, s in zip(center, (H, W, D))]
    return img[:, a[0]:a[0]+P, a[1]:a[1]+P, a[2]:a[2]+P]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--pos_per_vol", type=int, default=3)
    ap.add_argument("--neg_per_vol", type=int, default=3)
    ap.add_argument("--max_vols", type=int, default=150, help="cap volumes for prep speed")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed)

    imgs = sorted(glob.glob(os.path.join(cfg.data.root, "imagesTr", "*.nii.gz")))
    imgs = [p for p in imgs if not os.path.basename(p).startswith(".")][:args.max_vols]
    lbls = [p.replace("imagesTr", "labelsTr") for p in imgs]
    assert imgs, f"no images under {cfg.data.root}\\imagesTr"

    proc, conc = cfg.paths.processed, cfg.paths.concepts
    # Wipe stale outputs from any previous prep so splits stay self-consistent.
    # (A file left in the wrong split folder causes label KeyErrors in s02b.)
    # backbone.pt / class_labels.json live in proc root and are left untouched.
    for sub in ["train", "eval", "heldout"]:
        shutil.rmtree(os.path.join(proc, sub), ignore_errors=True)
    for name in list(SUBSTRUCTURES) + ["random", "_classbatch"]:
        shutil.rmtree(os.path.join(conc, name), ignore_errors=True)
    for sub in ["train", "eval", "heldout"]:
        os.makedirs(os.path.join(proc, sub), exist_ok=True)
    for name in list(SUBSTRUCTURES) + ["random", "_classbatch"]:
        os.makedirs(os.path.join(conc, name, "pos") if name in SUBSTRUCTURES
                    else os.path.join(conc, name), exist_ok=True)

    # ---- pass 1: enhancing volume per case -> median threshold for the class label
    enh_counts = []
    for lp in lbls:
        l = nib.load(lp).get_fdata()
        enh_counts.append(int((l == SUBSTRUCTURES["enhancing"]).sum()))
    thr = float(np.median(enh_counts))
    print(f"[s02] enhancing-volume median threshold = {thr:.0f} voxels")

    # ---- splits (70/15/15)
    idx = rng.permutation(len(imgs))
    n = len(idx); tr, ev = int(0.7*n), int(0.85*n)
    split = {i: ("train" if k < tr else "eval" if k < ev else "heldout")
             for k, i in enumerate(idx)}

    class_labels = {}
    P = args.patch
    for i, (ip, lp) in enumerate(zip(imgs, lbls)):
        img, lbl = load_msd(ip, lp)
        img = znorm(img)
        y = int(enh_counts[i] > thr)
        base = os.path.basename(ip)

        # save resized full volume into its split
        vol = resize_vol(img, cfg.data.vol_size)
        out = os.path.join(proc, split[i], base)
        save_nii(vol, out)
        class_labels[os.path.join(split[i], base)] = y

        # class-1 eval volumes double as the TCAV class batch
        if split[i] == "eval" and y == 1:
            save_nii(vol, os.path.join(conc, "_classbatch", base))

        # concept patches (from original-resolution volume)
        brain = img[0] != 0
        for name, code in SUBSTRUCTURES.items():
            vox = np.argwhere(lbl == code)
            if len(vox) == 0:
                continue
            for k in range(args.pos_per_vol):
                ctr = vox[rng.integers(len(vox))]
                patch = crop_patch(img, ctr, P)
                if patch.shape[1:] == (P, P, P):
                    save_nii(patch, os.path.join(conc, name, "pos", f"{base[:-7]}_{k}.nii.gz"))
        # random negatives: brain voxels not in any tumour substructure
        neg_mask = brain & (lbl == 0)
        nv = np.argwhere(neg_mask)
        for k in range(args.neg_per_vol):
            if len(nv) == 0:
                break
            ctr = nv[rng.integers(len(nv))]
            patch = crop_patch(img, ctr, P)
            if patch.shape[1:] == (P, P, P):
                save_nii(patch, os.path.join(conc, "random", f"{base[:-7]}_{k}.nii.gz"))

        if (i + 1) % 20 == 0:
            print(f"[s02] {i+1}/{len(imgs)} volumes processed")

    with open(os.path.join(proc, "class_labels.json"), "w") as f:
        json.dump(class_labels, f, indent=2)
    npos = {n: len(glob.glob(os.path.join(conc, n, "pos", "*.nii.gz"))) for n in SUBSTRUCTURES}
    print(f"[s02] done. class balance: "
          f"{sum(class_labels.values())}/{len(class_labels)} positive | "
          f"concept patches: {npos} | "
          f"random: {len(glob.glob(os.path.join(conc,'random','*.nii.gz')))} | "
          f"classbatch: {len(glob.glob(os.path.join(conc,'_classbatch','*.nii.gz')))}")


if __name__ == "__main__":
    main()
