"""Train the reference 3D classifier so the 'frozen' backbone is meaningful.

TCAV / faithfulness on a randomly-initialised network is meaningless, so we train
the small ReferenceEncoder3D on the binary task from s02, then save the weights.
Point the config at them with:  model.source: checkpoint

This is a minimal classifier to make the pipeline scientifically valid on your
8 GB GPU; for the paper, swap in a stronger backbone or a pretrained 3D FM
(model.source: checkpoint already supports loading one).

Run:
    python -m scripts.s02b_train_classifier --config configs\\rtx3070.yaml
"""
import argparse
import glob
import json
import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from vcf3d.config import load_config
from vcf3d.models.backbone import build_net, checkpoint_path


class VolDataset(Dataset):
    def __init__(self, proc, split, labels):
        allf = sorted(glob.glob(os.path.join(proc, split, "*.nii.gz")))
        # keep only files that have a label (guards against stale files left by a
        # previous prep run with a different split assignment)
        self.files = [f for f in allf
                      if os.path.join(split, os.path.basename(f)) in labels]
        self.labels = labels
        self.split = split

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        arr = nib.load(f).get_fdata().astype(np.float32)        # (H,W,D,C)
        x = torch.from_numpy(np.transpose(arr, (3, 0, 1, 2)))   # (C,H,W,D)
        key = os.path.join(self.split, os.path.basename(f))
        return x, int(self.labels[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=2)        # 8 GB-safe at 96^3
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dev = cfg.compute.device
    torch.manual_seed(cfg.seed)

    with open(os.path.join(cfg.paths.processed, "class_labels.json")) as f:
        labels = json.load(f)

    tr = DataLoader(VolDataset(cfg.paths.processed, "train", labels),
                    batch_size=args.batch, shuffle=True,
                    num_workers=int(getattr(cfg.data, "num_workers", 0)))
    ev = DataLoader(VolDataset(cfg.paths.processed, "eval", labels),
                    batch_size=args.batch, shuffle=False,
                    num_workers=int(getattr(cfg.data, "num_workers", 0)))

    net = build_net(cfg).to(dev)
    print(f"[s02b] training arch={getattr(cfg.model,'arch','reference')}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.compute.amp and dev == "cuda")

    for ep in range(args.epochs):
        net.train(); tot = 0.0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=cfg.compute.amp and dev == "cuda"):
                loss = lossf(net(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss) * x.size(0)

        net.eval(); correct = n = 0
        with torch.no_grad():
            for x, y in ev:
                x, y = x.to(dev), y.to(dev)
                correct += int((net(x).argmax(1) == y).sum()); n += x.size(0)
        print(f"epoch {ep+1:02d}/{args.epochs}  loss={tot/len(tr.dataset):.4f}  "
              f"val_acc={correct/max(n,1):.3f}")

    os.makedirs(cfg.paths.processed, exist_ok=True)
    ckpt = checkpoint_path(cfg)                          # backbone_{arch}.pt
    torch.save(net.state_dict(), ckpt)
    print(f"[s02b] saved {ckpt} — set model.source: checkpoint (checkpoint auto-"
          f"resolves per arch)")


if __name__ == "__main__":
    main()
