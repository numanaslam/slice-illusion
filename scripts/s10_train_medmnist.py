"""Train one backbone per MedMNIST3D dataset (genuine labels) for the Tier-1 breadth
experiment. Saves data/processed/backbone_<arch>_<flag>.pt, which s11 loads.

    # one dataset / arch
    python -m scripts.s10_train_medmnist --config configs/medmnist.yaml \
        --dataset nodulemnist3d --arch resnet --epochs 30

    # sweep a few (PowerShell: run sequentially)
    foreach ($d in "nodulemnist3d","organmnist3d","vesselmnist3d","fracturemnist3d") {
      python -m scripts.s10_train_medmnist --config configs/medmnist.yaml --dataset $d --arch resnet
    }
"""
import argparse
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from vcf3d.config import load_config
from vcf3d.models.backbone import build_net
from vcf3d.io.registry import load_split, info      # routes --dataset lidc | *mnist3d


@torch.no_grad()
def _evaluate(net, X, y, dev, bs=16):
    net.eval(); correct = 0
    for i in range(0, len(X), bs):
        pr = net(X[i:i + bs].to(dev)).argmax(1).cpu()
        correct += int((pr == y[i:i + bs]).sum())
    return correct / max(len(X), 1)


def run(cfg, flag, arch, epochs, size, lr, bs):
    dev = cfg.compute.device
    cfg.model.arch = arch
    cfg.data.num_channels = info(flag).get("num_channels", 1)   # BraTS=4, MedMNIST/LIDC=1
    cfg.model.num_classes = info(flag)["num_classes"]

    Xtr, ytr = load_split(flag, "train", cfg, size=size)
    Xva, yva = load_split(flag, "val", cfg, size=size)
    print(f"[{flag}/{arch}] train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  "
          f"classes={cfg.model.num_classes}", flush=True)

    net = build_net(cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    # class-balanced CE (Organ/Fracture are imbalanced)
    cnt = np.bincount(ytr.numpy(), minlength=cfg.model.num_classes).astype(np.float32)
    w = torch.tensor(cnt.sum() / (cnt + 1e-6), device=dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w / w.mean())
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    os.makedirs(cfg.paths.processed, exist_ok=True)
    out = os.path.join(cfg.paths.processed, f"backbone_{arch}_{flag}.pt")
    best = 0.0
    for ep in range(epochs):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
        sched.step()
        acc = _evaluate(net, Xva, yva, dev)
        if acc >= best:                       # persist BEST-val weights, not the last epoch
            best = acc
            torch.save(net.state_dict(), out)
        print(f"[{flag}/{arch}] epoch {ep + 1}/{epochs}  val_acc={acc:.3f}  best={best:.3f}",
              flush=True)
    print(f"[{flag}/{arch}] saved {out}  (best val_acc={best:.3f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/medmnist.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arch", default="resnet")          # reference | resnet | densenet
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config(a.config)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    run(cfg, a.dataset, a.arch, a.epochs, a.size, a.lr, a.batch)


if __name__ == "__main__":
    main()
