"""Classification performance of every frozen backbone (reviewer item: perf table).

Reports, per dataset x architecture, the test-set accuracy, balanced accuracy, macro-F1,
and AUROC with bootstrap 95% CIs, plus the evaluation-set size and the number retained
after the confidence gate. No training and no new experiments: it only evaluates the
checkpoints the faithfulness study already uses, so the paper can state that the models it
explains actually work.

    python -m scripts.s23_model_performance --config configs/medmnist.yaml \
        --datasets nodulemnist3d organmnist3d fracturemnist3d --archs resnet reference
    python -m scripts.s23_model_performance --config configs/brats_agg.yaml --datasets brats --archs resnet reference
Output: experiments/results/model_performance.csv  (accumulates rows)
"""
import argparse
import csv
import os
import numpy as np
import torch

from vcf3d.config import load_config
from vcf3d.models.backbone import load_backbone
from vcf3d.io.registry import load_split, info

COLS = ["dataset", "arch", "n_test", "n_conf", "accuracy", "acc_lo", "acc_hi",
        "balanced_acc", "macro_f1", "auroc", "auroc_lo", "auroc_hi"]


def _metrics(y, pred, prob, rng, B=2000):
    y, pred = np.asarray(y), np.asarray(pred)
    acc = float((y == pred).mean())
    # balanced accuracy + macro-F1
    classes = np.unique(y)
    recalls, f1s = [], []
    for c in classes:
        tp = int(((pred == c) & (y == c)).sum()); fn = int(((pred != c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        recalls.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    bacc, mf1 = float(np.mean(recalls)), float(np.mean(f1s))
    # AUROC (binary: positive class prob; multiclass: one-vs-rest macro)
    prob = np.asarray(prob)
    try:
        from sklearn.metrics import roc_auc_score
        if len(classes) == 2:
            auroc = float(roc_auc_score(y, prob[:, 1]))
        else:
            auroc = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    except Exception:
        auroc = float("nan")
    # bootstrap CIs for acc and auroc
    accs, aucs = [], []
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        accs.append((y[idx] == pred[idx]).mean())
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y[idx])) < 2:
                continue
            aucs.append(roc_auc_score(y[idx], prob[idx, 1]) if len(classes) == 2
                        else roc_auc_score(y[idx], prob[idx], multi_class="ovr", average="macro"))
        except Exception:
            pass
    alo, ahi = np.percentile(accs, [2.5, 97.5])
    ulo, uhi = (np.percentile(aucs, [2.5, 97.5]) if aucs else (float("nan"), float("nan")))
    return acc, float(alo), float(ahi), bacc, mf1, auroc, float(ulo), float(uhi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--archs", nargs="+", default=["resnet", "reference"])
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--max_eval", type=int, default=10**9)
    ap.add_argument("--conf_min", type=float, default=0.5)
    a = ap.parse_args()
    cfg = load_config(a.config)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = cfg.compute.device
    rng = np.random.default_rng(0)
    out = os.path.join(cfg.paths.results, "model_performance.csv")
    rows = {}
    if os.path.exists(out):
        for r in csv.DictReader(open(out)):
            rows[(r["dataset"], r["arch"])] = r

    for flag in a.datasets:
        for arch in a.archs:
            cfg.model.arch = arch; cfg.model.source = "checkpoint"
            cfg.data.num_channels = info(flag).get("num_channels", 1)
            cfg.model.num_classes = info(flag)["num_classes"]
            cfg.model.ckpt = os.path.join(cfg.paths.processed, f"backbone_{arch}_{flag}.pt")
            if not os.path.exists(cfg.model.ckpt):
                print(f"[skip] no checkpoint {cfg.model.ckpt}", flush=True); continue
            bb = load_backbone(cfg)
            X, y = load_split(flag, "test", cfg, size=a.size, max_n=a.max_eval)
            preds, probs, conf = [], [], 0
            for i in range(X.shape[0]):
                with torch.no_grad():
                    p = torch.softmax(bb(X[i].to(dev).unsqueeze(0)), 1)[0].cpu().numpy()
                probs.append(p); preds.append(int(p.argmax()))
                if p.max() >= a.conf_min:
                    conf += 1
            yl = [int(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
            m = _metrics(yl, preds, probs, rng)
            rows[(flag, arch)] = dict(zip(COLS, [flag, arch, len(yl), conf,
                round(m[0], 4), round(m[1], 4), round(m[2], 4), round(m[3], 4),
                round(m[4], 4), round(m[5], 4), round(m[6], 4), round(m[7], 4)]))
            print(f"[{flag}/{arch}] acc={m[0]:.3f} ({m[1]:.3f}-{m[2]:.3f}) "
                  f"bacc={m[3]:.3f} F1={m[4]:.3f} AUROC={m[5]:.3f} n={len(yl)} conf={conf}", flush=True)
            with open(out, "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(COLS)
                for k in sorted(rows):
                    w.writerow([rows[k][c] for c in COLS])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
