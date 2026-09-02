"""Config loading: YAML -> nested SimpleNamespace, with device resolution."""
from __future__ import annotations
import argparse
from types import SimpleNamespace
import yaml
import torch


def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    return d


def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        cfg = _to_ns(yaml.safe_load(f))
    # resolve device
    want = getattr(cfg.compute, "device", "cuda")
    cfg.compute.device = "cuda" if (want == "cuda" and torch.cuda.is_available()) else "cpu"
    return cfg


def parse_config_arg() -> SimpleNamespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args, _ = ap.parse_known_args()
    return load_config(args.config)
