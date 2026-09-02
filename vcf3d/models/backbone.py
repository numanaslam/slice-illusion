"""Frozen 3D backbones with a named-activation hook.

The pipeline never trains the backbone at explain time; it (a) reads activations
at a bottleneck layer and (b) differentiates a class logit w.r.t. that activation
tensor for volumetric TCAV.

Two architecture families (`cfg.model.arch`) so results can be shown to
generalise across model families, not just one net:
  - 'reference' : plain 3D CNN (ReferenceEncoder3D)
  - 'resnet'    : residual 3D CNN (ResEncoder3D)
Both expose enc1..enc4; use `enc3` as the concept bottleneck (nonlinearity
downstream — enc4 sits right before GAP+linear and gives degenerate TCAV).
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn


# --------------------------- architectures -----------------------------------
class ReferenceEncoder3D(nn.Module):
    """Plain 3D CNN classifier. Bottleneck layer named 'enc3'."""

    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv3d(ci, co, 3, padding=1), nn.BatchNorm3d(co), nn.ReLU(inplace=True))
        self.enc1 = block(in_ch, 16); self.p1 = nn.MaxPool3d(2)
        self.enc2 = block(16, 32);    self.p2 = nn.MaxPool3d(2)
        self.enc3 = block(32, 64);    self.p3 = nn.MaxPool3d(2)     # bottleneck l
        self.enc4 = block(64, 128)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.p1(self.enc1(x))
        x = self.p2(self.enc2(x))
        x = self.p3(self.enc3(x))
        x = self.enc4(x)
        return self.fc(self.gap(x).flatten(1))


class _ResBlock3D(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.c1 = nn.Conv3d(ci, co, 3, padding=1, bias=False); self.b1 = nn.BatchNorm3d(co)
        self.c2 = nn.Conv3d(co, co, 3, padding=1, bias=False); self.b2 = nn.BatchNorm3d(co)
        self.skip = (nn.Sequential(nn.Conv3d(ci, co, 1, bias=False), nn.BatchNorm3d(co))
                     if ci != co else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return self.act(h + self.skip(x))


class ResEncoder3D(nn.Module):
    """Residual 3D CNN — different family from ReferenceEncoder3D. Bottleneck 'enc3'."""

    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.enc1 = _ResBlock3D(in_ch, 16); self.p1 = nn.MaxPool3d(2)
        self.enc2 = _ResBlock3D(16, 32);    self.p2 = nn.MaxPool3d(2)
        self.enc3 = _ResBlock3D(32, 64);    self.p3 = nn.MaxPool3d(2)   # bottleneck l
        self.enc4 = _ResBlock3D(64, 128)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.p1(self.enc1(x))
        x = self.p2(self.enc2(x))
        x = self.p3(self.enc3(x))
        x = self.enc4(x)
        return self.fc(self.gap(x).flatten(1))


def build_net(cfg) -> nn.Module:
    arch = getattr(cfg.model, "arch", "reference")
    if arch == "reference":
        return ReferenceEncoder3D(cfg.data.num_channels, cfg.model.num_classes)
    if arch == "resnet":
        return ResEncoder3D(cfg.data.num_channels, cfg.model.num_classes)
    if arch == "densenet":
        # Standard-library, higher-capacity backbone (answers the "would a real model
        # behave differently?" reviewer point). For the Grad-CAM/TCAV hook, set
        # model.bottleneck to a real submodule, e.g. 'features.transition2'
        # (pool+conv transition with nonlinearity downstream — not the final norm).
        from monai.networks.nets import DenseNet121
        return DenseNet121(spatial_dims=3, in_channels=cfg.data.num_channels,
                           out_channels=cfg.model.num_classes)
    raise ValueError(f"unknown model.arch {arch!r} (use reference|resnet|densenet)")


def checkpoint_path(cfg) -> str:
    """Per-architecture checkpoint so archs never clobber each other."""
    arch = getattr(cfg.model, "arch", "reference")
    return os.path.join(cfg.paths.processed, f"backbone_{arch}.pt")


# --------------------------- frozen wrapper + hook ---------------------------
class Backbone(nn.Module):
    """Wraps a frozen net + forward hook that captures the bottleneck activation."""

    def __init__(self, cfg):
        super().__init__()
        self.net = build_net(cfg)
        if cfg.model.source == "checkpoint":
            # model.ckpt overrides for per-dataset runs (backbone_<arch>_<dataset>.pt);
            # else the per-arch default. Backward-compatible: existing configs omit it.
            ckpt = getattr(cfg.model, "ckpt", None) or checkpoint_path(cfg)
            try:                                        # our checkpoints are plain state_dicts
                sd = torch.load(ckpt, map_location="cpu", weights_only=True)
            except Exception:                           # fall back for full/exotic checkpoints
                sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.net.load_state_dict(sd, strict=False)

        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

        self._act = {}
        dict(self.net.named_modules())[cfg.model.bottleneck].register_forward_hook(
            self._save_hook)
        self.bottleneck = cfg.model.bottleneck

    def _save_hook(self, module, inp, out):
        self._act["z"] = out            # keep graph for autograd in TCAV

    def forward(self, x, need_activation: bool = False):
        logits = self.net(x)
        if need_activation:
            return logits, self._act["z"]
        return logits


def load_backbone(cfg) -> Backbone:
    return Backbone(cfg).to(cfg.compute.device)
