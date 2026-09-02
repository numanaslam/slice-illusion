# Bucket C runs (reviewer items 4 and 5): panel and tau sensitivity

Files in this drop (unpack over the repo on the GPU box):

| File | What changed |
| --- | --- |
| `scripts/s14_panel_sensitivity.py` | NEW. One GPU pass over the BraTS eval cohort computes every attribution (Grad-CAM, IG, occlusion, all TCAV concepts, random) per volume with no confidence gate, then assembles every panel x tau combination post hoc. |
| `scripts/s16_tau_sensitivity.py` | NEW. CPU-only aggregator: signal/ratio vs tau from any per-volume dump. |
| `scripts/s11_multidataset_faithfulness.py` | PATCHED. `--dump_pervol` writes per-volume records so grid cells become tau-assemblable too. |
| `scripts/s12_meta_analysis.py` | PATCHED. Forest plot now prints proper names (Residual / Plain CNN / ROAD) and the HKSJ pooled interval. Already run locally; included so the server copy stays in sync. |

## Run 1 (the important one): BraTS panel x tau sensitivity

```
conda activate vcf3d
python -m scripts.s14_panel_sensitivity --config configs/brats_agg.yaml
```

Overnight-class on the RTX 3070 (occlusion is ~300 forward passes per volume on top of
the deletion runs). Crash-safe and resumable: rerunning skips volumes already recorded in
`experiments/results/panel_pervol_resnet_context-nn.csv`.

What it answers, in one file (`experiments/results/panel_sensitivity.csv`):
- `concept` panel at tau=0.60  -> should reproduce ~46.9x (the Table 2 cell)
- `common` panel at tau=0.50   -> should land near ~20.8x (the grid cell)
- `common` at 0.60 and `concept` at 0.50 -> splits the difference into its panel part
  and its cohort part
- `union` and `only-*` panels across five taus -> the full sensitivity surface

## Run 2 (optional, adds the ROAD operator to the surface)

```
python -m scripts.s14_panel_sensitivity --config configs/brats_agg.yaml --operator road
```

## Run 3 (optional, isolates the checkpoint factor)

The s04 analysis used the `brats_agg.yaml` checkpoint; the grid used the s10 backbone.
If those files differ, this quantifies how much that matters:

```
python -m scripts.s14_panel_sensitivity --config configs/brats_agg.yaml \
    --ckpt <cfg.paths.processed>/backbone_resnet_brats.pt --tag gridckpt
```

## Run 4 (optional, tau sensitivity for the CT grid cells)

Re-runs grid cells with per-volume dumps (each cell costs what it cost in the original
grid run; pick the cells you care about rather than all sixteen):

```
python -m scripts.s11_multidataset_faithfulness --config configs/medmnist.yaml \
    --datasets nodulemnist3d organmnist3d fracturemnist3d \
    --archs resnet --operators context-nn --max_eval 60 --dump_pervol
python -m scripts.s16_tau_sensitivity --pervol "experiments/results/pervol_*.csv"
```

## Send back

Everything matching:

```
experiments/results/panel_pervol_*.csv
experiments/results/panel_sensitivity.csv
experiments/results/pervol_*.csv          (only if Run 4 was done)
experiments/results/tau_sensitivity.csv   (only if Run 4 was done)
```

Drop them into `vcf3d/experiments/results/` on the laptop as before. The paper edits
(sensitivity table or appendix, plus the reconciliation footnote upgrade) happen locally
from those files; no further GPU work needed.
