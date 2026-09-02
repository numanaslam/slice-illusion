# The Slice Illusion

Code for the paper **"The Slice Illusion: Why Per-Slice Faithfulness Metrics Give
Fragile Verdicts on 3D Medical Image Classifiers"** (Aslam, Mustafa, Qureshi).

Because a 3D model integrates evidence over the whole volume, a perturbation confined
to one axial slice barely moves the prediction. The response on which per-slice (2D)
deletion and insertion metrics rest largely vanishes, the scores that remain sit at the
scale of their own sampling noise, and a random attribution can score within noise of a
principled one. This repository provides a fully **volumetric** faithfulness protocol,
the Band-K diagnostic that isolates the role of 3D context, and the multi-dataset
experiments quantifying the effect (volumetric signals 3.8 to 36.9 times the slice-wise
signal across sixteen conditions, random-effects summary 11.2x).

The importable package is `vcf3d`. Server-specific setup (CUDA-matched PyTorch, dataset
downloads) is in [`SERVER_SETUP.md`](SERVER_SETUP.md); the algorithms are documented in
[`docs/METHOD.md`](docs/METHOD.md).

## What the code does

- **Volumetric faithfulness protocol** — equal-volume 3D deletion over SLIC supervoxels
  with an in-distribution, anatomically plausible perturbation operator, and a mandatory
  random-baseline sanity check.
- **The slice illusion diagnostic** — the 2D-vs-3D faithfulness-signal gap, with a
  band-width sweep (`K = 1 … depth`) that isolates 3D context from the quantity of
  tissue removed.
- **Explainer panel** — 3D Grad-CAM, integrated gradients, per-supervoxel occlusion, a
  random control, and (on BraTS) 3D-TCAV over substructure concepts.
- **Multi-dataset generality** — a grid of four datasets (CT and MRI), two backbones, and
  two deletion operators, aggregated with a random-effects meta-analysis reported with
  Hartung-Knapp-Sidik-Jonkman intervals.

## Installation

PyTorch must be installed first, matched to your CUDA version (see
[`SERVER_SETUP.md`](SERVER_SETUP.md) step 3), then the rest:

```bash
conda env create -f environment.yml && conda activate vcf3d
# or, with pip:
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e .            # importable as `vcf3d`
```

The MedMNIST3D CT tasks additionally need `pip install medmnist` (volumes download
automatically on first use). BraTS access is described below. The pipeline runs on a
single 8 GB consumer GPU (developed on an RTX 3070).

Verify the install with the synthetic end-to-end smoke test (no dataset required):

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
pytest -q tests/test_smoke.py
```

## Datasets

| Dataset | Modality | Task | Source |
| --- | --- | --- | --- |
| BraTS | MRI (4-channel) | enhancing-tumour burden (derived binary) | Medical Segmentation Decathlon / BraTS |
| OrganMNIST3D | CT | abdominal organ (11-class) | MedMNIST v2 |
| NoduleMNIST3D | CT | lung-nodule malignancy | MedMNIST v2 (LIDC-derived) |
| FractureMNIST3D | CT | rib-fracture type | MedMNIST v2 |

The MedMNIST v2 volumes download automatically via the `medmnist` package. BraTS (through
the Medical Segmentation Decathlon) is a no-DUA quick start; see
[`SERVER_SETUP.md`](SERVER_SETUP.md) §6 for the exact download commands. Two further
MedMNIST3D sets (VesselMNIST3D, AdrenalMNIST3D) are binary shape masks and are out of scope
for an intensity-based operator; the pipeline detects and excludes them automatically.

## Reproducing the paper

```bash
# 1. Sanity checks first (a random concept must score TCAV ~ 0.5)
python -m scripts.s04_run_faithfulness  --config configs/brats_agg.yaml   # BraTS main result

# 2. Train the CT backbones (per dataset, best-val checkpoint, cosine LR)
python -m scripts.s10_train_medmnist    --config configs/medmnist.yaml

# 3. The 16-cell generality grid (4 datasets x 2 backbones x 2 operators)
python -m scripts.s11_multidataset_faithfulness --config configs/medmnist.yaml

# 4. Random-effects meta-analysis and forest plot
python -m scripts.s12_meta_analysis     --config configs/medmnist.yaml

# 5. Publication figures
python -m scripts.s09_publication_figures --config configs/rtx3070.yaml
```

Mapping of scripts to the paper's artefacts:

| Script | Produces |
| --- | --- |
| `s04_run_faithfulness.py` | BraTS main result: Table 2, Figures 2 and 3, band-$K$ sweep (Table 3) |
| `s10_train_medmnist.py` | Frozen backbones for the three CT tasks |
| `s11_multidataset_faithfulness.py` | Generality grid: Table 4 and Appendix Table A.5 |
| `s12_meta_analysis.py` | HKSJ pooled ratio and forest plot (Figure 4) |
| `s13_method_pipeline_figure.py` | Method-overview figure (Figure 1) |
| `s14_panel_sensitivity.py` | Panel x threshold sensitivity grid (Appendix C) |
| `s16_tau_sensitivity.py` | Confidence-threshold sensitivity from per-volume dumps |
| `s17_seed_variance.py` | Across-seed variance of the random-control signal (Appendix C) |
| `s18_export_visuals.py` | Slice and attribution exports for Figures 1 and 2 |
| `s09_publication_figures.py` | Camera-ready `fig_main` and `fig_analysis` |

Numbers vary at the ~1% level across GPUs from floating-point non-determinism; the ranking
of methods and the direction of every effect are stable.

## Reproducibility contract

Always run the random-concept sanity check (`vcf3d/faithfulness/sanity.py`) before reporting
any result: a random concept must score `TCAV ~ 0.5` and the random control must be the
least faithful method under the volumetric protocol. If either fails, the pipeline is
misconfigured and the numbers are not interpretable.

## Repository layout

```
vcf3d/                       importable package
├── config.py                yaml -> namespace, device resolution
├── io/                      dataset registry + loaders (brats, medmnist3d, lidc, datasets)
├── models/                  frozen 3D backbones (plain CNN / residual) + activation hooks
├── concept/                 3D-TCAV, CAVs, post-hoc V-CBM, concept discovery
├── perturb/                 R(x,Omega): anatomical replacement + surrogate tissue bank
├── faithfulness/            deletion/insertion, slice_illusion, sanity, intervention
└── utils/                   SLIC-3D supervoxels, Grad-CAM, attributions
scripts/                     s01..s12 pipeline (preprocess -> train -> faithfulness -> meta -> figures)
configs/                     brats_agg, medmnist, rtx3070, default, lidc, ...
tests/test_smoke.py          synthetic end-to-end, no dataset
```

## Citation

```bibtex
@misc{aslam2026sliceillusion,
  title  = {The Slice Illusion: Why Per-Slice Faithfulness Metrics Give Fragile
            Verdicts on 3D Medical Image Classifiers},
  author = {Aslam, Numan and Mustafa, Ghulam and Qureshi, Adnan N.},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

Code released under the MIT License (see [`LICENSE`](LICENSE)). The datasets
retain their own licences (BraTS / Medical Segmentation Decathlon and MedMNIST v2); see
their sources for terms.

## Provenance note

An early circulated draft reported a secondary-architecture result of "32.6x (n=64)" on
BraTS. That figure traced to a checkpoint state that was later overwritten and cannot be
reproduced; it was removed rather than kept. The published secondary-architecture result
(residual network, 34.6x, n=71) regenerates deterministically from this repository, as
does every other number in the paper.
