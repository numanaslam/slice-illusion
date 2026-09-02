# Detailed Method — The Two Novel Pieces

This document specifies the algorithms implemented in `src/`. Both pieces operate on a
**frozen** 3D backbone `f` (no backbone training). Notation:

- `x ∈ R^{H×W×D×C_in}` — an input volume (C_in modalities/channels, e.g. 4 for BraTS MRI).
- `a_l(x) ∈ R^{H'×W'×D'×K}` — activations at bottleneck layer `l` (K channels, 3D spatial grid).
- `φ_l(x) = GAP3D(a_l(x)) ∈ R^K` — global-average-pooled feature vector over the 3D grid.
- `h_{l,k}(·)` — the network head mapping layer-`l` activations to the logit of class `k`.

---

## Piece 1 — Volumetric Concept-Based Explanation

### 1.1 Volumetric Concept Activation Vectors (3D-TCAV)

**Concept definition.** A concept `C` is given by a set of *positive sub-volume examples*
`P = {p_i}` (3D patches exhibiting the concept, e.g. necrotic-core patches) and a set of
*random negatives* `N = {q_j}` sampled from anatomy without the concept.

**CAV learning.** Extract pooled features `φ_l(p_i)` and `φ_l(q_j)`, then fit a linear
classifier (logistic regression / linear SVM) separating positives from negatives in
activation space. The **CAV** `v_C^l ∈ R^K` is the unit normal to the decision hyperplane:

```
v_C^l = w / ||w||,   where  w = argmin_w  Σ loss(⟨w, φ_l⟩ + b, y) + λ||w||²
```

**Volumetric conceptual sensitivity (the 3D nuance).** For input `x` and class `k`, the
per-voxel directional derivative of the logit along the concept direction is

```
S_C,k(x)[u,v,d] = ⟨ ∂ h_{l,k}(a_l(x)) / ∂ a_l(x)[u,v,d,:] ,  v_C^l ⟩
```

computed by reverse-mode autodiff (`dlgradient`) w.r.t. the **activation tensor**. The
scalar volumetric sensitivity aggregates over the *3D* spatial grid (preserving through-plane
context), rather than treating slices independently:

```
S_C,k(x) = AGG_{u,v,d} S_C,k(x)[u,v,d]        (AGG = mean or sum)
```

**TCAV score.** Over inputs `X_k` of class `k`:

```
TCAV_C,k = (1/|X_k|) Σ_{x∈X_k} 1[ S_C,k(x) > 0 ]
```

**Statistical validity.** Learn `M` CAVs from `M` random negative draws; report the mean
TCAV and a two-sided t-test against the null distribution from *random* concepts
(`sanityCheck.m`). A valid concept must be separable from random with `TCAV ≠ 0.5`.

**Why this is 3D-novel.** The CAV is learned on volumetric pooled features and the sensitivity
is aggregated over the full 3D receptive field. The *slice-wise* baseline learns a CAV per 2D
slice and averages — discarding through-plane structure. The divergence between the two is the
subject of the slice-illusion experiment (Piece 2.3).

### 1.2 Post-hoc Volumetric Concept Bottleneck (V-CBM)

Given a bank of CAVs `V = [v_{C_1}, …, v_{C_m}]`, project frozen features to a concept-score
vector and fit a sparse interpretable head by **distillation** of the frozen model:

```
c(x) = Vᵀ φ_l(x) ∈ R^m           (concept activations)
ŷ    = g_θ(c(x))                  (sparse linear head)
θ*   = argmin_θ  Σ_x KL( softmax(f(x)) || softmax(g_θ(c(x))) ) + β||θ||_1
```

The bottleneck `c(x)` is the explanation; `g_θ` gives per-concept contributions
`θ_{k,C} · c_C(x)` to class `k`. Faithfulness is measured by **inference-time intervention**
(Piece 2.2): forcing `c_C → 0` must move the prediction by an amount consistent with
`θ_{k,C}`.

### 1.3 Unsupervised volumetric concept discovery (optional, `conceptDiscovery.m`)

When concept labels are absent: over-segment volumes into 3D supervoxels (SLIC-3D,
`superVoxels.m`), embed each supervoxel with `φ_l`, cluster (k-means) across a corpus; each
cluster is a candidate concept whose members become the positive set `P`. This is the 3D
analogue of ACE.

---

## Piece 2 — Anatomically-Plausible 3D Faithfulness Protocol

### 2.1 The anatomical replacement operator `R`

**Problem.** Deletion/insertion faithfulness requires "removing" a region's information.
Zero/mean masking a sub-volume `Ω` yields an anatomically impossible volume → the prediction
drop is confounded by pure out-of-distribution (OOD) artifact.

**Operator.** With a soft mask `M_Ω` (binary mask dilated + Gaussian-feathered at the boundary
to suppress seams):

```
R(x, Ω) = x ⊙ (1 − M_Ω) + s_Ω ⊙ M_Ω
```

where `s_Ω` is an **anatomically-consistent surrogate** for the region. Three surrogate modes
(`anatomicalReplace.m`):

1. **`tissue-mean`** — replace with the intensity distribution of the *same tissue class*
   (from a `buildTissueBank` over held-out healthy anatomy; tissue class from
   TotalSegmentator labels or intensity GMM). Cheap, in-distribution baseline.
2. **`context-nn`** — nearest-neighbor healthy patch matched on the *surrounding ring* of `Ω`
   (context intensity statistics + position), blended in. Best in-distribution realism.
3. **`zero` / `mean`** — classical baselines, included *only* to quantify the OOD artifact.

`R` must satisfy an **in-distribution check**: the backbone's feature distribution of `R(x,Ω)`
over *unimportant* `Ω` should be statistically indistinguishable from unperturbed `x`
(reported as a control).

### 2.2 Volumetric deletion / insertion and inference-time intervention

**Supervoxel ranking.** Partition `x` into 3D supervoxels `{Ω_1,…,Ω_S}`. Score each by an
explanation method (3D-TCAV sensitivity, V-CBM contribution, or a saliency baseline).

**Deletion AUC.** Progressively apply `R` to supervoxels in *descending* importance; record
`p_k(x)` after each step; AUC of the probability-vs-fraction-removed curve. Faithful → steep
early drop → low AUC.

**Insertion AUC.** Start from a fully-replaced volume `R(x, all)`; progressively *restore*
supervoxels in descending importance; AUC. Faithful → steep early rise → high AUC.

**Inference-time intervention (for V-CBM).** Set concept score `c_C(x) → 0` and measure
`Δlogit_k = h_k(c) − h_k(c | c_C=0)`; a faithful concept yields `Δlogit_k ≈ θ_{k,C} c_C(x)`.

### 2.3 The Slice-Illusion diagnosis (headline experiment)

For each explanation method compute faithfulness **two ways**:

- **Volumetric (V):** rank & perturb 3D supervoxels with `R` (§2.2).
- **Slice-wise (S):** run standard 2D deletion/insertion independently per axial slice, then
  aggregate slice AUCs.

Report, across methods × datasets:

- **Kendall's τ** between the explainer *rankings* induced by V vs S. Low τ ⇒ the protocols
  disagree about which explanation is more faithful.
- **Verdict-flip rate:** fraction of method pairs `(A,B)` where `faith(A) > faith(B)` under S
  but reverses under V.

A high flip rate is the paper's core evidence that **2D faithfulness tools mislead on 3D
models.**

### 2.4 Perturbation-operator study

Repeat §2.2 under each surrogate mode (`zero`, `mean`, `tissue-mean`, `context-nn`) and
measure how much of the faithfulness signal is attributable to the OOD artifact
(difference between `zero` and `context-nn`). This isolates the methodological claim that the
masking operator, not the explanation, drives naive 3D faithfulness scores.

---

## Reproducibility contract

- Backbone weights frozen and hashed; CAV training seeds logged.
- All faithfulness runs report mean ± CI over `M` random-negative CAV draws and over the
  evaluation cohort.
- Random-concept sanity check must pass (`TCAV ≈ 0.5`, faithfulness ≈ chance) before any
  positive result is reported.
