"""Pipeline-style method-overview figure (fig_method) for the slice-illusion paper.

Left-to-right pipeline in the style of classic architecture diagrams:
input volume -> frozen 3D CNN -> explainer panel -> supervoxel mapping ->
deletion-evaluation module (with the Band-K knob) -> volumetric vs slice-wise
outcome curves -> faithfulness-signal verdict.  Vector PDF + PNG preview.

    python -m scripts.s13_method_pipeline_figure
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

# palette (Okabe-Ito accents + navy/soft fills like the reference style)
NAVY = "#1F3B73"; SLATE = "#2F4B6E"; TEAL = "#0072B2"; SKY = "#56B4E9"
ORANGE = "#E69F00"; GREEN = "#009E73"; VERM = "#D55E00"; GREY = "#5B6470"
PANEL = "#FBE9D6"; PALE = "#F3F7FC"; INK = "#1a2433"

W, H = 100.0, 43.0
fig = plt.figure(figsize=(10.0, 4.3))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
plt.rcParams["font.family"] = "DejaVu Sans"


def rbox(x, y, w, h, fc, ec=SLATE, lw=1.1, r=0.6, ls="-", z=3):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.12,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z)
    ax.add_patch(p); return p


def txt(x, y, s, size=8.5, color=INK, ha="center", va="center", w="normal", rot=0, z=6):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight=w, rotation=rot, zorder=z)


def arrow(x0, y0, x1, y1, color=INK, lw=1.6, rad=0.0, z=5, style="-|>", ms=11):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=ms, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z))


def label_box(x, y, w, s, size=8.2):
    rbox(x, y, w, 4.6, "white", ec=INK, lw=0.9, r=0.25)
    txt(x + w / 2, y + 2.3, s, size=size)


def cuboid(x, y, w, h, d, fc, z=4):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="white", lw=0.8, zorder=z))
    ax.add_patch(Polygon([(x, y + h), (x + d, y + h + d), (x + w + d, y + h + d),
                          (x + w, y + h)], fc=fc, ec="white", lw=0.8, zorder=z))
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d),
                          (x + w, y + h)], fc=fc, ec="white", lw=0.8, zorder=z))


FLOW = 27.0          # main flow centreline
LBL = 6.2            # label-row bottom

# --- A. input volume (stack of slices) --------------------------------------
VIS = np.load("/Users/numanaslam/Desktop/paper5/vcf3d/experiments/results/visual_export.npz")


def _norm(a):
    a = a - a.min()
    m = a.max()
    return a / m if m > 0 else a


_anat = VIS["BRATS_006_anat"]                       # (3, H, W) real axial FLAIR slices
for i in range(3):
    axi = fig.add_axes([0.028 + 0.010 * i, 0.465 + 0.056 * i, 0.088, 0.205])
    _sl = np.rot90(_anat[i])
    _img = np.where(_sl != 0, _norm(_sl), 1.0)          # white outside the brain
    axi.imshow(_img, cmap="gray", vmin=0, vmax=1)
    axi.set_xticks([]); axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_color("#4a6fa5"); sp.set_linewidth(0.9)
txt(7.8, 18.6, "$x \\in \\mathbb{R}^{C\\times H\\times W\\times D}$", size=7.6, color=GREY)
label_box(1.8, LBL, 12.6, "Input: 3D volume\n(CT / MRI)")
arrow(13.3, FLOW, 16.2, FLOW)

# --- B. frozen 3D CNN --------------------------------------------------------
bx = [17.0, 21.0, 24.6, 27.8]; bw = [3.4, 3.0, 2.6, 2.2]; bh = [7.0, 5.8, 4.8, 3.9]
for i, (x, w, h) in enumerate(zip(bx, bw, bh)):
    fc = "#159085" if i == 2 else ["#22456e", "#2d5586", "#3b689e", "#4d7cb0"][i]
    cuboid(x, FLOW - h / 2, w, h, 0.85, fc)
    txt(x + w / 2 + 0.4, FLOW - h / 2 - 1.5, f"enc{i+1}", size=7.2, color=GREY)
txt(25.9, FLOW + 4.6, "bottleneck", size=7.2, color="#159085", w="bold")
ax.plot([25.9, 25.9], [FLOW + 2.6, FLOW + 3.8], color="#159085", lw=0.9, zorder=5)
rbox(31.2, FLOW - 1.7, 5.2, 3.4, PALE, ec=SLATE, lw=1.0)
txt(33.8, FLOW, "GAP + fc", size=7.4)
arrow(30.6, FLOW, 31.1, FLOW, lw=1.2, ms=8)
label_box(16.2, LBL, 20.4, "Frozen 3D classifier $f$\n(explained, never retrained)")
arrow(36.7, FLOW, 38.6, FLOW)

# --- C. explainer panel (pills) ----------------------------------------------
pills = [("Grad-CAM", TEAL, "white"), ("Integrated gradients", SKY, "white"),
         ("3D-TCAV", ORANGE, "white"), ("Occlusion", GREEN, "white"),
         ("Random control", "white", GREY)]
for i, (name, fc, tc) in enumerate(pills):
    y = 35.6 - 3.95 * i
    rbox(38.8, y - 1.55, 12.8, 3.1, fc, ec=GREY if fc == "white" else fc,
         lw=1.1, ls="--" if fc == "white" else "-", r=1.2)
    txt(45.2, y, name, size=7.3, color=tc, w="bold" if fc != "white" else "normal")
    ax.plot([51.8, 53.1], [y, y], color=GREY, lw=0.9, zorder=2)
# example attribution overlays (mid slice): a real method and the random control
_sl = np.rot90(_anat[1])
_brain = _sl != 0
_mid = np.where(_brain, _norm(_sl), 1.0)                # white outside the brain
for j, (key, ttl) in enumerate([("Grad-CAM", "Grad-CAM"), ("random", "random")]):
    key = "gradcam" if j == 0 else "random"
    h = np.rot90(_norm(VIS[f"BRATS_006_{key}"][1]))
    thr = np.percentile(h[_brain], 70)
    axi = fig.add_axes([0.394 + 0.063 * j, 0.262, 0.056, 0.120])
    axi.imshow(_mid, cmap="gray", vmin=0, vmax=1)
    axi.imshow(np.ma.masked_where((h < thr) | ~_brain, h), cmap="inferno", alpha=0.65)
    axi.set_xticks([]); axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_color("#c8cdd6"); sp.set_linewidth(0.7)
    axi.set_title(ttl, fontsize=5.6, color=GREY, pad=1.5)
label_box(38.9, LBL, 19.5, "Explainer panel: attribution $\\phi$\nper method, on SLIC supervoxels", size=7.6)

# --- D. supervoxel bar -------------------------------------------------------
rbox(53.2, 15.2, 3.4, 23.2, NAVY, ec=NAVY, r=1.0)
txt(54.9, 26.8, "map $\\phi$ onto ${\\approx}$300 SLIC supervoxels", size=8.4,
    color="white", rot=90, w="bold")
arrow(56.8, FLOW, 59.0, FLOW)

# --- E. deletion-evaluation module ------------------------------------------
rbox(59.6, 12.0, 17.6, 27.6, PANEL, ec=SLATE, lw=1.4, r=1.0, z=2)
txt(68.4, 37.6, "One deletion evaluation", size=9.2, w="bold")
txt(68.4, 35.9, "(Algorithm 1)", size=7.4, color=GREY)
steps = [("next equal-volume bin $c_t$", "white", SLATE),
         ("confine to band of $K$\naxial slices", "#EAF6EC", GREEN),
         ("in-distribution fill $R(x,\\Omega)$:\ncontext-NN / ROAD", "white", SLATE),
         ("forward pass $p^{(t)} = f_k(x')$", "white", SLATE)]
ys = [14.0, 19.1, 24.2, 30.4]
hs = [3.4, 3.4, 4.6, 3.4]
for (s, fc, ec), y, h in zip(steps, ys, hs):
    rbox(60.6, y, 13.2, h, fc, ec=ec, lw=1.2)
    txt(67.2, y + h / 2, s, size=7.1)
for y0, y1 in [(17.4, 19.0), (22.5, 24.1), (28.8, 30.3)]:
    arrow(67.4, y0 + 0.1, 67.4, y1, lw=1.2, ms=8)
arrow(74.1, 31.9, 74.1, 16.0, rad=-0.20, lw=1.1, ms=8, color=GREY)
txt(67.2, 12.95, "repeat for $t = 1, \\dots, T$", size=6.3, color=GREY)
label_box(59.6, LBL, 17.6, "Band-$K$ diagnostic: $K$ slides the\nprotocol from 2D to 3D", size=7.6)

# --- F. outcome mini-plots ---------------------------------------------------
def mini(rect, border, title, collapse):
    a = fig.add_axes(rect); xs = np.linspace(0, 1, 60)
    curves = [(TEAL, 0.95 * np.exp(-2.8 * xs) + 0.03, "-"),
              (ORANGE, 0.95 * np.exp(-1.8 * xs) + 0.05, "-"),
              (GREEN, 0.95 * np.exp(-1.5 * xs) + 0.08, "-"),
              (SKY, 0.95 * np.exp(-1.1 * xs) + 0.11, "-"),
              (GREY, 0.97 - 0.34 * xs, (0, (4, 2)))]
    if collapse:
        # nearly tied, but individually visible; random interleaved among real methods
        curves = [(SKY, 0.955 - 0.022 * xs, "-"),
                  (GREY, 0.930 - 0.025 * xs, (0, (4, 2))),
                  (TEAL, 0.905 - 0.028 * xs, "-"),
                  (ORANGE, 0.880 - 0.024 * xs, "-"),
                  (GREEN, 0.855 - 0.026 * xs, "-")]
    for c, y, ls in curves:
        a.plot(xs, y, color=c, lw=1.4, ls=ls)
    a.set_xlim(0, 1); a.set_ylim(0, 1.05)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color(border); s.set_linewidth(1.4)
    a.set_title(title, fontsize=8.0, color=border, fontweight="bold", pad=2.5)
    a.set_ylabel("$f_k(x')$", fontsize=6.4, labelpad=1.0)
    return a

top = mini([0.815, 0.615, 0.165, 0.265], TEAL, "Volumetric ($K{=}D$)", False)
bot = mini([0.815, 0.295, 0.165, 0.265], VERM, "Slice-wise ($K{=}1$)", True)
bot.set_xlabel("fraction removed", fontsize=6.4, labelpad=1.5)
arrow(77.5, 31.6, 80.7, 32.4, lw=1.4); txt(79.0, 33.6, "$K{=}D$", size=7.0, color=TEAL, w="bold")
arrow(77.5, 18.4, 80.7, 17.6, lw=1.4); txt(79.0, 20.0, "$K{=}1$", size=7.0, color=VERM, w="bold")
label_box(79.0, LBL, 19.4,
          "signal = AUC(random) $-$ AUC(best):\nlarge in 3D; noise-scale in 2D",
          size=7.4)


out = "/Users/numanaslam/Desktop/paper5/vcf3d/docs/paper/figures/fig_method_v2"
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=220)
print("wrote", out + ".pdf/.png")
