"""
Generate side-by-side visualization panels for all 20 validation cases.
Each panel row: Original t1c | Ground Truth overlay | Prediction overlay
Shows the axial slice with the most tumor voxels for each case.
Output: analysis/outputs/validation_panels.png
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
TRAINING = PROJECT / "training" / "MICCAI-LH-BraTS2025-MET-Challenge-Training"
PRED_DIR = (
    PROJECT
    / "runpod_results"
    / "nnUNet_results"
    / "Dataset001_BraTsMET"
    / "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
    / "fold_0"
    / "validation"
)
OUT_DIR = PROJECT / "analysis" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Label colours (RGBA 0-1): 1=NETC red, 2=SNFH yellow, 3=ET green ──────────
LABEL_COLORS = {
    1: (1.0, 0.2, 0.2, 0.55),   # NETC — red
    2: (1.0, 0.85, 0.2, 0.45),  # SNFH — yellow
    3: (0.2, 0.8, 0.2, 0.65),   # ET   — green
}
LABEL_NAMES = {1: "NETC", 2: "SNFH", 3: "ET"}


def best_slice(seg: np.ndarray) -> int:
    """Return the axial index (axis=2) with the most non-zero label voxels."""
    counts = np.count_nonzero(seg, axis=(0, 1))
    return int(np.argmax(counts))


def norm(vol: np.ndarray) -> np.ndarray:
    """Clip to [1,99] percentile then normalise to [0,1]."""
    lo, hi = np.percentile(vol[vol > 0], [1, 99]) if vol.max() > 0 else (0, 1)
    return np.clip((vol.astype(float) - lo) / max(hi - lo, 1e-6), 0, 1)


def overlay(ax, img_slice: np.ndarray, seg_slice: np.ndarray, title: str):
    """Plot grayscale image with coloured segmentation overlay."""
    ax.imshow(img_slice.T, cmap="gray", origin="lower", interpolation="nearest")
    rgba = np.zeros((*img_slice.shape, 4))
    for label, color in LABEL_COLORS.items():
        mask = seg_slice == label
        rgba[mask] = color
    ax.imshow(rgba.transpose(1, 0, 2), origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=8, pad=3)
    ax.axis("off")


def plain(ax, img_slice: np.ndarray, title: str):
    """Plot grayscale image with no overlay."""
    ax.imshow(img_slice.T, cmap="gray", origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=8, pad=3)
    ax.axis("off")


# ── Collect the 20 cases ───────────────────────────────────────────────────────
pred_files = sorted(PRED_DIR.glob("*.nii.gz"))
cases = [p.stem.replace(".nii", "") for p in pred_files]  # strip .nii.gz
print(f"Found {len(cases)} prediction files.")

# ── Build figure: 20 rows × 3 columns ─────────────────────────────────────────
N = len(cases)
fig, axes = plt.subplots(N, 3, figsize=(9, 3.2 * N))
fig.patch.set_facecolor("#111111")

legend_patches = [
    mpatches.Patch(color=LABEL_COLORS[1][:3], label="NETC (1)"),
    mpatches.Patch(color=LABEL_COLORS[2][:3], label="SNFH (2)"),
    mpatches.Patch(color=LABEL_COLORS[3][:3], label="ET (3)"),
]

for row, case_id in enumerate(cases):
    case_dir = TRAINING / case_id
    t1c_path = case_dir / f"{case_id}-t1c.nii.gz"
    gt_path  = case_dir / f"{case_id}-seg.nii.gz"
    pred_path = PRED_DIR / f"{case_id}.nii.gz"

    missing = [p for p in [t1c_path, gt_path, pred_path] if not p.exists()]
    if missing:
        print(f"  SKIP {case_id}: missing {[str(m) for m in missing]}")
        for ax in axes[row]:
            ax.set_visible(False)
        continue

    t1c  = nib.load(t1c_path).get_fdata()
    gt   = nib.load(gt_path).get_fdata().astype(np.uint8)
    pred = nib.load(pred_path).get_fdata().astype(np.uint8)

    t1c_n = norm(t1c)
    # Use GT to find the most informative slice (fall back to pred if GT empty)
    sl = best_slice(gt) if gt.max() > 0 else best_slice(pred)

    short = case_id.replace("BraTS-MET-", "")  # e.g. "00025-000"

    plain(  axes[row, 0], t1c_n[:, :, sl], f"{short}\nOriginal (t1c) z={sl}")
    overlay(axes[row, 1], t1c_n[:, :, sl], gt[:, :, sl],   "Ground Truth")
    overlay(axes[row, 2], t1c_n[:, :, sl], pred[:, :, sl], "Prediction")

    print(f"  {case_id}  slice={sl}  GT labels={np.unique(gt)}  Pred labels={np.unique(pred)}")

# Global legend at the bottom
fig.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=3,
    fontsize=10,
    facecolor="#222222",
    labelcolor="white",
    framealpha=0.8,
    bbox_to_anchor=(0.5, 0.0),
)
fig.suptitle(
    "nnU-Net fold-0 validation — 20 cases\nOriginal  |  Ground Truth  |  Prediction",
    color="white",
    fontsize=13,
    y=1.001,
)
plt.tight_layout(rect=[0, 0.01, 1, 1])

out_path = OUT_DIR / "validation_panels.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved → {out_path}")
