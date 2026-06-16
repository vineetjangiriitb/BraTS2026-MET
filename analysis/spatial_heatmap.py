#!/usr/bin/env python3
"""
A3 — Spatial heatmap of lesion locations (SRI24 atlas cases only).

Only the non-UCSD cases live in the common SRI24 atlas space (240x240x155,
1mm isotropic), so masks are voxel-comparable and can be summed into a 3D
lesion-probability map. UCSD cases are native-space and are excluded.

Outputs:
  outputs/et_heatmap.nii.gz        — summed ET (label 3) occurrence count, SRI24 space
  outputs/wt_heatmap.nii.gz        — summed whole-tumor (any label) occurrence
  outputs/spatial_heatmap.png      — axial/coronal/sagittal max-intensity projections
"""
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "training", "MICCAI-LH-BraTS2025-MET-Challenge-Training")
OUT = os.path.join(ROOT, "analysis", "outputs")
os.makedirs(OUT, exist_ok=True)

SRI24_SHAPE = (240, 240, 155)
ET_LABEL = 3


def sri24_cases():
    for name in sorted(os.listdir(TRAIN)):
        if name.startswith(".") or name == "UCSD-Training":
            continue
        path = os.path.join(TRAIN, name)
        if os.path.isdir(path):
            yield name, path


def main():
    et_heat = np.zeros(SRI24_SHAPE, dtype=np.int32)
    wt_heat = np.zeros(SRI24_SHAPE, dtype=np.int32)
    n = 0
    ref_affine = None
    skipped = []

    for cid, path in sri24_cases():
        seg_file = os.path.join(path, f"{cid}-seg.nii.gz")
        if not os.path.exists(seg_file):
            continue
        img = nib.load(seg_file)
        if tuple(img.shape) != SRI24_SHAPE:
            skipped.append((cid, tuple(img.shape)))
            continue
        if ref_affine is None:
            ref_affine = img.affine
        seg = np.asarray(img.dataobj).astype(np.int16)
        et_heat += (seg == ET_LABEL).astype(np.int32)
        wt_heat += (seg > 0).astype(np.int32)
        n += 1
        if n % 100 == 0:
            print(f"  ...accumulated {n}")

    print(f"Accumulated {n} SRI24 cases. Skipped {len(skipped)} (off-grid shapes).")
    if skipped:
        print("  skipped:", skipped[:10])

    nib.save(nib.Nifti1Image(et_heat, ref_affine), os.path.join(OUT, "et_heatmap.nii.gz"))
    nib.save(nib.Nifti1Image(wt_heat, ref_affine), os.path.join(OUT, "wt_heatmap.nii.gz"))

    # MIP visualization (probability = occurrence / n)
    et_prob = et_heat / max(n, 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    views = [
        ("Axial (Z MIP)", lambda v: v.max(axis=2).T),
        ("Coronal (Y MIP)", lambda v: v.max(axis=1).T),
        ("Sagittal (X MIP)", lambda v: v.max(axis=0).T),
    ]
    for j, (title, proj) in enumerate(views):
        im0 = axes[0, j].imshow(proj(et_prob), origin="lower", cmap="hot")
        axes[0, j].set_title(f"ET prob — {title}")
        axes[0, j].axis("off")
        plt.colorbar(im0, ax=axes[0, j], fraction=0.046)
        im1 = axes[1, j].imshow(proj(wt_heat / max(n, 1)), origin="lower", cmap="viridis")
        axes[1, j].set_title(f"WT prob — {title}")
        axes[1, j].axis("off")
        plt.colorbar(im1, ax=axes[1, j], fraction=0.046)
    fig.suptitle(f"Lesion spatial distribution — {n} SRI24 atlas cases", fontsize=14)
    plt.tight_layout()
    out_png = os.path.join(OUT, "spatial_heatmap.png")
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"Saved {out_png}")
    print(f"Saved et_heatmap.nii.gz / wt_heatmap.nii.gz to {OUT}/")
    print(f"Peak ET co-occurrence: {et_heat.max()} cases at one voxel "
          f"({100*et_heat.max()/max(n,1):.1f}% of cases)")


if __name__ == "__main__":
    main()
