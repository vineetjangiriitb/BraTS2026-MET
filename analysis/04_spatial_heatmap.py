#!/usr/bin/env python3
"""
Script 4 — Spatial tumor-frequency heatmap (SRI24 atlas cases ONLY).

Only the 240x240x155 @ 1mm SRI24 cases are in a common coordinate space, so
their binary tumor masks (any label 1/2/3) are voxel-comparable and can be
summed into a frequency map; UCSD / native-space cases are excluded.

The whole-tumor frequency map was already accumulated by spatial_heatmap.py
(one binary "any label > 0" mask per SRI24 case, summed) and saved as
outputs/wt_heatmap.nii.gz. This script consumes that array, normalizes it to a
0-1 probability map, and renders the requested figures. N = number of 240^3
cases (read from per_case_records.json).

Outputs:
  outputs/spatial_heatmap_axial.png        — 6 axial slices (z=40,60,80,100,120,140)
  outputs/spatial_heatmap_projections.png  — sagittal + coronal MIPs
  outputs/spatial_heatmap_representative.png— single most-representative axial slice
"""
import os
import json

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "outputs")
RECORDS = os.path.join(OUT, "per_case_records.json")
WT_HEATMAP = os.path.join(OUT, "wt_heatmap.nii.gz")
SRI24_SHAPE = (240, 240, 155)
AXIAL_SLICES = [40, 60, 80, 100, 120, 140]


def main():
    with open(RECORDS) as f:
        records = json.load(f)
    n = sum(1 for r in records if tuple(r["shape"]) == SRI24_SHAPE)

    freq = np.asarray(nib.load(WT_HEATMAP).dataobj).astype(np.float64)
    prob = freq / max(n, 1)  # 0..1 probability map
    print("=" * 60)
    print("SPATIAL TUMOR-FREQUENCY HEATMAP (SRI24 atlas, any tumor label)")
    print("=" * 60)
    print(f"SRI24 cases (240x240x155): {n}")
    print(f"Peak co-occurrence: {int(freq.max())} cases "
          f"({100*prob.max():.1f}%) at the hottest voxel")

    # ---------- (a) 6 axial slices with hot overlay ----------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    vmax = prob.max()
    for ax, z in zip(axes.ravel(), AXIAL_SLICES):
        sl = prob[:, :, z].T  # (y,x) -> display
        im = ax.imshow(sl, origin="lower", cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(f"Axial z={z}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Tumor occurrence probability — {n} SRI24 cases", fontsize=15)
    plt.tight_layout()
    p_axial = os.path.join(OUT, "spatial_heatmap_axial.png")
    plt.savefig(p_axial, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---------- (b) sagittal + coronal MIPs ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    sag = prob.max(axis=0).T   # X-MIP -> (z,y)
    cor = prob.max(axis=1).T   # Y-MIP -> (z,x)
    for ax, img, title in [(axes[0], sag, "Sagittal MIP (max over X)"),
                           (axes[1], cor, "Coronal MIP (max over Y)")]:
        im = ax.imshow(img, origin="lower", cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Tumor probability max-intensity projections — {n} SRI24 cases",
                 fontsize=15)
    plt.tight_layout()
    p_proj = os.path.join(OUT, "spatial_heatmap_projections.png")
    plt.savefig(p_proj, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---------- (c) most-representative axial slice ----------
    # z-level with highest mean tumor frequency (over voxels that ever have tumor)
    per_z_mean = prob.mean(axis=(0, 1))
    best_z = int(per_z_mean.argmax())
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(prob[:, :, best_z].T, origin="lower", cmap="hot",
                   vmin=0, vmax=vmax)
    ax.set_title(f"Most representative axial slice: z={best_z}\n"
                 f"(highest mean tumor frequency)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    p_rep = os.path.join(OUT, "spatial_heatmap_representative.png")
    plt.savefig(p_rep, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Most representative axial slice: z={best_z} "
          f"(mean prob {per_z_mean[best_z]:.4f})")

    # ---------- top-5 most common tumor locations ----------
    flat_idx = np.argsort(freq.ravel())[::-1][:5]
    coords = np.unravel_index(flat_idx, freq.shape)
    print("\nTop 5 most common tumor voxel locations (SRI24 voxel coords x,y,z):")
    for rank in range(5):
        x, y, z = int(coords[0][rank]), int(coords[1][rank]), int(coords[2][rank])
        c = int(freq[x, y, z])
        print(f"  #{rank+1}: (x={x:3d}, y={y:3d}, z={z:3d})  "
              f"{c} cases ({100*c/n:.1f}%)  -> {region_hint(x, y, z)}")

    print(f"\nSaved {p_axial}")
    print(f"Saved {p_proj}")
    print(f"Saved {p_rep}")


def region_hint(x, y, z):
    """Very rough anatomical hint from SRI24 voxel coords (240x240x155, 1mm,
    approx center 120,120,77). Heuristic only — not an atlas lookup."""
    cx, cy, cz = 120, 120, 77
    lr = "left" if x < cx else "right"
    ap = "anterior" if y > cy else "posterior"
    si = "superior" if z > cz else "inferior"
    return f"{si} {ap} {lr} (rough)"


if __name__ == "__main__":
    main()
