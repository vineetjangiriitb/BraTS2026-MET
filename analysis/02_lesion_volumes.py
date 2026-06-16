#!/usr/bin/env python3
"""
Script 2 — Per-lesion ET volume distribution and zone classification.

A lesion = one connected component of the ET mask (label 3). Volumes were
computed in dataset_stats.py with each case's OWN header voxel spacing (mm^3) —
essential because UCSD cases are native-space anisotropic (~0.24 mm^3/voxel),
while SRI24 cases are 1 mm^3. Using a fixed 1 mm^3 would inflate UCSD volumes
~4x and corrupt the 27/275 mm^3 thresholds. The per-lesion volumes live in
outputs/per_case_records.json (key "lesion_volumes"); this script consumes them.

Zones (per individual lesion):
  Detection only   : < 27   mm^3
  Middle zone      : 27-275 mm^3   (inclusive of both bounds)
  Segmentation     : > 275  mm^3   (Dice-evaluated)

Outputs:
  outputs/lesion_volume_distribution.png   (log-x histogram + 27/275 lines)
  outputs/lesion_volumes.csv               (case_id, lesion_id, volume_mm3, zone)
"""
import os
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "outputs")
RECORDS = os.path.join(OUT, "per_case_records.json")

VOL_SMALL = 27.0
VOL_MID = 275.0


def zone_of(v):
    if v < VOL_SMALL:
        return "detection_only"
    if v <= VOL_MID:
        return "middle_zone"
    return "segmentation_zone"


def main():
    with open(RECORDS) as f:
        records = json.load(f)

    rows = []
    all_vols = []
    for r in records:
        for i, v in enumerate(r["lesion_volumes"], start=1):
            rows.append((r["case_id"], i, v, zone_of(v)))
            all_vols.append(v)

    v = np.array(all_vols, dtype=float)
    n = len(v)

    # ---- per-lesion CSV ----
    out_csv = os.path.join(OUT, "lesion_volumes.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "lesion_id", "volume_mm3", "zone"])
        for cid, lid, vol, z in rows:
            w.writerow([cid, lid, round(vol, 2), z])

    # ---- zone counts ----
    small = int((v < VOL_SMALL).sum())
    mid = int(((v >= VOL_SMALL) & (v <= VOL_MID)).sum())
    big = int((v > VOL_MID).sum())

    print("=" * 60)
    print("PER-LESION ET VOLUME DISTRIBUTION")
    print("=" * 60)
    print(f"Total lesions across dataset: {n}")
    print(f"  Detection only   (<27 mm3)   : {small:>5}  ({100*small/n:.1f}%)")
    print(f"  Middle zone   (27-275 mm3)   : {mid:>5}  ({100*mid/n:.1f}%)")
    print(f"  Segmentation     (>275 mm3)  : {big:>5}  ({100*big/n:.1f}%)")
    print(f"  Median : {np.median(v):.1f} mm3")
    print(f"  Mean   : {v.mean():.1f} mm3")
    print(f"  p5     : {np.percentile(v, 5):.1f} mm3")
    print(f"  p95    : {np.percentile(v, 95):.1f} mm3")

    # ---- histogram (log-x) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    vpos = v[v > 0]
    bins = np.logspace(np.log10(vpos.min()), np.log10(vpos.max()), 60)
    ax.hist(vpos, bins=bins, color="#4477aa", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.axvline(VOL_SMALL, color="darkorange", linestyle="--", linewidth=2,
               label=f"27 mm³ (detection floor)")
    ax.axvline(VOL_MID, color="crimson", linestyle="--", linewidth=2,
               label=f"275 mm³ (segmentation floor)")
    ax.set_xlabel("Lesion volume (mm³, log scale)")
    ax.set_ylabel("Number of lesions")
    ax.set_title(f"ET lesion volume distribution (n={n} lesions)\n"
                 f"detection<27 ({100*small/n:.0f}%)  |  "
                 f"middle 27-275 ({100*mid/n:.0f}%)  |  "
                 f"seg>275 ({100*big/n:.0f}%)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    out_png = os.path.join(OUT, "lesion_volume_distribution.png")
    plt.savefig(out_png, dpi=120)
    print(f"\nSaved {out_png}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
