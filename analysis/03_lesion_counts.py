#!/usr/bin/env python3
"""
Script 3 — Lesion count per case; single vs multi-lesion breakdown.

A lesion = one connected component of the ET mask (label 3, 26-connectivity).
Counts were computed in dataset_stats.py (scipy.ndimage.label) and stored as
"n_lesions" per case in outputs/per_case_records.json; this script consumes them.

  single-lesion : exactly 1 ET connected component
  multi-lesion  : 2+ ET connected components
(Cases with 0 ET components are reported separately — they are neither single
 nor multi by the 1-vs-2+ definition.)

Outputs:
  outputs/lesion_count_distribution.png   (x=#lesions, y=#cases)
  outputs/lesion_counts.csv               (case_id, num_lesions, single_or_multi)
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


def classify(n):
    if n == 0:
        return "no_lesion"
    return "single" if n == 1 else "multi"


def main():
    with open(RECORDS) as f:
        records = json.load(f)

    rows = []
    counts = []
    for r in records:
        n = int(r["n_lesions"])
        rows.append((r["case_id"], n, classify(n)))
        counts.append(n)

    # ---- CSV ----
    out_csv = os.path.join(OUT, "lesion_counts.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "num_lesions", "single_or_multi"])
        w.writerows(rows)

    arr = np.array(counts)
    total = len(arr)
    n_none = int((arr == 0).sum())
    n_single = int((arr == 1).sum())
    n_multi = int((arr >= 2).sum())

    # case with most lesions
    imax = int(arr.argmax())
    max_case, max_count = rows[imax][0], rows[imax][1]

    print("=" * 60)
    print("LESION COUNT PER CASE (ET connected components)")
    print("=" * 60)
    print(f"Total cases:                 {total}")
    print(f"  No ET lesion (0):          {n_none}  ({100*n_none/total:.1f}%)")
    print(f"  Single-lesion (1):         {n_single}  ({100*n_single/total:.1f}%)")
    print(f"  Multi-lesion (2+):         {n_multi}  ({100*n_multi/total:.1f}%)")
    # single vs multi among cases that HAVE lesions
    have = n_single + n_multi
    if have:
        print(f"  Among cases with >=1 lesion: "
              f"single {100*n_single/have:.1f}% / multi {100*n_multi/have:.1f}%")
    print(f"\nMost lesions in one case: {max_case} with {max_count} lesions")

    # ---- histogram: x = number of lesions, y = number of cases ----
    fig, ax = plt.subplots(figsize=(11, 6))
    maxc = arr.max()
    bins = np.arange(0, maxc + 2) - 0.5
    ax.hist(arr, bins=bins, color="#228833", edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")  # long tail (1 case @ 393) vs 274 single -> log y
    ax.set_xlabel("Number of ET lesions in case")
    ax.set_ylabel("Number of cases (log scale)")
    ax.set_title(f"Lesion-count distribution per case (n={total} cases)\n"
                 f"single {100*n_single/total:.0f}%  |  multi {100*n_multi/total:.0f}%  "
                 f"|  max {max_count} ({max_case})")
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    out_png = os.path.join(OUT, "lesion_count_distribution.png")
    plt.savefig(out_png, dpi=120)
    print(f"\nSaved {out_png}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
