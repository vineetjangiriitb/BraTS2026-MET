#!/usr/bin/env python3
"""
Select 100 balanced non-UCSD SRI24-canonical cases for the nnU-Net educational run.

WHY THIS FILTERING (first principles):
  nnU-Net "fingerprints" the dataset before planning the network — it computes the
  MEDIAN voxel spacing and image shape across all cases, then designs the patch size
  and architecture from that. If we mixed the UCSD cohort (native clinical space,
  0.12-0.6 mm voxels, 512x512x* shapes) with the non-UCSD cohort (SRI24 atlas,
  1.0 mm isotropic, 240x240x155), the median fingerprint would be a meaningless
  average of two very different distributions and the planned patch could blow up
  past any GPU's memory.

  So we restrict to the 328 non-UCSD cases that sit on the canonical
  240x240x155 @ 1.0 mm^3 grid. Same space, same resolution -> clean fingerprint.

BALANCING:
  75% of BraTS-MET cases are multi-focal. We stratify the 100-case sample across
  lesion-count buckets so the model sees single, few, and many-lesion cases in
  representative proportion, rather than an accidental skew from naive sampling.
"""
import json
import random
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RECORDS = PROJECT / "analysis" / "outputs" / "per_case_records.json"
OUT = Path(__file__).resolve().parent / "case_list_100.txt"

# Fixed seed -> reproducible selection (important: same 100 cases every run, so
# the train/val split and reported metrics are reproducible for the presentation).
SEED = 42
N_TARGET = 100

# Stratification buckets by lesion count, with target counts that roughly mirror
# the dataset's natural distribution (single ~21%, multi ~75%, highly-multifocal tail).
BUCKETS = {
    "single (1 lesion)":      (lambda n: n == 1,        25),
    "few (2-5 lesions)":      (lambda n: 2 <= n <= 5,   50),
    "many (6+ lesions)":      (lambda n: n >= 6,        25),
}


def main():
    recs = json.loads(RECORDS.read_text())

    # Filter: non-UCSD, canonical SRI24 grid, has enhancing tumor (label 3).
    canonical = [
        r for r in recs
        if r["cohort"] == "non-UCSD"
        and r["shape"] == [240, 240, 155]
        and r["voxel_mm3"] == 1.0
        and 3 in r["labels_present"]
    ]
    print(f"Canonical non-UCSD cases available: {len(canonical)}")

    rng = random.Random(SEED)
    selected = []

    for name, (predicate, target) in BUCKETS.items():
        pool = [r["case_id"] for r in canonical if predicate(r["n_lesions"])]
        rng.shuffle(pool)
        take = pool[:target]
        selected.extend(take)
        print(f"  {name:24s} pool={len(pool):3d}  selected={len(take)}")

    # If a bucket ran short, top up from remaining canonical cases to hit N_TARGET.
    if len(selected) < N_TARGET:
        remaining = [r["case_id"] for r in canonical if r["case_id"] not in set(selected)]
        rng.shuffle(remaining)
        topup = remaining[: N_TARGET - len(selected)]
        selected.extend(topup)
        print(f"  topped up with {len(topup)} extra cases to reach {N_TARGET}")

    selected = selected[:N_TARGET]
    OUT.write_text("\n".join(selected) + "\n")
    print(f"\nWrote {len(selected)} case IDs -> {OUT}")


if __name__ == "__main__":
    main()
