#!/usr/bin/env python3
"""
Script 1 — Pre- vs post-treatment classification via resection cavity (label 4).

Labels: 1=NCR(NETC), 2=SNFH, 3=ET, 4=RC.
A case is "post-treatment" iff label 4 (RC) is present, else "pre-treatment".

This is a VIEW builder: instead of re-streaming ~1300 .nii.gz files off Google
Drive (~30 min), it reuses analysis/outputs/per_case_records.json, which was
produced by dataset_stats.py with correct per-case header voxel spacing and a
full np.unique(seg) label list per case (`labels_present`). The numbers are
therefore identical to a fresh nibabel pass.

Outputs:
  outputs/treatment_status.csv  (case_id, has_RC, has_ET, has_SNFH, has_NCR, treatment_status)
"""
import os
import csv
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "outputs")
RECORDS = os.path.join(OUT, "per_case_records.json")

NCR, SNFH, ET, RC = 1, 2, 3, 4


def main():
    with open(RECORDS) as f:
        records = json.load(f)

    rows = []
    post_ids = []
    n_pre = n_post = 0
    for r in records:
        labels = set(r["labels_present"])
        has_rc = RC in labels
        status = "post-treatment" if has_rc else "pre-treatment"
        if has_rc:
            n_post += 1
            post_ids.append(r["case_id"])
        else:
            n_pre += 1
        rows.append({
            "case_id": r["case_id"],
            "has_RC": has_rc,
            "has_ET": ET in labels,
            "has_SNFH": SNFH in labels,
            "has_NCR": NCR in labels,
            "treatment_status": status,
        })

    out_csv = os.path.join(OUT, "treatment_status.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "has_RC", "has_ET",
                                          "has_SNFH", "has_NCR", "treatment_status"])
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    print("=" * 60)
    print("TREATMENT STATUS (resection cavity = label 4)")
    print("=" * 60)
    print(f"Total cases:     {total}")
    print(f"Pre-treatment:   {n_pre}  ({100*n_pre/total:.1f}%)")
    print(f"Post-treatment:  {n_post}  ({100*n_post/total:.1f}%)")
    print()
    print(f"Post-treatment case IDs ({len(post_ids)}):")
    for cid in post_ids:
        print(f"  {cid}")
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
