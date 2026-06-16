#!/usr/bin/env python3
"""
BraTS2026-MET — Dataset Statistics & Analysis (Section A: A1–A6)

Produces the statistics needed for the initial competition presentation and for
model-design decisions. Handles the two cohorts correctly:

  - non-UCSD : SRI24 atlas space, 240x240x155 @ 1mm isotropic
  - UCSD     : native clinical space, variable shape & anisotropic voxels

Label scheme (BraTS-MET 2025):
  0 = background, 1 = NETC, 2 = SNFH/edema, 3 = ET (enhancing tumor), 4 = RC (resection cavity)

Lesion = connected component of the ET mask (label 3), 26-connectivity.
Volumes are computed with each case's OWN voxel spacing (mm^3) — essential
because UCSD voxels are ~0.49x0.49x1.0 mm, not 1mm^3.

Competition volume thresholds (per-lesion):
  < 27   mm^3 : below detection floor / detection-only regime
  27-275 mm^3 : detection regime
  > 275  mm^3 : segmentation evaluated (Dice counts)
"""
import os
import sys
import json
import csv
from collections import defaultdict

import numpy as np
import nibabel as nib
from scipy import ndimage

# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "training", "MICCAI-LH-BraTS2025-MET-Challenge-Training")
UCSD = os.path.join(TRAIN, "UCSD-Training")
OUT = os.path.join(ROOT, "analysis", "outputs")
os.makedirs(OUT, exist_ok=True)

ET_LABEL = 3
RC_LABEL = 4
MODALITIES = ["t1c", "t1n", "t2f", "t2w"]

VOL_SMALL = 27.0    # mm^3
VOL_MID = 275.0     # mm^3

CONNECTIVITY = ndimage.generate_binary_structure(3, 3)  # 26-connectivity


def list_cases(folder, exclude_subdirs=()):
    cases = []
    for name in sorted(os.listdir(folder)):
        if name.startswith("."):
            continue
        path = os.path.join(folder, name)
        if not os.path.isdir(path):
            continue
        if name in exclude_subdirs:
            continue
        cases.append((name, path))
    return cases


def timepoint(case_id):
    # BraTS-MET-XXXXX-TTT  -> TTT
    return case_id.rsplit("-", 1)[-1]


def analyze_case(case_id, path, cohort):
    seg_file = os.path.join(path, f"{case_id}-seg.nii.gz")
    if not os.path.exists(seg_file):
        return None

    img = nib.load(seg_file)
    seg = np.asarray(img.dataobj).astype(np.int16)
    zooms = img.header.get_zooms()[:3]
    voxel_mm3 = float(zooms[0] * zooms[1] * zooms[2])

    # --- A1/A6: lesion count via connected components on ET ---
    et_mask = seg == ET_LABEL
    labeled, n_lesions = ndimage.label(et_mask, structure=CONNECTIVITY)

    # --- A2: per-lesion volumes (mm^3) ---
    lesion_volumes = []
    if n_lesions > 0:
        counts = np.bincount(labeled.ravel())[1:]  # drop background bin
        lesion_volumes = (counts * voxel_mm3).tolist()

    # --- A4: resection cavity presence (post-treatment marker) ---
    rc_voxels = int(np.count_nonzero(seg == RC_LABEL))
    has_rc = rc_voxels > 0

    # --- A5: modality availability ---
    modality_present = {
        m: os.path.exists(os.path.join(path, f"{case_id}-{m}.nii.gz"))
        for m in MODALITIES
    }

    return {
        "case_id": case_id,
        "cohort": cohort,
        "timepoint": timepoint(case_id),
        "shape": list(seg.shape),
        "voxel_mm3": voxel_mm3,
        "zooms": [float(z) for z in zooms],
        "n_lesions": int(n_lesions),
        "lesion_volumes": lesion_volumes,
        "total_et_mm3": float(sum(lesion_volumes)),
        "has_rc": has_rc,
        "rc_voxels": rc_voxels,
        "labels_present": np.unique(seg).tolist(),
        **{f"has_{m}": modality_present[m] for m in MODALITIES},
    }


def main():
    non_ucsd = [(c, p, "non-UCSD") for c, p in list_cases(TRAIN, exclude_subdirs={"UCSD-Training"})]
    ucsd = [(c, p, "UCSD") for c, p in list_cases(UCSD)]
    all_cases = non_ucsd + ucsd
    print(f"Found {len(non_ucsd)} non-UCSD + {len(ucsd)} UCSD = {len(all_cases)} timepoints")

    records = []
    for i, (cid, path, cohort) in enumerate(all_cases, 1):
        rec = analyze_case(cid, path, cohort)
        if rec is not None:
            records.append(rec)
        if i % 100 == 0:
            print(f"  ...{i}/{len(all_cases)}")
    print(f"Analyzed {len(records)} cases with segmentations\n")

    # Persist raw per-case records
    with open(os.path.join(OUT, "per_case_records.json"), "w") as f:
        json.dump(records, f, indent=2)

    # Flat CSV (one row per case)
    with open(os.path.join(OUT, "per_case.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "cohort", "timepoint", "n_lesions", "total_et_mm3",
                    "has_rc", "voxel_mm3", *[f"has_{m}" for m in MODALITIES]])
        for r in records:
            w.writerow([r["case_id"], r["cohort"], r["timepoint"], r["n_lesions"],
                        round(r["total_et_mm3"], 1), r["has_rc"], round(r["voxel_mm3"], 4),
                        *[r[f"has_{m}"] for m in MODALITIES]])

    summarize(records)


def pct(n, d):
    return f"{100.0*n/d:.1f}%" if d else "n/a"


def summarize(records):
    report = []
    def line(s=""):
        report.append(s)
        print(s)

    by_cohort = defaultdict(list)
    for r in records:
        by_cohort[r["cohort"]].append(r)

    line("=" * 70)
    line("BraTS2026-MET DATASET STATISTICS  (Section A)")
    line("=" * 70)
    line(f"Total timepoints analyzed: {len(records)}")
    for ch, recs in by_cohort.items():
        line(f"  {ch}: {len(recs)}")
    line()

    # ---------------- A1: per-lesion count distribution ----------------
    line("-" * 70)
    line("A1 — PER-LESION COUNT DISTRIBUTION (ET connected components)")
    line("-" * 70)
    counts = [r["n_lesions"] for r in records]
    arr = np.array(counts)
    line(f"  Cases with >=1 ET lesion : {int((arr>0).sum())} / {len(arr)} ({pct((arr>0).sum(),len(arr))})")
    line(f"  Cases with ZERO ET       : {int((arr==0).sum())} (pre-tx / fully-resected / edema-only)")
    nz = arr[arr > 0]
    if len(nz):
        line(f"  Mean lesions/case (incl 0): {arr.mean():.2f}")
        line(f"  Mean lesions/case (>0 only): {nz.mean():.2f}")
        line(f"  Median (>0 only)           : {np.median(nz):.0f}")
        line(f"  Max lesions in one case    : {arr.max()}")
    # histogram
    line("  Histogram (lesion count -> #cases):")
    hist = defaultdict(int)
    for c in counts:
        hist[c] += 1
    for k in sorted(hist):
        bar = "#" * min(hist[k], 60)
        line(f"    {k:>3} : {hist[k]:>4}  {bar}")
    line()

    # ---------------- A6: single vs multi-lesion breakdown ----------------
    line("-" * 70)
    line("A6 — SINGLE vs MULTI-LESION BREAKDOWN")
    line("-" * 70)
    buckets = {"0 (no ET)": 0, "1 (single)": 0, "2-5 (multi)": 0, "6+ (highly multi-focal)": 0}
    for c in counts:
        if c == 0:
            buckets["0 (no ET)"] += 1
        elif c == 1:
            buckets["1 (single)"] += 1
        elif c <= 5:
            buckets["2-5 (multi)"] += 1
        else:
            buckets["6+ (highly multi-focal)"] += 1
    for k, v in buckets.items():
        line(f"  {k:<26}: {v:>4}  ({pct(v,len(counts))})")
    multi = buckets["2-5 (multi)"] + buckets["6+ (highly multi-focal)"]
    line(f"  => Multi-focal (2+ lesions): {multi} ({pct(multi,len(counts))}) — the hard cases")
    line()

    # ---------------- A2: lesion-wise volume distribution ----------------
    line("-" * 70)
    line("A2 — LESION-WISE VOLUME DISTRIBUTION (per individual lesion, mm^3)")
    line("-" * 70)
    all_vols = []
    for r in records:
        all_vols.extend(r["lesion_volumes"])
    v = np.array(all_vols)
    line(f"  Total individual ET lesions across dataset: {len(v)}")
    if len(v):
        small = int((v < VOL_SMALL).sum())
        mid = int(((v >= VOL_SMALL) & (v <= VOL_MID)).sum())
        big = int((v > VOL_MID).sum())
        line(f"  < {VOL_SMALL:g} mm^3 (sub-detection)      : {small:>5}  ({pct(small,len(v))})")
        line(f"  {VOL_SMALL:g}-{VOL_MID:g} mm^3 (detection)        : {mid:>5}  ({pct(mid,len(v))})")
        line(f"  > {VOL_MID:g} mm^3 (Dice-evaluated)     : {big:>5}  ({pct(big,len(v))})")
        line(f"  Median lesion volume : {np.median(v):.1f} mm^3")
        line(f"  Mean   lesion volume : {v.mean():.1f} mm^3")
        for p in (10, 25, 50, 75, 90, 95, 99):
            line(f"    p{p:<2} : {np.percentile(v,p):.1f} mm^3")
        line(f"  Largest single lesion: {v.max():.0f} mm^3")
        # log-scale histogram
        line("  Log-binned histogram:")
        edges = [0, 27, 100, 275, 1000, 5000, 20000, 1e9]
        labels = ["<27", "27-100", "100-275", "275-1k", "1k-5k", "5k-20k", ">20k"]
        idx = np.digitize(v, edges) - 1
        for i, lab in enumerate(labels):
            n = int((idx == i).sum())
            bar = "#" * min(n // 5, 60)
            line(f"    {lab:>8} mm^3 : {n:>5}  {bar}")
    line()

    # ---------------- A4: pre vs post-treatment (RC presence) ----------------
    line("-" * 70)
    line("A4 — PRE vs POST-TREATMENT (resection cavity, label 4)")
    line("-" * 70)
    for ch in ("non-UCSD", "UCSD"):
        recs = by_cohort.get(ch, [])
        if not recs:
            continue
        with_rc = sum(r["has_rc"] for r in recs)
        line(f"  {ch}: {with_rc}/{len(recs)} have RC ({pct(with_rc,len(recs))}) "
             f"=> post-treatment; {len(recs)-with_rc} pre-treatment/no-cavity")
    # By timepoint within UCSD
    ucsd = by_cohort.get("UCSD", [])
    if ucsd:
        line("  UCSD RC by timepoint:")
        tp = defaultdict(lambda: [0, 0])
        for r in ucsd:
            tp[r["timepoint"]][0] += 1
            tp[r["timepoint"]][1] += int(r["has_rc"])
        for k in sorted(tp):
            tot, rc = tp[k]
            line(f"    tp {k}: {rc}/{tot} have RC ({pct(rc,tot)})")
    line()

    # ---------------- A5: modality availability ----------------
    line("-" * 70)
    line("A5 — MODALITY AVAILABILITY")
    line("-" * 70)
    for ch in ("non-UCSD", "UCSD"):
        recs = by_cohort.get(ch, [])
        if not recs:
            continue
        line(f"  {ch} ({len(recs)} cases):")
        for m in MODALITIES:
            present = sum(r[f"has_{m}"] for r in recs)
            flag = "" if present == len(recs) else "  <-- MISSING IN SOME"
            line(f"    {m}: {present}/{len(recs)} ({pct(present,len(recs))}){flag}")
        # cases missing any modality
        incomplete = [r["case_id"] for r in recs
                      if not all(r[f"has_{m}"] for m in MODALITIES)]
        if incomplete:
            line(f"    Incomplete cases ({len(incomplete)}): {incomplete[:10]}{' ...' if len(incomplete)>10 else ''}")
    line()

    line("=" * 70)
    line("NOTE on A3 (spatial heatmap): computed separately for SRI24 cases only")
    line("(common atlas space). See spatial_heatmap.py.")
    line("=" * 70)

    with open(os.path.join(OUT, "SECTION_A_REPORT.txt"), "w") as f:
        f.write("\n".join(report))
    print(f"\nReport written to {os.path.join(OUT, 'SECTION_A_REPORT.txt')}")
    print(f"Per-case CSV/JSON written to {OUT}/")


if __name__ == "__main__":
    main()
