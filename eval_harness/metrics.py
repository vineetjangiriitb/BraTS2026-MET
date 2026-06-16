"""Score one (prediction, ground-truth) pair into the leaderboard columns.

Composes regions + lesionwise + small_instance into a flat dict whose keys
mirror the BraTS-METS Synapse leaderboard, e.g.:
    Lesionwise_dsc_mean_et, Lesionwise_nsd_mean_et   (+ _nsd0.5 / _nsd1.0 variants)
    Small_instance_tp_et, Small_instance_fn_et, Small_instance_fp_et, Small_instance_f1_et
... for each region in {et, tc, wt, rc}.

A region absent from the GROUND TRUTH is scored NaN for that case and skipped in
aggregation (METRIC_DECISIONS.md sec 2, 6).
"""

import numpy as np
import nibabel as nib

from regions import REGIONS, region_mask, region_present
from lesionwise import lesionwise_region, NSD_TOLERANCES_MM
from small_instance import small_instance_region


def _load(path):
    img = nib.load(str(path))
    seg = np.asarray(img.dataobj).astype(np.int16)
    # spacing in mm from the header (sz, sy, sx order matches array axes).
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return seg, spacing


def score_case(pred_path, gt_path) -> dict:
    """Score a single case. Keys mirror the leaderboard columns."""
    gt, gt_spacing = _load(gt_path)
    pred, _ = _load(pred_path)

    if pred.shape != gt.shape:
        raise ValueError(
            f"shape mismatch: pred {pred.shape} vs gt {gt.shape} "
            f"({pred_path} / {gt_path})"
        )

    row = {}
    for region in REGIONS:
        present = region_present(gt, region)
        gt_r = region_mask(gt, region)
        pred_r = region_mask(pred, region)

        if not present:
            # absent region -> NaN, skipped downstream.
            row[f"Lesionwise_dsc_mean_{region}"] = float("nan")
            for t in NSD_TOLERANCES_MM:
                row[f"Lesionwise_nsd{t}_mean_{region}"] = float("nan")
            for k in ("tp", "fn", "fp", "f1"):
                row[f"Small_instance_{k}_{region}"] = float("nan")
            continue

        lw = lesionwise_region(gt_r, pred_r, gt_spacing)
        row[f"Lesionwise_dsc_mean_{region}"] = lw["dsc"]
        for t in NSD_TOLERANCES_MM:
            row[f"Lesionwise_nsd{t}_mean_{region}"] = lw[f"nsd_{t}"]

        si = small_instance_region(gt_r, pred_r, gt_spacing)
        for k in ("tp", "fn", "fp", "f1"):
            row[f"Small_instance_{k}_{region}"] = si[k]

    return row


def column_order():
    """Canonical column order for outputs (matches leaderboard grouping)."""
    cols = []
    for region in REGIONS:
        cols.append(f"Lesionwise_dsc_mean_{region}")
    for region in REGIONS:
        for t in NSD_TOLERANCES_MM:
            cols.append(f"Lesionwise_nsd{t}_mean_{region}")
    for region in REGIONS:
        for k in ("tp", "fn", "fp", "f1"):
            cols.append(f"Small_instance_{k}_{region}")
    return cols
