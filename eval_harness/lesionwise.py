"""Lesion-wise DSC and NSD for one region.

Mirrors the official BraTS-METS lesion-wise algorithm (METRIC_DECISIONS.md sec 3-4):
  - 26-connectivity connected components on the GT region mask.
  - each GT lesion below LESION_VOLUME_THRESH mm^3 is dropped (noise floor).
  - each remaining GT lesion is dilated by DILATION_FACTOR and intersected with
    the prediction; the overlapping predicted voxels are that lesion's match.
  - per-lesion DSC and NSD are computed on (gt_lesion, matched_pred).
  - a GT lesion with no predicted overlap is a false negative -> score 0.
  - predicted components touching no GT lesion are false positives -> score 0.
  - the region score is the mean over (all GT lesions + all FP lesions).

VERIFIED constants are MET-specific. NSD tolerances and the FP/FN NSD penalty
are the open ANCHOR-TODO knobs; see METRIC_DECISIONS.md sec 4.
"""

import numpy as np
from scipy import ndimage
from surface_distance import (
    compute_surface_distances,
    compute_surface_dice_at_tolerance,
)

# --- VERIFIED constants (MET variant), METRIC_DECISIONS.md sec 3 ---
DILATION_FACTOR = 1          # MET uses 1 (GLI/PED/SSA use 3)
CONNECTIVITY = 26            # 3D 26-connectivity
LESION_VOLUME_THRESH = 2.0   # mm^3; GT lesions below this are excluded

# --- NSD tolerances: compute BOTH, prune after anchor (sec 4) ---
NSD_TOLERANCES_MM = (0.5, 1.0)

# FN/FP lesion penalties.
DSC_MISS_PENALTY = 0.0       # VERIFIED
NSD_MISS_PENALTY = 0.0       # ANCHOR TODO (sec 4): ratio metric -> 0 is sensible

# 26-connectivity structuring element (3x3x3 all-ones).
_CONN26 = np.ones((3, 3, 3), dtype=bool)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    asum, bsum = a.sum(), b.sum()
    if asum + bsum == 0:
        return 1.0
    return 2.0 * np.logical_and(a, b).sum() / (asum + bsum)


def _nsd(gt: np.ndarray, pred: np.ndarray, spacing, tol: float) -> float:
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    if gt.sum() == 0 or pred.sum() == 0:
        return 0.0
    sd = compute_surface_distances(gt, pred, spacing_mm=spacing)
    return float(compute_surface_dice_at_tolerance(sd, tol))


def lesionwise_region(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    spacing,
):
    """Return per-region lesion-wise scores for one binary region.

    Args:
        gt_mask, pred_mask: boolean volumes for the SAME region.
        spacing: (sz, sy, sx) voxel size in mm (from the NIfTI header).

    Returns dict with keys:
        dsc                     mean lesion-wise Dice
        nsd_0.5, nsd_1.0        mean lesion-wise NSD at each tolerance
        n_gt_lesions, n_fp_lesions   bookkeeping
    """
    voxel_vol = float(np.prod(spacing))

    gt_cc, n_gt = ndimage.label(gt_mask, structure=_CONN26)
    pred_cc, n_pred = ndimage.label(pred_mask, structure=_CONN26)

    # Keep only GT lesions at/above the volume threshold.
    gt_labels = []
    for gl in range(1, n_gt + 1):
        if (gt_cc == gl).sum() * voxel_vol >= LESION_VOLUME_THRESH:
            gt_labels.append(gl)

    matched_pred_labels = set()
    dsc_scores, nsd_scores = [], {t: [] for t in NSD_TOLERANCES_MM}

    for gl in gt_labels:
        gt_lesion = gt_cc == gl
        # dilate this GT lesion, see which predicted components it touches.
        dil = ndimage.binary_dilation(
            gt_lesion, structure=_CONN26, iterations=DILATION_FACTOR
        )
        hit_labels = np.unique(pred_cc[dil & (pred_cc > 0)])
        if hit_labels.size == 0:
            # false negative lesion
            dsc_scores.append(DSC_MISS_PENALTY)
            for t in NSD_TOLERANCES_MM:
                nsd_scores[t].append(NSD_MISS_PENALTY)
            continue
        matched_pred = np.isin(pred_cc, hit_labels)
        matched_pred_labels.update(int(x) for x in hit_labels)
        dsc_scores.append(_dice(gt_lesion, matched_pred))
        for t in NSD_TOLERANCES_MM:
            nsd_scores[t].append(_nsd(gt_lesion, matched_pred, spacing, t))

    # Predicted components matched to no GT lesion are false positives.
    n_fp = 0
    for pl in range(1, n_pred + 1):
        if pl in matched_pred_labels:
            continue
        if (pred_cc == pl).sum() * voxel_vol < LESION_VOLUME_THRESH:
            continue  # sub-threshold predicted speckle is ignored, not an FP
        n_fp += 1
        dsc_scores.append(DSC_MISS_PENALTY)
        for t in NSD_TOLERANCES_MM:
            nsd_scores[t].append(NSD_MISS_PENALTY)

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    out = {
        "dsc": _mean(dsc_scores),
        "n_gt_lesions": len(gt_labels),
        "n_fp_lesions": n_fp,
    }
    for t in NSD_TOLERANCES_MM:
        out[f"nsd_{t}"] = _mean(nsd_scores[t])
    return out
