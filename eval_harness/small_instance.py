"""Small-instance detection metrics (TP / FN / FP / F1) for one region.

A pure DETECTION track: did we find each small lesion, regardless of boundary
quality. METRIC_DECISIONS.md sec 5.

  - "small" = GT connected components with volume < SMALL_VOLUME_THRESH mm^3.
  - matching = ANY overlap (a predicted component touching a small GT lesion
    counts as detected). overlap threshold -> 0, detection-first.
  - TP = small GT lesion detected; FN = small GT lesion missed;
    FP = predicted small lesion overlapping no GT lesion.
  - F1 = TP / (TP + 0.5*(FP + FN))  -- panoptica Recognition Quality (RQ),
    which IS the F1 score (verified, panoptica metrics.md).

SMALL_VOLUME_THRESH and the any-overlap rule are ANCHOR-TODO knobs (sec 5).
"""

import numpy as np
from scipy import ndimage

# ANCHOR TODO (sec 5): small = < 27 mm^3 (this project's sub-detection band).
SMALL_VOLUME_THRESH = 27.0

_CONN26 = np.ones((3, 3, 3), dtype=bool)


def small_instance_region(gt_mask: np.ndarray, pred_mask: np.ndarray, spacing):
    """Return {tp, fn, fp, f1} for one binary region's small lesions."""
    voxel_vol = float(np.prod(spacing))

    gt_cc, n_gt = ndimage.label(gt_mask, structure=_CONN26)
    pred_cc, n_pred = ndimage.label(pred_mask, structure=_CONN26)

    # Identify the SMALL populations (GT and pred) by volume.
    small_gt = [
        gl for gl in range(1, n_gt + 1)
        if (gt_cc == gl).sum() * voxel_vol < SMALL_VOLUME_THRESH
    ]
    small_pred = {
        pl for pl in range(1, n_pred + 1)
        if (pred_cc == pl).sum() * voxel_vol < SMALL_VOLUME_THRESH
    }

    tp = 0
    fn = 0
    matched_pred = set()
    for gl in small_gt:
        gt_lesion = gt_cc == gl
        # any-overlap: which predicted components does this lesion touch?
        hits = np.unique(pred_cc[gt_lesion & (pred_cc > 0)])
        hits = [int(h) for h in hits if h != 0]
        if hits:
            tp += 1
            matched_pred.update(hits)
        else:
            fn += 1

    # FP = small predicted lesions matched to no small GT lesion.
    fp = len(small_pred - matched_pred)

    denom = tp + 0.5 * (fp + fn)
    f1 = float(tp / denom) if denom > 0 else float("nan")

    return {"tp": tp, "fn": fn, "fp": fp, "f1": f1}
