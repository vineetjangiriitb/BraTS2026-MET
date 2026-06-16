"""Region (ROI) construction for BraTS-METS 2025 scoring.

Maps the raw integer label map (our dataset convention) to the four scored
binary regions. Verified against the official scorer's get_TissueWiseSeg unions
(rachitsaluja/BraTS-2023-Metrics). See eval_harness/METRIC_DECISIONS.md sec 1.

Dataset labels (this project):
    0 = background, 1 = NETC, 2 = SNFH, 3 = ET, 4 = RC
"""

import numpy as np

# region name -> set of integer labels unioned to form the binary mask.
# ET={3}, TC={1,3} (NETC+ET), WT={1,2,3}, RC={4}. METRIC_DECISIONS.md sec 1-2.
REGION_LABELS = {
    "et": (3,),
    "tc": (1, 3),
    "wt": (1, 2, 3),
    "rc": (4,),
}

# Order used everywhere downstream (column order in outputs).
REGIONS = ("et", "tc", "wt", "rc")


def region_mask(seg: np.ndarray, region: str) -> np.ndarray:
    """Binary mask for one region from an integer label volume."""
    labels = REGION_LABELS[region]
    out = np.zeros(seg.shape, dtype=bool)
    for lab in labels:
        out |= seg == lab
    return out


def region_present(seg: np.ndarray, region: str) -> bool:
    """True if the region has any voxel in this volume.

    Used for aggregation: a region absent from the ground truth is skipped for
    that case (METRIC_DECISIONS.md sec 6), the same way RC is handled when a
    case has no resection cavity (sec 2).
    """
    labels = REGION_LABELS[region]
    return bool(np.isin(seg, labels).any())
