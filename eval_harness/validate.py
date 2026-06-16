"""Invariant tests for the harness (no leaderboard anchor needed yet).

We validate by CONSTRUCTION: synthetic label volumes with known answers exercise
each rule in METRIC_DECISIONS.md. Run: python eval_harness/validate.py
"""

import numpy as np

from regions import region_mask, region_present
from lesionwise import lesionwise_region, LESION_VOLUME_THRESH
from small_instance import small_instance_region, SMALL_VOLUME_THRESH

SPACING = (1.0, 1.0, 1.0)  # 1mm iso (data is resampled to 1mm; voxel=1mm^3)
_fails = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


def cube(vol, center, half, label):
    z, y, x = center
    vol[z - half:z + half + 1, y - half:y + half + 1, x - half:x + half + 1] = label


print("== regions ==")
seg = np.zeros((20, 20, 20), np.int16)
seg[5, 5, 5] = 3   # ET
seg[5, 5, 6] = 1   # NETC
seg[5, 5, 7] = 2   # SNFH
check("ET mask = label 3 only", region_mask(seg, "et").sum() == 1)
check("TC mask = NETC+ET (2 vox)", region_mask(seg, "tc").sum() == 2)
check("WT mask = NETC+SNFH+ET (3 vox)", region_mask(seg, "wt").sum() == 3)
check("RC absent -> not present", not region_present(seg, "rc"))
check("ET present", region_present(seg, "et"))

print("== lesionwise: perfect prediction ==")
gt = np.zeros((30, 30, 30), bool)
cube(gt, (15, 15, 15), 3, True)        # 7^3 = 343 mm^3 lesion
lw = lesionwise_region(gt, gt.copy(), SPACING)
check("perfect DSC == 1.0", abs(lw["dsc"] - 1.0) < 1e-9)
check("perfect NSD@0.5 == 1.0", abs(lw["nsd_0.5"] - 1.0) < 1e-9)
check("perfect NSD@1.0 == 1.0", abs(lw["nsd_1.0"] - 1.0) < 1e-9)
check("1 GT lesion counted", lw["n_gt_lesions"] == 1)
check("0 FP lesions", lw["n_fp_lesions"] == 0)

print("== lesionwise: missed lesion (false negative) ==")
empty = np.zeros((30, 30, 30), bool)
lw = lesionwise_region(gt, empty, SPACING)
check("missed lesion DSC == 0", lw["dsc"] == 0.0)
check("missed lesion NSD@1.0 == 0", lw["nsd_1.0"] == 0.0)

print("== lesionwise: phantom prediction (false positive) ==")
pred = np.zeros((30, 30, 30), bool)
cube(pred, (5, 5, 5), 3, True)         # pred lesion far from (empty) GT
lw = lesionwise_region(empty, pred, SPACING)
check("FP lesion counted", lw["n_fp_lesions"] == 1)
check("FP-only mean DSC == 0", lw["dsc"] == 0.0)

print("== lesionwise: sub-threshold GT lesion excluded ==")
tiny = np.zeros((30, 30, 30), bool)
tiny[15, 15, 15] = True                 # 1 mm^3 < 2 mm^3 threshold
lw = lesionwise_region(tiny, tiny.copy(), SPACING)
check("sub-2mm3 GT lesion dropped (0 lesions)", lw["n_gt_lesions"] == 0)
check("no lesions -> NaN DSC", np.isnan(lw["dsc"]))

print("== lesionwise: two lesions, one hit one missed -> 0.5 ==")
gt2 = np.zeros((40, 40, 40), bool)
cube(gt2, (10, 10, 10), 3, True)
cube(gt2, (30, 30, 30), 3, True)
pred2 = np.zeros((40, 40, 40), bool)
cube(pred2, (10, 10, 10), 3, True)      # hit lesion A, miss lesion B
lw = lesionwise_region(gt2, pred2, SPACING)
check("DSC mean == 0.5 (1.0 + 0.0)/2", abs(lw["dsc"] - 0.5) < 1e-9)

print("== small-instance: detect / miss / phantom ==")
# one small GT lesion (3^3=27? -> use 2^3=8 mm^3 < 27), detected by overlap.
sgt = np.zeros((30, 30, 30), bool)
cube(sgt, (15, 15, 15), 1, True)        # 3^3 = 27 -> use half=1 => 3^3? careful
# half=1 -> side 3 -> 27 mm^3 which is NOT < 27. shrink to single-plane 2x2x2.
sgt[:] = False
sgt[15:17, 15:17, 15:17] = True          # 8 mm^3 < 27
si = small_instance_region(sgt, sgt.copy(), SPACING)
check("small lesion detected -> tp=1", si["tp"] == 1)
check("detected -> f1 == 1.0", abs(si["f1"] - 1.0) < 1e-9)

si = small_instance_region(sgt, np.zeros_like(sgt), SPACING)
check("missed small lesion -> fn=1", si["fn"] == 1)
check("missed -> f1 == 0.0", si["f1"] == 0.0)

sphantom = np.zeros((30, 30, 30), bool)
sphantom[5:7, 5:7, 5:7] = True           # phantom small pred, no GT
si = small_instance_region(np.zeros_like(sphantom), sphantom, SPACING)
check("phantom small pred -> fp=1", si["fp"] == 1)

print("== small-instance: large lesion excluded from small track ==")
big = np.zeros((30, 30, 30), bool)
cube(big, (15, 15, 15), 3, True)         # 343 mm^3 >= 27, not "small"
si = small_instance_region(big, big.copy(), SPACING)
check("large lesion not in small track (tp=0,fn=0)", si["tp"] == 0 and si["fn"] == 0)

print()
if _fails:
    print(f"FAILED {len(_fails)}: {_fails}")
    raise SystemExit(1)
print("ALL INVARIANTS PASSED")
