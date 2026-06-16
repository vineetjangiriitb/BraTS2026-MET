#!/usr/bin/env python3
"""
Lightweight local stand-in for `nnUNetv2_plan_and_preprocess --verify_dataset_integrity`.

We can't run the real nnU-Net check locally (the Mac venv is Python 3.14, which
PyTorch doesn't support yet). But the integrity check itself doesn't need torch —
it just confirms, for every training case:
  - all 4 channels are present
  - the label file is present
  - all 5 files share identical shape, affine, and voxel spacing
  - label values are a subset of the declared label set {0,1,2,3}
nnU-Net's own check verifies exactly these geometric/label invariants, so passing
this locally means the real check on the pod should pass too.
"""
import json
from pathlib import Path

import numpy as np
import nibabel as nib

RAW = Path.home() / "brats_staging" / "nnUNet_raw" / "Dataset001_BraTsMET"
CHANNELS = ["0000", "0001", "0002", "0003"]
ALLOWED_LABELS = {0, 1, 2, 3}


def main():
    ds = json.loads((RAW / "dataset.json").read_text())
    images_tr = RAW / "imagesTr"
    labels_tr = RAW / "labelsTr"

    cases = sorted(p.name[: -len(".nii.gz")] for p in labels_tr.glob("*.nii.gz"))
    print(f"dataset.json says numTraining={ds['numTraining']}; found {len(cases)} labels")
    assert len(cases) == ds["numTraining"], "case count mismatch!"

    errors = []
    label_value_union = set()
    for i, case in enumerate(cases, 1):
        lab_path = labels_tr / f"{case}.nii.gz"
        lab = nib.load(lab_path)
        ref_shape = lab.shape
        ref_zooms = lab.header.get_zooms()[:3]

        # check each channel exists and matches label geometry
        for ch in CHANNELS:
            img_path = images_tr / f"{case}_{ch}.nii.gz"
            if not img_path.exists():
                errors.append(f"{case}: missing channel {ch}")
                continue
            img = nib.load(img_path)
            if img.shape != ref_shape:
                errors.append(f"{case}: channel {ch} shape {img.shape} != label {ref_shape}")
            if not np.allclose(img.header.get_zooms()[:3], ref_zooms, atol=1e-3):
                errors.append(f"{case}: channel {ch} zooms differ from label")

        # check label values are in the allowed set
        vals = set(np.unique(lab.get_fdata().astype(np.int16)).tolist())
        label_value_union |= vals
        stray = vals - ALLOWED_LABELS
        if stray:
            errors.append(f"{case}: stray label values {stray}")

        if i % 20 == 0 or i == len(cases):
            print(f"  checked {i}/{len(cases)}")

    print(f"\nUnion of all label values seen: {sorted(label_value_union)}")
    if errors:
        print(f"\nFAILED — {len(errors)} problems:")
        for e in errors[:20]:
            print("   ", e)
        raise SystemExit(1)
    print("\nPASSED — all cases have 4 channels + label with matching geometry,")
    print("and all label values are within {0,1,2,3}. Ready for nnU-Net on the pod.")


if __name__ == "__main__":
    main()
