#!/usr/bin/env python3
"""
Convert the selected 100 BraTS-MET cases into the nnU-Net v2 raw dataset format.

nnU-Net v2 REQUIRED LAYOUT:
  nnUNet_raw/Dataset001_BraTsMET/
    dataset.json
    imagesTr/
      <case>_0000.nii.gz   channel 0
      <case>_0001.nii.gz   channel 1
      <case>_0002.nii.gz   channel 2
      <case>_0003.nii.gz   channel 3
    labelsTr/
      <case>.nii.gz        (NO channel suffix on labels)

CHANNEL MAPPING (the 4-digit suffix is the channel index nnU-Net feeds the network):
  t1c -> _0000   (T1 post-contrast: enhancing tumor is brightest here)
  t1n -> _0001   (T1 native)
  t2f -> _0002   (T2 FLAIR: edema / SNFH is brightest here)
  t2w -> _0003   (T2 weighted)
  The ORDER must be identical for every case and must match dataset.json channel_names.

WHY REAL COPIES INTO A LOCAL STAGING DIR (not symlinks on Drive):
  The source data lives on a Google Drive FUSE mount. We copy into a plain local
  directory so the result can be tar'd and uploaded to the pod intact (symlinks
  break across tar/upload/extract). 100 cases x ~16 MB = ~1.6 GB — trivial to copy.

LABELS:
  Non-UCSD cases use labels {0,1,2,3} (no RC/class 4 — that's UCSD-only). We copy
  the seg file verbatim; values are already 0/1/2/3 so no remapping is needed.
  (Verified: select_cases.py only admits cases whose labels_present subset {0,1,2,3}.)
"""
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CASE_LIST = Path(__file__).resolve().parent / "case_list_100.txt"
SRC_ROOT = PROJECT / "training" / "MICCAI-LH-BraTS2025-MET-Challenge-Training"

# Staging on the LOCAL home disk, NOT on the Google Drive mount, so tar is fast
# and we don't write 2 GB back through the Drive sync client.
STAGING = Path.home() / "brats_staging" / "nnUNet_raw" / "Dataset001_BraTsMET"

# modality suffix in source file -> nnU-Net channel index
CHANNELS = {"t1c": "0000", "t1n": "0001", "t2f": "0002", "t2w": "0003"}


def main():
    case_ids = [c.strip() for c in CASE_LIST.read_text().splitlines() if c.strip()]
    print(f"Converting {len(case_ids)} cases -> {STAGING}")

    images_tr = STAGING / "imagesTr"
    labels_tr = STAGING / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    missing = []
    for i, case in enumerate(case_ids, 1):
        case_dir = SRC_ROOT / case
        # 4 modality channels
        ok = True
        for mod, ch in CHANNELS.items():
            src = case_dir / f"{case}-{mod}.nii.gz"
            if not src.exists():
                missing.append(str(src))
                ok = False
                continue
            dst = images_tr / f"{case}_{ch}.nii.gz"
            shutil.copy2(src, dst)
        # label
        seg = case_dir / f"{case}-seg.nii.gz"
        if not seg.exists():
            missing.append(str(seg))
            ok = False
        else:
            shutil.copy2(seg, labels_tr / f"{case}.nii.gz")

        if i % 20 == 0 or i == len(case_ids):
            print(f"  {i}/{len(case_ids)} cases done")

    if missing:
        print(f"\nWARNING: {len(missing)} missing source files:")
        for m in missing[:10]:
            print("   ", m)
        raise SystemExit("Aborting — fix missing files before proceeding.")

    # dataset.json — nnU-Net reads this to know channels, labels, and case count.
    dataset_json = {
        "channel_names": {"0": "T1c", "1": "T1n", "2": "T2f", "3": "T2w"},
        "labels": {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3},
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
        "description": "BraTS2026-MET 100-case non-UCSD SRI24 educational run",
    }
    (STAGING / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"\nWrote dataset.json ({len(case_ids)} training cases, 4 classes)")
    print(f"nnUNet_raw dataset ready at: {STAGING}")
    print(f"\nNext: tar it up for upload:")
    print(f"  cd {STAGING.parent.parent}")
    print(f"  tar czf ~/brats_nnunet_raw.tar.gz nnUNet_raw/")


if __name__ == "__main__":
    main()
