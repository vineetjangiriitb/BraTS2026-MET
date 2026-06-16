# BraTS2026-MET

Work toward the BraTS 2026 Brain **Metastases (MET)** segmentation challenge.
Two completed phases: (A) dataset statistics, (B) nnU-Net end-to-end training run.

---

## Project layout

```
training/MICCAI-LH-BraTS2025-MET-Challenge-Training/
    BraTS-MET-XXXXX-TTT/          # non-UCSD: SRI24 atlas, 240×240×155 @ 1mm³ isotropic
    UCSD-Training/
        BraTS-MET-XXXXX-TTT/      # UCSD: native clinical space, anisotropic (0.12–0.6mm)
validation/Validation/             # 179 cases, MRI only (no seg — held by competition)
corrected-labels/                  # organiser-issued label corrections (2 cases, batch 1)
analysis/                          # Section A: dataset statistics
    dataset_stats.py               # A1–A2,A4–A6 — lesion counts, volumes, RC, modalities
    spatial_heatmap.py             # A3 — SRI24 spatial frequency heatmap
    01_treatment_status.py         # view: pre/post-treatment classification
    02_lesion_volumes.py           # view: per-lesion volume histogram
    03_lesion_counts.py            # view: lesion count distribution
    04_spatial_heatmap.py          # view: axial/MIP heatmap renders
    outputs/                       # all cached outputs (JSON, CSV, PNG, NIfTI, reports)
scripts/                           # Section B: nnU-Net training pipeline
    select_cases.py                # stratified 100-case selection from per_case.csv
    convert_to_nnunet.py           # BraTS format → nnUNet_raw layout + dataset.json
    verify_dataset.py              # local geometry/label integrity check
    runpod_setup.sh                # RunPod one-time env setup (nnunetv2, rclone, tmux)
    upload_data.sh                 # rclone Drive→pod dataset transfer
    train.sh                       # plan_and_preprocess + train (tmux, full logging)
    inference_and_eval.sh          # evaluate val predictions + optional competition inference
    download_results.sh            # rclone pod→Drive results transfer
    plot_training_curve.py         # parse training log → learning curve PNG
    case_list_100.txt              # the 100 selected case IDs (seed=42, reproducible)
runpod_results/                    # downloaded training artifacts (synced from pod)
    plans/                         # nnUNetPlans.json + dataset_fingerprint.json
    logs/                          # preprocess, train, eval stdout logs
    nnUNet_results/                # model checkpoints, val predictions, summary.json
    summary.json                   # final per-class Dice results
TRAINING_NOTES.md                  # first-principles explanation of every nnU-Net decision
RESULTS.md                         # complete record of the training run and results
SESSION_STATE.md                   # current project state (read this first each session)
```

Each case folder: `-t1c -t1n -t2f -t2w -seg` (`.nii.gz`).
Labels: `0`=background, `1`=NETC, `2`=SNFH/edema, `3`=ET, `4`=RC (UCSD only).

---

## Section A — Dataset statistics (complete)

**Key findings:**
- 1296 training timepoints: 650 non-UCSD (SRI24 atlas) + 646 UCSD (native space), both longitudinal
- 75.3% multi-focal; 10,119 ET lesions total; 41% sub-detection (<27mm³), only 19% Dice-evaluated (>275mm³)
- RC (resection cavity) only in UCSD cohort (25.9%); non-UCSD is entirely pre-treatment
- All 4 modalities present in 100% of cases
- SRI24 spatial hotspot: superior posterior right cerebrum, peak at voxel (143,88,102), 48/328 cases (14.6%)

**Critical gotcha:** lesion volumes must use per-case header voxel spacing. UCSD voxels ≈0.24mm³ — assuming 1mm³ inflates volumes ~4×.

```bash
# Reproduce from scratch (slow — reads off Drive, ~30 min)
python3 analysis/dataset_stats.py
python3 analysis/spatial_heatmap.py
# Fast view scripts (read cached outputs/per_case_records.json — seconds)
python3 analysis/01_treatment_status.py
python3 analysis/02_lesion_volumes.py
python3 analysis/03_lesion_counts.py
python3 analysis/04_spatial_heatmap.py
```

---

## Section B — nnU-Net training run (complete)

**Setup:** 100 non-UCSD SRI24-canonical cases, RTX 5090 on RunPod, nnunetv2 2.8.0 + torch 2.8.0 + CUDA 12.8.

**Auto-configured architecture (3d_fullres):**
- Patch size: 128×160×112 | Batch size: 2 | 6-stage U-Net, features 32→64→128→256→320→320
- InstanceNorm3d | LeakyReLU | ZScoreNormalization per channel | Dice + CE loss

**Final results (fold 0, 20 held-out val cases, 250 epochs):**

| Class | Dice |
|-------|------|
| NETC  | 0.460 |
| SNFH  | 0.596 |
| ET    | 0.661 |

See `RESULTS.md` for full analysis, `TRAINING_NOTES.md` for first-principles explanations.

```bash
# Re-run case selection + conversion locally
python3 scripts/select_cases.py
python3 scripts/convert_to_nnunet.py
python3 scripts/verify_dataset.py
# Plot training curve from downloaded log
python3 scripts/plot_training_curve.py runpod_results/logs/02_train_fold0.log
```
