# nnU-Net on BraTS-MET — Training Notes (First Principles)

This document explains every design decision in our nnU-Net training run, from first
principles. It is written to be lifted directly into your presentation. Each section
answers a "why" your professor is likely to ask.

---

## 0. What problem are we solving?

**Task:** 3D semantic segmentation of brain metastases. Given 4 co-registered MRI
volumes per patient, label every voxel as one of 4 classes:

| Label | Name | What it is |
|-------|------|------------|
| 0 | background | not tumor |
| 1 | NETC | necrotic / non-enhancing tumor core |
| 2 | SNFH | surrounding FLAIR hyperintensity (edema) |
| 3 | ET | enhancing tumor |

**Input:** 4 channels — `t1c` (T1 post-contrast), `t1n` (T1 native), `t2f` (FLAIR),
`t2w` (T2). Each modality highlights different tissue: ET is brightest on t1c; edema
is brightest on FLAIR. The network learns to fuse all four.

**Output:** a 240×240×155 integer label map per case.

---

## 1. Why nnU-Net at all?

nnU-Net ("no-new-net") is not a new architecture — it's a *self-configuring pipeline*
around a fairly standard U-Net. Its thesis: for medical segmentation, **how you
configure and train** a U-Net matters more than fancy architecture. It automates the
choices a human would otherwise hand-tune (patch size, normalization, augmentation,
learning rate schedule) by deriving them from the dataset itself. It has won or placed
in dozens of medical segmentation challenges, including prior BraTS editions, which is
exactly why it's the right baseline to learn on.

---

## 2. Why only 100 cases, and why the non-UCSD cohort?

The BraTS-MET training set has 1296 timepoints across two cohorts:
- **non-UCSD** (650): mostly the SRI24 atlas — 240×240×155 @ 1.0 mm³ isotropic.
- **UCSD** (646): native clinical space — 512×512×*, voxels 0.12–0.6 mm³, highly
  anisotropic, 70+ distinct shapes.

**The key first-principles reason we isolate one cohort:** nnU-Net's first step is
**dataset fingerprinting** — it computes the *median* voxel spacing and image shape
across all cases, and every downstream decision (patch size, resampling target,
network depth) is derived from that median. If we feed it a mix of 1.0 mm isotropic
and 0.2 mm anisotropic data, the median is a meaningless blend of two distributions:
it would try to resample everything to a compromise grid and plan a patch size that
fits neither cohort well — possibly one too large to fit in GPU memory.

By restricting to the **328 non-UCSD cases that sit on the canonical 240×240×155
grid**, the fingerprint is clean and unambiguous. We then sample **100** of those,
stratified by lesion count (25 single-lesion, 50 with 2–5, 25 with 6+), so the model
sees the natural multi-focal distribution. 100 cases is enough to converge a useful
model (prior BraTS work shows reasonable Dice from 50–80 cases) while keeping
preprocessing under 2 hours and training under ~$5 of GPU time.

Note: the RC label (class 4, resection cavity) appears **only** in UCSD cases, so our
4-class problem is the natural label set for this cohort.

---

## 3. The nnU-Net dataset format (and why it's shaped that way)

```
nnUNet_raw/Dataset001_BraTsMET/
  dataset.json
  imagesTr/<case>_0000.nii.gz   # channel 0 = t1c
  imagesTr/<case>_0001.nii.gz   # channel 1 = t1n
  imagesTr/<case>_0002.nii.gz   # channel 2 = t2f
  imagesTr/<case>_0003.nii.gz   # channel 3 = t2w
  labelsTr/<case>.nii.gz        # label map (no channel suffix)
```

The 4-digit suffix is the **channel index**. nnU-Net stacks channels 0..3 into the
network's input tensor in that fixed order, identically for every case, matching the
`channel_names` in `dataset.json`. Labels get no suffix because there's one label map
per case, not one per channel. This rigid convention is what lets nnU-Net be fully
automatic — it never has to guess which file is which.

---

## 4. Preprocessing: fingerprint → plan → preprocess

When you run `nnUNetv2_plan_and_preprocess`, three things happen:

### 4a. Fingerprint
nnU-Net scans every case and records:
- **Spatial stats:** median shape, median voxel spacing, anisotropy.
- **Intensity stats per modality:** mean, std, and 0.5 / 99.5 percentiles of the
  foreground (non-background) voxels.
- **Class frequencies:** how rare each label is (drives sampling during training).

### 4b. Plan (the heuristic that designs the network)
From the fingerprint, rule-based heuristics choose:
- **Target spacing:** the median spacing (here ~1.0 mm³). All cases get resampled to it.
- **Patch size:** the largest patch that (a) fits the GPU memory budget and (b) covers
  enough anatomical context. For 240³ @ 1 mm we expect roughly **192×192×128 or
  160×160×128**. (Check the actual value in `nnUNetPlans.json` after planning.)
- **Batch size:** chosen so one batch of patches fits in VRAM. For 3D it's typically
  **2**. (See §6 for why small batch size drives the normalization choice.)
- **Network depth:** number of downsampling stages, chosen so the smallest feature map
  is ~4–8 voxels. Feature channels start at 32 and double each stage, capped at 320.

### 4c. Preprocess
Each case is then: resampled to target spacing → **z-score normalized per modality
per case** (subtract that case's foreground mean, divide by its std) → cropped to the
foreground bounding box → saved as a compressed tensor for fast loading during training.

**Why per-case z-score for MRI (not global, not CT-style clipping)?** MRI intensities
are not physically calibrated — the same tissue can have very different raw values
across scanners/protocols. Normalizing each case to zero-mean/unit-variance removes
that scanner bias so the network sees consistent contrast. (CT, by contrast, *is*
calibrated in Hounsfield units, so nnU-Net uses global percentile clipping there
instead — different modality, different rule.)

---

## 5. Why patch-based training (not whole-volume)?

A full 240×240×155×4 volume plus all the U-Net activations and gradients won't fit in
24–32 GB of VRAM. So nnU-Net trains on **random patches** (e.g. 192×192×128) sampled
from each volume. To make sure the network sees enough tumor (which is tiny relative to
the brain), it **oversamples foreground**: a fixed fraction of patches are forced to be
centered on a labeled voxel. At inference time it tiles the whole volume with
overlapping patches and stitches the predictions (Gaussian-weighted to reduce seams).

---

## 6. Why Instance Normalization, not Batch Normalization?

Batch Norm computes its mean/variance statistics **across the batch**. With a 3D batch
size of only **2** (forced by memory), those statistics are computed over just 2 samples
— far too noisy to be a stable estimate. Instance Norm instead normalizes **each sample,
each channel independently**, so it's completely insensitive to batch size. This is why
nnU-Net (and most 3D medical segmentation) uses Instance Norm. Activation is Leaky ReLU
(slope 0.01) to avoid dead units.

---

## 7. Why Dice + Cross-Entropy loss?

The loss is the **sum of soft Dice loss and cross-entropy**, because each fixes the
other's weakness:

- **Cross-entropy** gives smooth, well-behaved per-voxel gradients, but it's dominated
  by the background class — in a brain volume that's >99% background, so CE alone barely
  cares about tiny tumors.
- **Dice loss** directly optimizes region overlap and is inherently normalized by region
  size, so it handles the extreme class imbalance — but its gradient is nearly flat when
  a prediction is already mostly right, which slows late-stage learning.

Summing them gives stable optimization (from CE) **and** imbalance robustness (from
Dice). This combination is one of nnU-Net's most important, empirically-validated
defaults.

---

## 8. Why 5-fold cross-validation — and why we run only fold 0

With only 100 cases, any single train/val split is a luck-of-the-draw estimate of
performance. nnU-Net's default is **5-fold CV**: it splits the data into 5 folds and
trains 5 models, each holding out a different 20% for validation, so every case is
validated exactly once. A real competition submission then **ensembles** all 5 models.

For this educational run we train **fold 0 only** (80 train / 20 val). That's enough to
(a) demonstrate the full pipeline end-to-end and (b) produce real Dice/HD95 numbers on
20 held-out cases — at one fifth the GPU cost. Scaling to all 5 folds later is just
re-running the train command with folds 1–4.

---

## 9. Why 250 epochs instead of the 1000 default?

nnU-Net uses a **cosine-annealed learning rate** that starts high and decays smoothly to
near-zero over the configured number of epochs (default 1000). On a 100-case dataset the
validation Dice curve typically plateaus by ~200–250 epochs — the remaining epochs buy
only marginal gains while costing the same per epoch. We use the predefined
`nnUNetTrainer_250epochs` trainer (nnU-Net has **no** `--max_num_epochs` flag; you select
training length by trainer variant: 250 / 500 / 750 / 1000). 250 epochs ≈ 3 hours on an
RTX 5090 ≈ ~$3, leaving budget headroom.

---

## 10. What is "pseudo-Dice" in the training log?

Computing true Dice over full validation volumes every epoch would be slow. So during
training nnU-Net reports **pseudo-Dice** — Dice computed on the network's patch outputs
from recent mini-batches. It tracks *relative* progress epoch-to-epoch (and the LR
schedule and checkpointing key off it), but it reads a few points lower than the true
Dice. The **true** per-class Dice + HD95 is computed only afterward, by
`nnUNetv2_evaluate_folder`, on the held-out validation predictions. Those final numbers
are what go in the presentation.

---

## 11. Inference and the BraTS competition validation set

Two distinct "validation" concepts, do not confuse them:

- **Internal validation (20 cases):** held out from our 100 training cases by fold 0.
  We have their ground-truth labels, so we get real local Dice/HD95. nnU-Net even
  predicts these automatically at the end of training (in `fold_0/validation/`).
- **Competition validation (179 cases):** released by BraTS with **no labels**. The
  organizers hold the ground truth on their server. You run inference to produce
  segmentation `.nii.gz` files and submit those to the leaderboard, which scores them
  server-side. You never see those labels locally — that's how the challenge prevents
  overfitting to the validation set.

So: you cannot "train on" or locally evaluate against the competition val set. Your local
quality signal comes entirely from the internal fold-0 validation.

---

## 12. Metrics: Dice and HD95

- **Dice coefficient** = `2·|pred ∩ gt| / (|pred| + |gt|)`, per class, range 0–1. Measures
  volumetric overlap. 1.0 = perfect. This is the primary BraTS metric.
- **HD95 (95th-percentile Hausdorff distance)**, in mm: how far apart the predicted and
  true boundaries are, taking the 95th percentile to ignore a few outlier voxels. Lower
  is better. Captures boundary/shape errors that Dice can miss.

**Expected ballpark for this run (100 cases, fold 0, 250 epochs):** ET Dice ~0.65–0.80,
SNFH ~0.70–0.85, NETC ~0.50–0.70. Respectable for an educational subset run; full
competitors use all 1296 cases, 1000 epochs, and a 5-fold ensemble.

**Actual results from this run (2026-06-09):**

| Class | True Val Dice | Notes |
|-------|---------------|-------|
| NETC  | 0.460 | Hardest class — small, no unique MRI signature, only 224/100 cases have it |
| SNFH  | 0.596 | Large edema region; lower than pseudo-Dice suggested → some overfitting |
| ET    | 0.661 | Primary BraTS metric; solid for 100-case single-fold run |

The true val Dice came in ~15–20 points below the pseudo-Dice tracked during training
(which peaked at NETC=0.64, SNFH=0.85, ET=0.80 at epoch 179). This gap is larger than
the typical 3–5 points and indicates **overfitting** — with only 80 training cases the
model partially memorized training-specific patterns rather than fully generalizing.
This is a valuable finding: it motivates more data, stronger augmentation, and/or
all-5-fold ensembling for a real competition submission.

---

## 13. The full pipeline, one line each

```
select_cases.py        # pick 100 balanced canonical cases
convert_to_nnunet.py   # build nnUNet_raw (4 channels + label per case) + dataset.json
verify_dataset.py      # local geometry/label sanity check (stand-in for the GPU check)
runpod_setup.sh        # install nnunetv2 + rclone + tmux, set env vars  [on pod]
upload_data.sh         # rclone the dataset tarball from Drive to pod    [on pod]
train.sh               # plan_and_preprocess  ->  train fold 0           [on pod, tmux]
inference_and_eval.sh  # evaluate fold-0 val + (optional) predict comp. set [on pod]
download_results.sh    # rclone model + logs + predictions back to Drive [on pod]
plot_training_curve.py # parse the log into a learning-curve PNG         [local]
```
