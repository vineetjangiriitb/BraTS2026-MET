# SESSION_STATE — BraTS2026-MET

_Last updated: 2026-06-16_

## Update 2026-06-16 — Local eval harness BUILT (`eval_harness/`)
Reproduces the 24 Synapse-leaderboard metrics locally so we can score every
nnU-Net run without the daily submission limit. On branch `feature/eval-harness`
(repo now git-initialised on `main`; codegraph initialised + indexed).
- **Metrics**: `Lesionwise_dsc/nsd_mean_{et,tc,wt,rc}` + `Small_instance_{tp,fn,fp,f1}_{...}`.
  Regions ET={3},TC={1,3},WT={1,2,3},RC={4}. Lesion-wise: dilation=1, 26-conn,
  2mm³ thresh, miss→0. NSD computed at BOTH 0.5 & 1.0mm (prune after 1st anchor).
  Small-instance: GT lesions <27mm³, any-overlap detection, F1=TP/(TP+0.5(FP+FN)).
- **Provenance + every gap decision logged**: `eval_harness/METRIC_DECISIONS.md`
  (✅ verified-from-source vs 🟡 our-decision-pending-anchor). 6 `ANCHOR TODO`
  knobs to lock once we have ONE real leaderboard score.
- **Validated by construction**: `python eval_harness/validate.py` → 24 known-answer
  invariants all PASS. Full CLI smoke-tested end-to-end on synthetic NIfTI.
- **Run**: `python eval_harness/run_eval.py --pred DIR --gt DIR --out outputs`
  → per_case.csv + summary.json + console leaderboard table. Pairs by filename
  stem, strips nnU-Net `_0000` suffix. Works on fold_0/validation/ dump unchanged.
- Deps added to `.venv`: `surface-distance` (NSD). scipy used for 26-conn CC.
- NOT YET done: per-epoch checkpoint hook; wiring into `inference_and_eval.sh`;
  first real run against actual nnU-Net predictions; the leaderboard anchor.

---
_Earlier — last updated: 2026-06-04_

## Project
BraTS 2025/2026 Brain **Metastases (MET)** challenge. Goal right now: build an
initial understanding (dataset stats + competition framing + winning approaches)
to drive a kickoff presentation and subsequent model design.

## Dataset (as it sits on disk)
Two cohorts inside `training/MICCAI-LH-BraTS2025-MET-Challenge-Training/`:
- **non-UCSD** (650 timepoints, IDs 00001–~01056): **mostly NOT a clean atlas**.
  Only **328/650 are on the canonical 240×240×155 @ 1mm³ SRI24 grid**; the other
  322 span 57 distinct shapes (256³, 320³, 400×400×90, …) and ≥6 voxel sizes
  (1.0, 1.108, 2.06, 0.709 mm³ …). RC (resection cavity) never present.
- **UCSD** (646 timepoints, in `UCSD-Training/` subfolder, IDs 01057+):
  **native clinical space**, 74 distinct shapes (512×512×*, 256×256×*),
  voxels 0.12–0.6 mm³ — highly anisotropic and heterogeneous.

  => There is NO single common space. Resampling/registration is required
  before any cross-case spatial modeling. A3 heatmap is valid only on the
  328-case 240³ sub-cohort.
- Both cohorts are **longitudinal** — timepoint suffix -000..-006.
- Validation: 179 cases (`validation/Validation/`), no seg.
- Label scheme: 0=bg, 1=NETC, 2=SNFH/edema, 3=ET, 4=RC.
- All 4 modalities (t1c,t1n,t2f,t2w) present in 100% of all 1296 training cases.

## Done (Section A — dataset statistics)
Code in `analysis/`, outputs in `analysis/outputs/`.
- **A1** lesion-count distribution (ET connected components, 26-conn)
- **A2** per-lesion volume distribution vs 27/275 mm³ thresholds
- **A3** SRI24 spatial heatmap (running / done — see outputs/spatial_heatmap.png)
- **A4** pre/post-treatment via RC presence, by cohort & timepoint
- **A5** modality availability (100% complete everywhere)
- **A6** single vs multi-lesion breakdown

### Headline numbers
- 1296 training timepoints; 1250 (96.5%) have ≥1 ET lesion; 46 have zero ET.
- **75.3% of cases are multi-focal (2+ lesions)** — 37% are 6+ ("highly multi-focal", max 393).
- **10,119 individual ET lesions**; **41% are <27mm³** (sub-detection),
  39.5% in 27–275mm³ detection band, only **19.3% are >275mm³ (Dice-evaluated)**.
- Median lesion = 39.7mm³ (tiny). Mean dragged to 569mm³ by a long tail (max 96cc).
- RC only in UCSD (167/646 = 25.9%); RC fraction rises with timepoint (tp004 = 90%).
  non-UCSD is entirely pre-treatment / no-cavity.

## Critical gotcha (do not regress)
Volumes MUST use per-case header zooms, NOT a fixed 1mm³. UCSD voxels are
~0.24mm³ — assuming 1mm³ inflates UCSD volumes ~4× and corrupts the 27/275mm³
threshold analysis. `analyze_case()` already reads zooms correctly.

## Performance note
Reading .nii.gz directly off Google Drive is very slow (full A1–A6 pass over
1296 cases took ~30 min, CPU-bound on stream/decode). Before any training or
heavy preprocessing, copy unzipped data to local SSD.

## Update 2026-06-06 — requested per-analysis view scripts (done)
Built 4 thin "view builder" scripts that read the cached
`outputs/per_case_records.json` + saved heatmaps (no Drive re-scan) to emit the
specifically-requested CSVs/PNGs. Numbers verified == SECTION_A_REPORT.
- `analysis/01_treatment_status.py` -> `treatment_status.csv`
  (case_id,has_RC,has_ET,has_SNFH,has_NCR,treatment_status). 167 post / 1129 pre.
- `analysis/02_lesion_volumes.py` -> `lesion_volumes.csv` (per-lesion rows) +
  `lesion_volume_distribution.png` (log-x hist, 27/275 lines). 10,119 lesions,
  41.2/39.5/19.3% zones.
- `analysis/03_lesion_counts.py` -> `lesion_counts.csv` +
  `lesion_count_distribution.png`. 75.3% multi; max 393 (BraTS-MET-00406-000).
- `analysis/04_spatial_heatmap.py` (SRI24-only, 328 cases) ->
  `spatial_heatmap_axial.png` (6 slices), `_projections.png` (sag/cor MIP),
  `_representative.png` (z=91). Hotspot cluster ~(142,89,102), 48 cases (14.6%).
- Consolidated writeup: `analysis/outputs/FINAL_RESULTS.md`.
- Deps installed in project-local `.venv/` (system Python is PEP-668 managed).

## Update 2026-06-09 — nnU-Net educational training run (Section B) — LOCAL PREP DONE
Goal: train nnU-Net end-to-end on a 100-case subset as an educational run + for the
prof presentation. Reason from first principles; document every decision.

**Decisions made (first principles):**
- Subset = 100 of the 328 non-UCSD SRI24-canonical cases (240³ @ 1mm). WHY: nnU-Net
  fingerprints on the MEDIAN spacing/shape; mixing UCSD (0.2mm anisotropic) would
  poison the fingerprint. 4 classes only (RC/class-4 is UCSD-only). Stratified
  25/50/25 by lesion count. Reproducible (seed=42).
- GPU = RTX 5090 on RunPod ($0.99/hr). WHY: ~same total cost as 4090 (~$3.5) but
  trains faster (Blackwell) + 32GB VRAM headroom. ~$6–7 total run, inside $10 budget.
- Container disk only (NO network volume) — rclone results back to Drive at the end.
- 250 epochs via `nnUNetTrainer_250epochs` (nnU-Net has NO --max_num_epochs flag;
  no 300-epoch variant exists). fold 0 only (5-fold is the default; 1 fold for cost).
- Transfer via rclone (Drive→pod); data is on Drive, rsync can't pull from Drive.

**Done locally (all in `scripts/`):**
- `select_cases.py` → `case_list_100.txt` (ran: 328 pool → 100 selected 25/50/25).
- `convert_to_nnunet.py` → built nnUNet_raw at `~/brats_staging/` (400 imgs+100 lbls).
- `verify_dataset.py` → PASSED (geometry + labels {0,1,2,3} clean). Local stand-in
  for the GPU integrity check (Mac venv is py3.14, torch unsupported).
- Tarred → `~/brats_nnunet_raw.tar.gz` (1.5GB) + copied to Drive root of this project
  (`brats_nnunet_raw.tar.gz`) — syncing to cloud.
- RunPod scripts: `runpod_setup.sh`, `upload_data.sh`, `train.sh` (tmux),
  `inference_and_eval.sh`, `download_results.sh`, `plot_training_curve.py`.
- `TRAINING_NOTES.md` — full first-principles writeup for the presentation.

**RUNPOD RUN IN PROGRESS (started 2026-06-09 ~18:00 IST):**
- Pod: RTX 5090, image `pytorch:2.8.0-cu128` (torch 2.8.0+cu128, py3.12), 60GB disk,
  256 vCPU / 755GB RAM. SSH key = ~/.ssh/id_ed25519 (added to RunPod, pod restarted).
- nnunetv2 2.8.0 installed via `pip --break-system-packages` (PEP668 pod); torch untouched.
- Transfer LESSON: scp from Mac was ~63 KB/s (home upload bottleneck, ~6h ETA). Killed it.
  Used rclone gdrive→pod instead: 1.5GB in 29s (~52 MB/s datacenter backbone). rclone
  OAuth done headlessly via SSH port-forward (-L 53682) so Mac browser hit pod's callback.
- GOTCHA: macOS tar injected `._*` AppleDouble files → nnU-Net saw 200 cases. `find -name
  '._*' -delete` fixed it. (For next time: tar with COPYFILE_DISABLE=1.)
- nnU-Net PLAN (3d_fullres, key presentation artifact, backed up to Drive runpod_results/plans/):
  patch 128×160×112, batch 2, InstanceNorm3d, LeakyReLU, 6 stages 32→64→128→256→320→320,
  z-score per-channel, spacing 1mm, median cropped size 141×176×138. 3d_lowres dropped
  (no cascade needed). All matches TRAINING_NOTES.md predictions.
- Running `train.sh` in tmux session `brats`: preprocess (done planning) → train fold0
  250 epochs (nnUNetTrainer_250epochs). Monitoring via Monitor task.

**COMPLETED 2026-06-09 19:43 UTC. FINAL RESULTS (20 val cases, fold 0, 250 epochs):**
- NETC (class 1): Dice = 0.4600
- SNFH (class 2): Dice = 0.5961
- ET   (class 3): Dice = 0.6614
- HD95 = nan (nnU-Net's internal val metric; proper HD95 needs full nnUNetv2_evaluate_folder run)
- All results + model + logs on Drive: BraTS2026-MET/runpod_results/

**NEXT:**
- TERMINATE the RunPod pod immediately (billing still running).
- Local: run `plot_training_curve.py` on downloaded logs → training_curve.png.
- Analyse results, build presentation slides from TRAINING_NOTES.md + these numbers.

## Next (Section A leftovers)
- Eyeball `analysis/outputs/spatial_heatmap.png` for lesion clustering (A3).
- Build the kickoff presentation from these stats.
- Survey winning BraTS-MET approaches (not started).
- No git repo yet; no `.codegraph/` (not source-heavy — skip unless code grows).
