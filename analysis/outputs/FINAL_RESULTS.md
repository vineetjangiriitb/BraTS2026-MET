# BraTS2026-MET — Dataset Analysis: Final Results

_Generated: 2026-06-06_

Four analyses over the BraTS2025-MET training set (1296 timepoints: 650 non-UCSD
+ 646 UCSD). Labels: **1=NCR/NETC, 2=SNFH/edema, 3=ET, 4=RC (resection cavity)**.

> **Methodology note.** Lesions = connected components of the ET mask (label 3,
> 26-connectivity, `scipy.ndimage.label`). All volumes use each case's **own
> NIfTI header voxel spacing** — SRI24 cases are 1 mm³, UCSD cases are
> native/anisotropic (~0.24 mm³). Using a fixed 1 mm³ would inflate UCSD volumes
> ~4× and corrupt the 27/275 mm³ thresholds. These view scripts read the cached
> `per_case_records.json` (built by `dataset_stats.py` with the correct
> per-case spacing), so the numbers equal a fresh nibabel pass without the
> ~30-min Drive re-scan.

---

## 1. Pre- vs post-treatment (resection cavity = label 4)
Script: `analysis/01_treatment_status.py` → `outputs/treatment_status.csv`

| Status | Count | % |
|---|---:|---:|
| Pre-treatment (no RC) | 1129 | 87.1% |
| **Post-treatment (RC present)** | **167** | **12.9%** |

- RC appears **only in UCSD** cases (167/646 = 25.9% of UCSD). Every non-UCSD
  case (650) is pre-treatment / no cavity.
- Post-treatment fraction **rises with timepoint** within UCSD: tp000 0.7% →
  tp001 28% → tp002 30% → tp003 33% → **tp004 90.5%** (longitudinal post-op).
- All 167 post-treatment case IDs are listed in the script output and the CSV.
- CSV columns: `case_id, has_RC, has_ET, has_SNFH, has_NCR, treatment_status`.

## 2. Per-lesion ET volume distribution
Script: `analysis/02_lesion_volumes.py`
→ `outputs/lesion_volumes.csv`, `outputs/lesion_volume_distribution.png`

**Total individual ET lesions across dataset: 10,119**

| Zone | Range | Count | % |
|---|---|---:|---:|
| Detection only | < 27 mm³ | 4166 | 41.2% |
| Middle zone | 27–275 mm³ | 3997 | 39.5% |
| Segmentation (Dice-evaluated) | > 275 mm³ | 1956 | 19.3% |

- Median lesion **39.7 mm³** (tiny); mean **569.1 mm³** (dragged up by a long
  tail — largest single lesion ≈ 96 cc).
- p5 = **2.6 mm³**, p95 = **2677.9 mm³**.
- **Takeaway:** ~41% of lesions sit below the 27 mm³ detection floor and only
  ~19% are large enough to be Dice-scored — detection/FROC, not Dice, dominates.
- CSV columns: `case_id, lesion_id, volume_mm3, zone`.

## 3. Lesion count per case — single vs multi-lesion
Script: `analysis/03_lesion_counts.py`
→ `outputs/lesion_counts.csv`, `outputs/lesion_count_distribution.png`

| Category | Count | % of 1296 |
|---|---:|---:|
| No ET lesion (0) | 46 | 3.5% |
| Single-lesion (1) | 274 | 21.1% |
| **Multi-lesion (2+)** | **976** | **75.3%** |

- Among cases that have ≥1 lesion: 21.9% single / **78.1% multi**.
- **Most lesions in one case: `BraTS-MET-00406-000` with 393 lesions.**
- Distribution is heavy at 1–5 lesions with a long tail (histogram uses log-y).
- CSV columns: `case_id, num_lesions, single_or_multi`.

## 4. Spatial tumor-frequency heatmap (SRI24 atlas only)
Script: `analysis/04_spatial_heatmap.py`
→ `outputs/spatial_heatmap_axial.png`, `outputs/spatial_heatmap_projections.png`,
  `outputs/spatial_heatmap_representative.png`

- Restricted to the **328 SRI24 cases** (shape 240×240×155, 1 mm³). UCSD /
  native-space cases excluded — not voxel-comparable. Binary "any tumor label
  (1/2/3)" masks summed → frequency map → probability map (÷328).
- **Peak co-occurrence: 48 cases (14.6%)** at the hottest voxel.
- **Most representative axial slice: z = 91** (highest mean tumor frequency).
- Figures: 6 axial slices (z=40,60,80,100,120,140) with `hot` overlay; sagittal
  + coronal MIPs; single most-representative slice.
- **Top-5 most common tumor voxel locations** (SRI24 voxel coords x,y,z),
  all clustered together near (≈142, ≈89, 102):

  | Rank | (x, y, z) | Cases | % | Rough region |
  |---|---|---:|---:|---|
  | 1 | (143, 88, 102) | 48 | 14.6% | superior posterior right |
  | 2 | (142, 90, 102) | 47 | 14.3% | superior posterior right |
  | 3 | (142, 88, 102) | 47 | 14.3% | superior posterior right |
  | 4 | (143, 89, 102) | 47 | 14.3% | superior posterior right |
  | 5 | (141, 89, 102) | 47 | 14.3% | superior posterior right |

  The hotspot is a single tight cluster (a likely common metastasis site in the
  posterior cerebrum / parieto-occipital region); region labels are a coarse
  coordinate heuristic, not an atlas lookup. The coronal MIP shows the expected
  bilateral grey/white-matter-junction predominance typical of mets.

---

## Output files (in `analysis/outputs/`)
| File | From |
|---|---|
| `treatment_status.csv` | Script 1 |
| `lesion_volumes.csv`, `lesion_volume_distribution.png` | Script 2 |
| `lesion_counts.csv`, `lesion_count_distribution.png` | Script 3 |
| `spatial_heatmap_axial.png`, `spatial_heatmap_projections.png`, `spatial_heatmap_representative.png` | Script 4 |
| `per_case_records.json`, `per_case.csv`, `SECTION_A_REPORT.txt` | source `dataset_stats.py` (pre-existing) |
| `et_heatmap.nii.gz`, `wt_heatmap.nii.gz` | source `spatial_heatmap.py` (pre-existing) |

## Verification
- CSV row counts: treatment 1296, lesion_volumes 10,119, lesion_counts 1296 ✓
- Zone tally (4166/3997/1956) and single/multi tally (274/976/46) match report ✓
- All 5 PNGs render correct anatomy / distributions (visually inspected) ✓
- Numbers consistent with the established `SECTION_A_REPORT.txt`.
