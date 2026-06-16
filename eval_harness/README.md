# BraTS-METS Local Evaluation Harness

Reproduces the BraTS-METS 2025 Synapse leaderboard metrics **locally**, so you can
score every training run without hitting the daily submission limit. Works on a
fully-trained model or on the held-out predictions nnU-Net dumps mid-training —
it just scores `(prediction, ground_truth)` pairs and is agnostic to training state.

## The 24 metrics

For each region `{ET, TC, WT, RC}`:

| Block | Columns | What it measures |
|-------|---------|------------------|
| Lesion-wise DSC | `Lesionwise_dsc_mean_<r>` | per-lesion Dice, averaged; missed/phantom lesion = 0 |
| Lesion-wise NSD | `Lesionwise_nsd0.5_mean_<r>`, `Lesionwise_nsd1.0_mean_<r>` | per-lesion surface dice at 0.5 & 1.0 mm tolerance |
| Small-instance | `Small_instance_{tp,fn,fp,f1}_<r>` | detection of lesions < 27 mm³ (F1 = TP/(TP+0.5(FP+FN))) |

Regions: `ET={3}`, `TC={1,3}`, `WT={1,2,3}`, `RC={4}` (dataset labels
`1=NETC, 2=SNFH, 3=ET, 4=RC`). A region absent from the GT is scored NaN for that
case and skipped in aggregation.

> **We currently emit NSD at BOTH 0.5 and 1.0 mm.** The leaderboard reports one of
> them; prune the other once you have a single real leaderboard score to anchor
> against. Every such open knob is logged in `METRIC_DECISIONS.md` and marked
> `ANCHOR TODO` in the code.

## Usage

```bash
source ../.venv/bin/activate          # numpy, scipy, nibabel, pandas, surface-distance
cd eval_harness

# score any two folders of .nii.gz (GT + predictions)
python run_eval.py --pred /path/to/predictions --gt /path/to/labels --out outputs

# directly on nnU-Net's end-of-training fold-0 dump:
python run_eval.py \
    --pred $nnUNet_results/Dataset001_BraTsMET/<trainer>__nnUNetPlans__3d_fullres/fold_0/validation \
    --gt   $nnUNet_raw/Dataset001_BraTsMET/labelsTr \
    --out  outputs
```

Cases are paired by filename stem; a trailing nnU-Net `_0000` channel suffix is
stripped automatically.

## Outputs (in `--out`)

- `per_case.csv` — one row per case (all 24 columns) + a final `MEAN` aggregate row.
- `summary.json` — machine-readable aggregate + per-case (for tracking across runs).
- A leaderboard-style table printed to the console.

## Files

| File | Role |
|------|------|
| `regions.py` | label map → binary region masks |
| `lesionwise.py` | 26-conn CC matching → per-lesion DSC + NSD |
| `small_instance.py` | small-lesion detection → TP/FN/FP/F1 |
| `metrics.py` | score one case → 24-column dict |
| `run_eval.py` | CLI: pair folders, aggregate, write 3 outputs |
| `validate.py` | known-answer invariant tests (run this after any change) |
| `METRIC_DECISIONS.md` | every constant + gap-resolution, with provenance |

## Validation

No leaderboard anchor yet, so the harness is validated **by construction**:

```bash
python validate.py        # 24 known-answer invariants, all must PASS
```

When you get your first real leaderboard score, reconcile the 6 `ANCHOR TODO`
knobs in `METRIC_DECISIONS.md` §8 — each is a one-line parameter change, not a
rebuild.
