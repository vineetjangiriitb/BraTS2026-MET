#!/usr/bin/env python
"""BraTS-METS local evaluation harness — CLI.

Score a folder of predictions against a folder of ground-truth segmentations,
reproducing the Synapse leaderboard columns locally so you can evaluate every
run without the daily submission limit. See eval_harness/METRIC_DECISIONS.md.

Usage:
    python run_eval.py --pred PRED_DIR --gt GT_DIR [--out OUT_DIR]

Cases are paired by the filename stem (ignoring a trailing nnU-Net channel
suffix if present). Works directly on nnU-Net's fold_0/validation/ dump.

Outputs (in --out, default ./outputs):
    per_case.csv     one row per case + an aggregate MEAN row
    summary.json     machine-readable aggregate + per-case
    (and a leaderboard-style table printed to the console)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import score_case, column_order
from regions import REGIONS

# small-instance counts are summed across cases; everything else is meaned.
_COUNT_COLS = tuple(
    f"Small_instance_{k}_{r}" for r in REGIONS for k in ("tp", "fn", "fp")
)


def _stem(name: str) -> str:
    """Case id from a filename, dropping extension and nnU-Net _0000 suffix."""
    base = name
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    base = re.sub(r"_\d{4}$", "", base)  # strip channel suffix if any
    return base


def _index(folder: Path) -> dict:
    out = {}
    for p in sorted(folder.iterdir()):
        if p.name.endswith((".nii.gz", ".nii")):
            out[_stem(p.name)] = p
    return out


def main():
    ap = argparse.ArgumentParser(description="BraTS-METS local eval harness")
    ap.add_argument("--pred", required=True, type=Path, help="predictions folder")
    ap.add_argument("--gt", required=True, type=Path, help="ground-truth folder")
    ap.add_argument("--out", type=Path, default=Path("outputs"),
                    help="output folder (default: ./outputs)")
    args = ap.parse_args()

    preds, gts = _index(args.pred), _index(args.gt)
    cases = sorted(set(preds) & set(gts))
    missing_pred = sorted(set(gts) - set(preds))
    missing_gt = sorted(set(preds) - set(gts))
    if missing_pred:
        print(f"WARNING: {len(missing_pred)} GT cases have no prediction "
              f"(e.g. {missing_pred[:3]}) -- skipped")
    if missing_gt:
        print(f"WARNING: {len(missing_gt)} predictions have no GT "
              f"(e.g. {missing_gt[:3]}) -- skipped")
    if not cases:
        raise SystemExit("No paired cases found between --pred and --gt.")

    print(f"Scoring {len(cases)} cases...")
    rows = []
    for i, cid in enumerate(cases, 1):
        row = {"case": cid}
        row.update(score_case(preds[cid], gts[cid]))
        rows.append(row)
        print(f"  [{i}/{len(cases)}] {cid}", end="\r")
    print()

    cols = column_order()
    df = pd.DataFrame(rows)[["case"] + cols]

    # Aggregate: NaN-skipping mean (region-present cases only, sec 6),
    # counts summed across cases.
    agg = {"case": "MEAN"}
    for c in cols:
        vals = df[c].values
        if c in _COUNT_COLS:
            agg[c] = float(np.nansum(vals))
        elif np.isnan(vals).all():
            agg[c] = float("nan")  # region in no case -> n/a (avoid warning)
        else:
            agg[c] = float(np.nanmean(vals))
    df_out = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "per_case.csv"
    df_out.to_csv(csv_path, index=False)

    summary = {
        "n_cases": len(cases),
        "aggregate": {c: agg[c] for c in cols},
        "per_case": {r["case"]: {c: r.get(c) for c in cols} for r in rows},
    }
    json_path = args.out / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    _print_table(agg)
    print(f"\nWrote {csv_path}\n      {json_path}")


def _print_table(agg):
    print("\n" + "=" * 60)
    print("  BraTS-METS LOCAL LEADERBOARD  (aggregate over cases)")
    print("=" * 60)
    hdr = f"{'region':>6} | {'LW-DSC':>7} | {'NSD@0.5':>7} | {'NSD@1.0':>7} | {'sm-F1':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in REGIONS:
        def g(k):
            v = agg.get(k)
            return f"{v:.4f}" if v == v else "  n/a "  # NaN-safe
        print(f"{r.upper():>6} | {g(f'Lesionwise_dsc_mean_{r}'):>7} | "
              f"{g(f'Lesionwise_nsd0.5_mean_{r}'):>7} | "
              f"{g(f'Lesionwise_nsd1.0_mean_{r}'):>7} | "
              f"{g(f'Small_instance_f1_{r}'):>6}")
    print("-" * len(hdr))
    print("(NSD shown at both 0.5 & 1.0 mm; prune one after leaderboard anchor)")


if __name__ == "__main__":
    main()
