#!/bin/bash
# ============================================================================
# Evaluate the trained model and (optionally) predict on the competition val set.
# Run inside the same tmux session after train.sh finishes.
# ============================================================================
set -e

export nnUNet_raw=${nnUNet_raw:-/workspace/nnUNet_raw}
export nnUNet_preprocessed=${nnUNet_preprocessed:-/workspace/nnUNet_preprocessed}
export nnUNet_results=${nnUNet_results:-/workspace/nnUNet_results}

DATASET=1
CONFIG=3d_fullres
FOLD=0
TRAINER=nnUNetTrainer_250epochs
MODEL_DIR=$nnUNet_results/Dataset001_BraTsMET/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}

echo "############################################################"
echo "# EVALUATE on the 20 internal validation cases             "
echo "############################################################"
# KEY INSIGHT: nnU-Net AUTOMATICALLY predicted the 20 fold-0 validation cases at the
# end of training and wrote them to fold_0/validation/. Those 20 cases DO have ground
# truth labels (they came from our training set). So we evaluate those predictions
# directly against the ground-truth labelsTr — no extra inference pass needed.
# This is our real, presentable Dice + HD95.
#
# nnUNetv2_evaluate_folder signature: <gt_folder> <pred_folder> -djfile <dataset.json> -pfile <plans.json>
nnUNetv2_evaluate_folder \
    $nnUNet_raw/Dataset001_BraTsMET/labelsTr \
    $MODEL_DIR/validation \
    -djfile $nnUNet_raw/Dataset001_BraTsMET/dataset.json \
    -pfile $nnUNet_preprocessed/Dataset001_BraTsMET/nnUNetPlans.json \
    2>&1 | tee /workspace/logs/03_evaluation.log

echo ""
echo "=== Per-class Dice + HD95 summary ==="
cat $MODEL_DIR/validation/summary.json | python -c "
import sys, json
s = json.load(sys.stdin)
print('Mean over the 20 validation cases:')
for lab, m in s['mean'].items():
    print(f\"  class {lab}: Dice={m.get('Dice', float('nan')):.4f}  HD95={m.get('HD95', float('nan')):.2f} mm\")
" 2>/dev/null || echo "(see summary.json for full metrics)"

# ----------------------------------------------------------------------------
# OPTIONAL: predict on the BraTS competition validation set (179 cases, no labels).
# These predictions are what you submit to the BraTS leaderboard. They have NO
# local ground truth — the competition computes your score server-side.
# Requires the 179 val cases converted to nnU-Net format at /workspace/imagesTs.
# (We can build that with a converter if/when you want a leaderboard submission.)
# ----------------------------------------------------------------------------
if [ -d /workspace/imagesTs ]; then
  echo ""
  echo "############################################################"
  echo "# PREDICT on competition validation set (179 cases)        "
  echo "############################################################"
  nnUNetv2_predict \
      -i /workspace/imagesTs \
      -o /workspace/predictions_competition \
      -d $DATASET -c $CONFIG -tr $TRAINER -f $FOLD \
      2>&1 | tee /workspace/logs/04_inference_competition.log
fi

echo ""
echo "=== DONE. Next: bash download_results.sh ==="
