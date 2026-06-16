#!/bin/bash
# ============================================================================
# Preprocess + train nnU-Net on the 100-case BraTS-MET dataset.
# RUN THIS INSIDE A TMUX SESSION so it survives SSH disconnects:
#     tmux new -s brats
#     bash train.sh
#     (detach with Ctrl-b then d ; reattach later with `tmux attach -t brats`)
# ============================================================================
set -e

# Ensure env vars are loaded (in case this runs in a fresh shell)
export nnUNet_raw=${nnUNet_raw:-/workspace/nnUNet_raw}
export nnUNet_preprocessed=${nnUNet_preprocessed:-/workspace/nnUNet_preprocessed}
export nnUNet_results=${nnUNet_results:-/workspace/nnUNet_results}
mkdir -p /workspace/logs

DATASET=1
CONFIG=3d_fullres
FOLD=0
# WHY nnUNetTrainer_250epochs: nnU-Net has NO --max_num_epochs flag. To train
# fewer than the 1000-epoch default you pass a predefined trainer variant via -tr.
# nnU-Net ships 250/500/750/... variants (no 300). On 100 cases, the cosine-annealed
# loss is essentially converged by ~250 epochs; 500 doubles GPU cost for marginal
# Dice gain. 250 keeps us well inside the $10 budget (~$3 on a 5090).
TRAINER=nnUNetTrainer_250epochs

echo "############################################################"
echo "# PHASE 1: PLAN + PREPROCESS (fingerprint -> architecture)  "
echo "############################################################"
# This reads all 100 cases, computes the dataset fingerprint (median spacing/shape,
# per-modality intensity percentiles, class frequencies), then PLANS the network
# (patch size, batch size, pooling depth) and PREPROCESSES (resample + z-score
# normalize + crop) into .npz tensors. --verify_dataset_integrity is the official
# integrity check. -np 8 uses 8 CPU workers.
nnUNetv2_plan_and_preprocess -d $DATASET --verify_dataset_integrity -np 8 \
    2>&1 | tee /workspace/logs/01_preprocess.log

echo ""
echo "=== Architecture nnU-Net planned (read this for the presentation): ==="
cat $nnUNet_preprocessed/Dataset001_BraTsMET/nnUNetPlans.json \
    | python -c "import sys,json; p=json.load(sys.stdin); c=p['configurations']['$CONFIG']; print(json.dumps({k:c[k] for k in ['patch_size','batch_size','spacing','UNet_class_name','features_per_stage' if 'features_per_stage' in c.get('architecture',{}).get('arch_kwargs',{}) else 'architecture']}, indent=2, default=str))" \
    2>/dev/null || cat $nnUNet_preprocessed/Dataset001_BraTsMET/nnUNetPlans.json

echo ""
echo "############################################################"
echo "# PHASE 2: TRAIN  ($CONFIG, fold $FOLD, $TRAINER)           "
echo "############################################################"
# --npz saves softmax outputs (needed if we later want postprocessing/ensembling).
# Training on 80 cases, validating on 20 (fold 0 of nnU-Net's internal 5-fold split).
nnUNetv2_train $DATASET $CONFIG $FOLD -tr $TRAINER --npz \
    2>&1 | tee /workspace/logs/02_train_fold0.log

echo ""
echo "=== TRAINING COMPLETE ==="
echo "Model + logs: $nnUNet_results/Dataset001_BraTsMET/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}/"
echo "Next: bash inference_and_eval.sh"
