#!/bin/bash
# ============================================================================
# Push all results back to Google Drive BEFORE terminating the pod.
# Container disk is ephemeral — anything not copied out is lost on pod stop.
# ============================================================================
set -e

export nnUNet_results=${nnUNet_results:-/workspace/nnUNet_results}
export nnUNet_preprocessed=${nnUNet_preprocessed:-/workspace/nnUNet_preprocessed}

DEST="gdrive:BraTS2026-MET/runpod_results"

echo "=== Uploading trained model + logs to Google Drive ($DEST) ==="

# 1. Trained model weights, training log, val predictions, summary.json
rclone copy $nnUNet_results "$DEST/nnUNet_results" --progress

# 2. The planning artifacts (architecture + fingerprint) — key for the presentation
rclone copy $nnUNet_preprocessed/Dataset001_BraTsMET/nnUNetPlans.json "$DEST/plans/" --progress
rclone copy $nnUNet_preprocessed/Dataset001_BraTsMET/dataset_fingerprint.json "$DEST/plans/" --progress 2>/dev/null || true

# 3. All stdout logs we tee'd
rclone copy /workspace/logs "$DEST/logs" --progress

# 4. Competition predictions, if we generated them
if [ -d /workspace/predictions_competition ]; then
  rclone copy /workspace/predictions_competition "$DEST/predictions_competition" --progress
fi

echo ""
echo "=== Upload complete. Safe to terminate the pod. ==="
echo "Results are in your Drive at: BraTS2026-MET/runpod_results/"
