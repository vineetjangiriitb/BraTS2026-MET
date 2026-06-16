#!/bin/bash
# ============================================================================
# RunPod one-time setup. Run this FIRST after SSHing into a fresh pod.
# Installs nnU-Net, rclone, tmux, and sets the three nnU-Net env vars.
# ============================================================================
set -e

echo "=== Installing system tools (tmux, curl, unzip) ==="
apt-get update -qq && apt-get install -y -qq tmux curl unzip < /dev/null || true

echo "=== Installing nnU-Net v2 ==="
pip install --quiet nnunetv2

echo "=== Installing rclone (for Google Drive transfer) ==="
curl -s https://rclone.org/install.sh | bash || true

echo "=== Setting nnU-Net environment variables ==="
# These three dirs are nnU-Net's contract: raw input, preprocessed cache, results.
# We put them on /workspace (the pod's large container disk).
cat >> ~/.bashrc <<'EOF'

# --- nnU-Net environment ---
export nnUNet_raw=/workspace/nnUNet_raw
export nnUNet_preprocessed=/workspace/nnUNet_preprocessed
export nnUNet_results=/workspace/nnUNet_results
EOF

export nnUNet_raw=/workspace/nnUNet_raw
export nnUNet_preprocessed=/workspace/nnUNet_preprocessed
export nnUNet_results=/workspace/nnUNet_results
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results /workspace/logs

echo ""
echo "=== Setup complete. Verify: ==="
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
nnUNetv2_plan_and_preprocess -h > /dev/null 2>&1 && echo "nnUNetv2 CLI OK"
echo ""
echo "NEXT STEPS:"
echo "  1. Configure rclone for Google Drive:   rclone config"
echo "     (create a remote named 'gdrive', type=drive, follow the browser OAuth)"
echo "  2. Then run:   bash upload_data.sh"
