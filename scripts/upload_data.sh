#!/bin/bash
# ============================================================================
# Pull the 100-case dataset tarball from Google Drive onto the pod and extract.
# Run AFTER runpod_setup.sh and AFTER `rclone config` (remote named "gdrive").
# ============================================================================
set -e

# Path to the tarball on Google Drive, as rclone sees it.
# It lives in:  My Drive / BraTS2026-MET / brats_nnunet_raw.tar.gz
# With a standard "drive" remote rooted at My Drive, that is:
GDRIVE_TARBALL="gdrive:BraTS2026-MET/brats_nnunet_raw.tar.gz"

echo "=== Downloading dataset tarball from Google Drive ==="
rclone copy "$GDRIVE_TARBALL" /workspace/ --progress

echo "=== Extracting into /workspace ==="
cd /workspace
tar xzf brats_nnunet_raw.tar.gz      # creates /workspace/nnUNet_raw/Dataset001_BraTsMET
rm brats_nnunet_raw.tar.gz

echo "=== Verifying ==="
ls /workspace/nnUNet_raw/Dataset001_BraTsMET/
echo "imagesTr files: $(ls /workspace/nnUNet_raw/Dataset001_BraTsMET/imagesTr | wc -l) (expect 400)"
echo "labelsTr files: $(ls /workspace/nnUNet_raw/Dataset001_BraTsMET/labelsTr | wc -l) (expect 100)"
echo ""
echo "Dataset ready. Next: bash train.sh"

# ----------------------------------------------------------------------------
# ALTERNATIVE (if you prefer not to use rclone): from your Mac, push directly:
#   scp -P <pod_ssh_port> ~/brats_nnunet_raw.tar.gz root@<pod_ip>:/workspace/
# then on the pod:  cd /workspace && tar xzf brats_nnunet_raw.tar.gz
# ----------------------------------------------------------------------------
