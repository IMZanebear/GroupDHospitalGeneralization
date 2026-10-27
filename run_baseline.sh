#!/usr/bin/env bash
# Stop on failed commands (error e), undefined variables (unset u), or failures inside pipelines (pipeline failure capture -o pipefail).
# This prevents training from continuing if data preparation fails.
set -euo pipefail

# Run from your project directory after activating the nnU-Net environment.
# nnU-Net uses three directories: raw datasets, processed data/plans, and trained models.
# If these variables are already set, preserve them; otherwise use folders under $PWD.
export nnUNet_raw="${nnUNet_raw:-$PWD/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PWD/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PWD/nnUNet_results}"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# Dataset ID must match the DatasetXXX_RUNMC folder created by the Python script.
# 101 is an arbitrary ID, not the SAML dataset number.
DATASET_ID="${DATASET_ID:-101}"
# First two positional arguments are the original MRI and mask folders.
# The :? syntax prints an error if a required argument is missing.
# first command-line argument
IMAGES="${1:?Usage: bash run_baseline.sh /path/to/RUNMC/images /path/to/RUNMC/masks}"
# second command-line argument
MASKS="${2:?Provide the matching RUNMC mask directory}"

# Validate image/mask pairs, merge CG/PZ labels, and write nnU-Net files.
python prepare_runmc.py --images "$IMAGES" --labels "$MASKS" --dataset-id "$DATASET_ID"
# Official nnU-Net v2 command: inspect dataset, choose spacing/patch size,
# and preprocess images and labels using the generated plans.
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity
# Train five separate models. In each fold, about 24 RUNMC patients train
# and 6 validate (for a 30-case dataset); splits are patient-level by case.
# --npz saves validation probabilities for later model/ensemble analysis.
for fold in 0 1 2 3 4; do
  nnUNetv2_train "$DATASET_ID" 2d "$fold" --npz
done

# Optional external evaluation: after training, use all five fold models
# together on images from an unseen site. No target-site labels are used for training.
# Ensure target images are converted to nnU-Net naming (_0000.nii.gz) first.
# Optional after training: predictions on an unseen site prepared as imagesTs:
# nnUNetv2_predict -i /path/to/target/imagesTs -o /path/to/predictions -d "$DATASET_ID" -c 2d -f 0 1 2 3 4
