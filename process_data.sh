#!/usr/bin/env bash
set -euo pipefail

# Locate the repository regardless of the current working directory.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "========================================"
echo "Prostate MRI metadata and EDA pipeline"
echo "========================================"

# Activate the project virtual environment if available.
if [[ -f ".venv/bin/activate" ]]; then
    source ".venv/bin/activate"
    echo "Activated .venv"
fi

# Check prerequisites.
if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: Python is not available."
    exit 1
fi

if [[ ! -d "Processed_data_nii" ]]; then
    echo "ERROR: Processed_data_nii/ not found."
    exit 1
fi

# Extract MRI metadata.
echo
echo "[1/2] Extracting MRI metadata..."
python extract_processed_nii_metadata.py

# Generate EDA visualizations.
echo
echo "[2/2] Generating EDA figures..."
python plot_prostate_eda.py

echo
echo "========================================"
echo "Processing completed successfully!"
echo "========================================"
echo "Metadata: processed_nii_metadata.csv"
echo "Figures:  eda_figures/"
