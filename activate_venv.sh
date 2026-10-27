# run with source, not bash
# source activate_venv.sh

#!/usr/bin/env bash

# Locate the directory containing this script.
# This allows activation from any working directory.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check that the virtual environment exists.
if [[ ! -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
    echo "Error: .venv not found. Run first-time setup."
    # Redirects 2 (standard error (stderr)) to /dev/null
    return 1 2>/dev/null || exit 1
fi

# Activate the virtual environment in the current shell.
source "$PROJECT_DIR/.venv/bin/activate"

echo "BDD Python environment activated."
echo "Python: $(command -v python)"
