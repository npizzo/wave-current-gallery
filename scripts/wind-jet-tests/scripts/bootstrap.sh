#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'Unity x86_64 Linux required.'; exit 1; }
export REGIME_ROOT="$PWD"
source scripts/common.sh
"$PYTHON" -c 'import numpy,matplotlib; print("Analysis runtime",numpy.__version__,matplotlib.__version__)'
"$JULIA" --version
"$PYTHON" scripts/guard.py verify --root "$PWD"
echo 'Separate wind-forced jet campaign and pinned runtime verified. No simulation submitted.'
