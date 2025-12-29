#!/usr/bin/env bash
set -euo pipefail

# End-to-end installer + test runner for the *real* Allen SDK pipeline.
#
# This creates a dedicated venv using Python 3.11 (recommended for allensdk),
# installs the project + E2E deps, and runs tests in tests_e2e/.
#
# Usage:
#   bash scripts/run_e2e_allensdk.sh
#
# Options:
#   PYTHON_BIN=/path/to/python3.11 bash scripts/run_e2e_allensdk.sh
#

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-e2e-allensdk}"

if [ -z "$PYTHON_BIN" ]; then
  for candidate in python3.11 python3.10 python3.9 python3.8; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: No compatible Python found. Install Python 3.11 (recommended) or set PYTHON_BIN to Python <3.12." >&2
  exit 2
fi

"$PYTHON_BIN" -V

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info >= (3, 12):
    raise SystemExit(
        "ERROR: allensdk E2E requires Python <3.12 (prefer 3.11). "
        "Set PYTHON_BIN=python3.11 (or 3.10/3.9) and rerun."
    )
PY

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install -U pip setuptools wheel

# allensdk and parts of the scientific stack are frequently not yet compatible with NumPy 2.
"$VENV_DIR/bin/python" -m pip install "numpy<2"

# Install project + minimal E2E deps. Prefer extras if possible.
"$VENV_DIR/bin/python" -m pip install -e "$REPO_ROOT[dev,e2e_allen]" || \
  "$VENV_DIR/bin/python" -m pip install -r "$REPO_ROOT/requirements-e2e-allensdk.txt"

# Live mode: use the real allensdk, not the offline stub.
export BWM_ALLEN_OFFLINE=0

# Persist Allen downloads across runs.
export BWM_E2E_CACHE_DIR="${BWM_E2E_CACHE_DIR:-${REPO_ROOT}/.cache-e2e/allensdk}"
mkdir -p "$BWM_E2E_CACHE_DIR"

"$VENV_DIR/bin/python" -m pytest -q "$REPO_ROOT/tests_e2e" -ra
