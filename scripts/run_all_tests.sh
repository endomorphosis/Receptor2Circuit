#!/usr/bin/env bash
set -euo pipefail

# Run the entire test surface area:
#  1) Default deterministic suite (Python 3.12+; offline Allen stub)
#  2) Real Allen SDK end-to-end suite (Python <3.12; typically 3.11)
#  3) Opt-in live Allen SDK tests (Python <3.12; requires allensdk)
#
# Usage:
#   bash scripts/run_all_tests.sh
#
# Options:
#   PYTHON_BIN=/path/to/python3.11 bash scripts/run_all_tests.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Help discover uv-installed python binaries.
if [ -d "$HOME/.local/bin" ]; then
	export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> (1/3) Default deterministic suite (.venv)"
"$REPO_ROOT/.venv/bin/python" -m pytest -q

echo "==> (2/3) Real Allen SDK E2E suite (.venv-e2e-allensdk)"
if [ -z "${PYTHON_BIN:-}" ]; then
	for candidate in "$HOME/.local/bin/python3.11" python3.11 python3.10 python3.9 python3.8; do
		if command -v "$candidate" >/dev/null 2>&1; then
			PYTHON_BIN="$candidate"
			break
		fi
	done
fi

PYTHON_BIN="${PYTHON_BIN:-}" bash "$REPO_ROOT/scripts/run_e2e_allensdk.sh"

echo "==> (3/3) Live Allen SDK tests (marked 'allen_live')"
# Reuse the E2E venv (it has allensdk installed if step 2 succeeded)
export BWM_ALLEN_OFFLINE=0
"$REPO_ROOT/.venv-e2e-allensdk/bin/python" -m pytest -q --run-allen-live -m allen_live tests -ra

echo "==> All test suites completed successfully."
