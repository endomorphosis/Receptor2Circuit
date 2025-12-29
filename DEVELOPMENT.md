# Development Installation

Install the package in development mode with testing dependencies:

```bash
pip install -e ".[dev]"
```

# Running Tests

Run all tests:
```bash
pytest
```

Run the full end-to-end test surface area (default + real allensdk E2E + live Allen tests):

```bash
bash scripts/run_all_tests.sh
```

This script will try to auto-detect a usable Python <3.12 for the Allen SDK pieces (including `~/.local/bin/python3.11` from `uv`).

Run with coverage:
```bash
pytest --cov=brainwidemap --cov-report=html
```

## Live integration tests (Allen SDK)

By default, the Allen connectivity test suite runs in a deterministic offline mode
(no `allensdk`, no network) so CI and local runs are stable.

Live Allen SDK tests are marked `allen_live` and are **deselected by default** so
the standard test run does not report skips.

To run the *live* Allen SDK integration tests (requires `allensdk` and usually cache/network):

```bash
# Prefer Python 3.11 for allensdk compatibility
pip install -e ".[dev,allen]"

# Disable offline stub
BWM_ALLEN_OFFLINE=0 pytest --run-allen-live -m allen_live
```

## End-to-end (E2E) Allen SDK installer + tests

For a single command that:
- creates a dedicated venv (Python 3.11 recommended),
- installs `allensdk` + this repo + dev deps,
- runs a true end-to-end test suite that is **not skipped** and will **fail** if dependencies/data are broken,

run:

```bash
bash scripts/run_e2e_allensdk.sh
```

If you don't have `python3.11` on PATH, you can point the script at any Python <3.12:

```bash
PYTHON_BIN=/path/to/python3.11 bash scripts/run_e2e_allensdk.sh
```

### Installing Python 3.11 without sudo (uv)

On machines where you don't have `python3.11` available (and you can't use `sudo`), you can install a user-local Python via `uv`:

```bash
python -m pip install -U uv
uv python install 3.11

# If uv warns that ~/.local/bin is not on PATH:
export PATH="$HOME/.local/bin:$PATH"

PYTHON_BIN="$HOME/.local/bin/python3.11" bash scripts/run_e2e_allensdk.sh
```

This runs `pytest tests_e2e/` with `BWM_ALLEN_OFFLINE=0`.

Run specific test file:
```bash
pytest tests/test_statistics.py
```

# Code Style

Format code with Black:
```bash
black brainwidemap/ tests/
```

Check code style with flake8:
```bash
flake8 brainwidemap/ tests/
```

# Type Checking

Run type checks with mypy:
```bash
mypy brainwidemap/
```
