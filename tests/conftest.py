import os

import pytest


def pytest_configure(config):
    # Ensure Allen connectivity tests run deterministically without requiring
    # allensdk/network access. Users can override by exporting BWM_ALLEN_OFFLINE=0.
    os.environ.setdefault("BWM_ALLEN_OFFLINE", "1")


def pytest_addoption(parser):
    parser.addoption(
        "--run-allen-live",
        action="store_true",
        default=False,
        help="Run live Allen SDK integration tests (marked 'allen_live').",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-allen-live"):
        return

    deselected = [item for item in items if "allen_live" in item.keywords]
    if not deselected:
        return

    selected = [item for item in items if item not in deselected]
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
