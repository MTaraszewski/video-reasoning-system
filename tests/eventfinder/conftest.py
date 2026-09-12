"""Shared fixtures. Locating the labelled footage is the interesting part.

The repo runs in a container where `./data` is mounted read-only at `/data`,
and on a laptop where it sits at `./data`. A test that hardcodes one of those
does not fail in the other -- it SKIPS, silently, and a green run then means
"nothing was checked". Resolving both is the difference between a suite that
ran and a suite that reported success.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# In the container's mount order, then the working tree's.
_DATA_ROOTS = [Path(os.getenv("EF_DATA_DIR", "/data")), Path("data")]


def find_clip(name: str) -> Path | None:
    for root in _DATA_ROOTS:
        p = root / "eval" / name
        if p.exists():
            return p
    return None


def require_clip(name: str) -> Path:
    p = find_clip(name)
    if p is None:
        pytest.skip(f"{name} not present under any of {[str(r) for r in _DATA_ROOTS]}; "
                    "build the evaluation set first")
    return p


@pytest.fixture(scope="session")
def data_root() -> Path:
    for root in _DATA_ROOTS:
        if (root / "eval").is_dir():
            return root
    pytest.skip("no evaluation data found")
