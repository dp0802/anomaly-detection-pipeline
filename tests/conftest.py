"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `src` and `models` importable from tests/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def features():
    return ["temperature", "pressure", "vibration", "power_consumption"]
