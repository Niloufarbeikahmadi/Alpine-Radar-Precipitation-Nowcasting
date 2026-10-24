"""Shared pytest fixtures."""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from radar_nowcasting.synthetic import create_dummy_radar_data  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def synthetic_nc(tmp_path_factory) -> Path:
    """A small scaled-reflectivity radar cube (50 frames, 64 x 64)."""
    path = tmp_path_factory.mktemp("radar") / "synthetic.nc"
    return create_dummy_radar_data(path, n_timesteps=50, grid_size=64, seed=1)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
