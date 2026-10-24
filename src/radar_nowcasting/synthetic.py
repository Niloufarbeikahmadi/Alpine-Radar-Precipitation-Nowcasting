"""
Synthetic radar data for smoke tests and demos.

Generates a NetCDF file with the same structure as the study dataset
(``reflectivity(time, y, x)`` sampled every 5 minutes) containing a few
Gaussian convective cells that are advected across the domain while they
grow and decay. It is **not** used to produce any reported result - it only
lets the pipeline run end-to-end when the real radar file is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

PathLike = Union[str, Path]


def create_dummy_radar_data(
    output_path: PathLike,
    n_timesteps: int = 200,
    grid_size: int = 64,
    n_cells: int = 3,
    scaled: bool = True,
    seed: int = 0,
    start_time: str = "2025-06-01T10:00:00",
) -> Path:
    """Write a synthetic radar reflectivity cube to ``output_path`` and return the path.

    Parameters
    ----------
    scaled : bool
        If ``True`` reflectivity is min-max scaled to ``[0, 1]`` (like the study
        data); otherwise it is stored in dBZ.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:grid_size, 0:grid_size].astype(np.float32)

    # Cell properties: start position, velocity (px / step), peak dBZ, radius
    starts = rng.uniform(0.15 * grid_size, 0.5 * grid_size, size=(n_cells, 2))
    velocity = rng.uniform(0.2, 0.6, size=(n_cells, 2))
    peaks = rng.uniform(45.0, 58.0, size=n_cells)
    radii = rng.uniform(3.0, 7.0, size=n_cells)

    dbz = np.full((n_timesteps, grid_size, grid_size), 5.0, dtype=np.float32)
    t_axis = np.arange(n_timesteps)
    life = np.sin(np.pi * (t_axis + 1) / (n_timesteps + 1))  # growth then decay

    for t in range(n_timesteps):
        field = dbz[t]
        for k in range(n_cells):
            cy, cx = (starts[k] + velocity[k] * t) % grid_size
            dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
            cell = peaks[k] * life[t] * np.exp(-dist2 / (2 * radii[k] ** 2))
            field = np.maximum(field, cell)
        dbz[t] = field + rng.normal(0.0, 0.5, size=field.shape).astype(np.float32)

    dbz = np.clip(dbz, 0.0, 60.0)
    if scaled:
        reflectivity = (dbz - dbz.min()) / (dbz.max() - dbz.min())
        units, description = "1", "Min-max scaled radar reflectivity (synthetic)"
    else:
        reflectivity = dbz
        units, description = "dBZ", "Radar reflectivity (synthetic)"

    times = pd.date_range(start_time, periods=n_timesteps, freq="5min")
    ds = xr.Dataset(
        {
            "reflectivity": (
                ("time", "y", "x"),
                reflectivity.astype(np.float32),
                {"units": units, "description": description},
            )
        },
        coords={
            "time": times,
            "y": np.arange(grid_size, dtype=np.float32),
            "x": np.arange(grid_size, dtype=np.float32),
        },
        attrs={"title": "Synthetic radar reflectivity for pipeline testing"},
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    return output_path
