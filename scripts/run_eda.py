#!/usr/bin/env python
"""Task 1 - EDA & pre-processing of radar reflectivity.

Examples
--------
    python scripts/run_eda.py                              # uses data/raw/radar_data.nc
    python scripts/run_eda.py --data path/to/radar_data.nc
    python scripts/run_eda.py --synthetic                  # demo, written to demo_run/
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from radar_nowcasting.config import PATHS, PROJECT_ROOT, Paths
from radar_nowcasting.pipeline import run_task1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="Radar NetCDF file (default: data/raw/radar_data.nc)")
    parser.add_argument("--synthetic", action="store_true", help="Use a generated synthetic radar cube (demo)")
    parser.add_argument("--timesteps", type=int, default=200, help="Synthetic: number of frames")
    parser.add_argument("--grid-size", type=int, default=64, help="Synthetic: grid size (pixels)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = Paths.at(PROJECT_ROOT / "demo_run") if args.synthetic else PATHS
    run_task1(
        data_file=Path(args.data) if args.data else paths.raw_data_file,
        create_dummy_if_missing=args.synthetic,
        dummy_timesteps=args.timesteps,
        dummy_grid_size=args.grid_size,
        figures_dir=paths.eda_figures_dir,
        processed_dir=paths.processed_dir,
    )
