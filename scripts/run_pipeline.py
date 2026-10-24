#!/usr/bin/env python
"""Run the full study: Task 1 (EDA) -> Task 2 (modelling) -> Task 3 (evaluation).

Examples
--------
    python scripts/run_pipeline.py                         # uses data/raw/radar_data.nc
    python scripts/run_pipeline.py --synthetic --epochs 3  # demo, written to demo_run/

Synthetic (demo) runs are written under ``demo_run/`` so they never overwrite
the reported results in ``data/processed`` and ``outputs/``.
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from radar_nowcasting.config import DEFAULT_ARCHITECTURE, PATHS, PROJECT_ROOT, Paths
from radar_nowcasting.pipeline import run_task1, run_task2, run_task3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="Radar NetCDF file (default: data/raw/radar_data.nc)")
    parser.add_argument("--synthetic", action="store_true", help="Use a generated synthetic radar cube (demo)")
    parser.add_argument("--architecture", default=DEFAULT_ARCHITECTURE, choices=["light", "moderate", "unet"])
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs (default 70)")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Write all artefacts under this folder (default: project root, or demo_run/ with --synthetic)",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Paths:
    if args.output_root:
        return Paths.at(args.output_root)
    if args.synthetic:
        return Paths.at(PROJECT_ROOT / "demo_run")
    return PATHS


if __name__ == "__main__":
    args = parse_args()
    paths = resolve_paths(args)
    data_file = Path(args.data) if args.data else paths.raw_data_file

    task1 = run_task1(
        data_file=data_file,
        create_dummy_if_missing=args.synthetic,
        figures_dir=paths.eda_figures_dir,
        processed_dir=paths.processed_dir,
    )
    if task1 is None:
        raise SystemExit("Task 1 failed - see logs/storm_nowcasting_eda.log")

    task2 = run_task2(
        processed_dir=paths.processed_dir,
        architecture=args.architecture,
        epochs=args.epochs,
        model_path=paths.best_model_file,
        training_figures_dir=paths.training_figures_dir,
        metrics_dir=paths.metrics_dir,
        predictions_file=paths.test_predictions_file,
    )
    if task2 is None:
        raise SystemExit("Task 2 failed")

    run_task3(
        predictions_file=paths.test_predictions_file,
        figures_dir=paths.evaluation_figures_dir,
        metrics_dir=paths.metrics_dir,
    )
    print(f"All artefacts written under: {paths.root}")
