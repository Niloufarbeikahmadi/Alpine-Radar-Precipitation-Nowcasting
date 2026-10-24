#!/usr/bin/env python
"""Task 3 - comprehensive verification of the persistence and ConvLSTM forecasts.

Runs from the saved test-set predictions, so it does not need TensorFlow or the
raw radar data. With the predictions shipped in ``outputs/predictions`` it
reproduces the reported metrics exactly.

    python scripts/evaluate.py
"""

import argparse

import _bootstrap  # noqa: F401
from radar_nowcasting.config import PATHS
from radar_nowcasting.pipeline import run_task3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", default=str(PATHS.test_predictions_file))
    parser.add_argument("--figures-dir", default=str(PATHS.evaluation_figures_dir))
    parser.add_argument("--metrics-dir", default=str(PATHS.metrics_dir))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_task3(
        predictions_file=args.predictions,
        figures_dir=args.figures_dir,
        metrics_dir=args.metrics_dir,
    )
