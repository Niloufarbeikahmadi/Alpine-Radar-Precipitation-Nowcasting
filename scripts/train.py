#!/usr/bin/env python
"""Task 2 - train the ConvLSTM nowcasting model and compare it with persistence.

Examples
--------
    python scripts/train.py                          # adopted light ConvLSTM, 70 epochs
    python scripts/train.py --architecture unet      # reproduce the rejected heavy U-Net
    python scripts/train.py --epochs 5               # quick check
"""

import argparse

import _bootstrap  # noqa: F401
from radar_nowcasting.config import DEFAULT_ARCHITECTURE, PATHS
from radar_nowcasting.pipeline import run_task2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--architecture",
        default=DEFAULT_ARCHITECTURE,
        choices=["light", "moderate", "unet"],
        help="ConvLSTM variant (default: light, the adopted model)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs (default 70)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (default 4)")
    parser.add_argument("--processed-dir", default=str(PATHS.processed_dir))
    parser.add_argument("--model-path", default=str(PATHS.best_model_file))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_task2(
        processed_dir=args.processed_dir,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=args.model_path,
    )
