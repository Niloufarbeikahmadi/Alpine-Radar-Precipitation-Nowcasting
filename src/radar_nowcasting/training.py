"""Task 2 - training utilities: callbacks, diagnostics and learning curves."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from .config import TrainingConfig

PathLike = Union[str, Path]


def build_callbacks(cfg: TrainingConfig, checkpoint_path: PathLike) -> List:
    """Early stopping, LR-on-plateau scheduling and best-model checkpointing (on val loss)."""
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_lr,
            verbose=1,
        ),
        ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def log_data_diagnostics(logger: logging.Logger, X_train: np.ndarray, y_train: np.ndarray) -> None:
    """Log value ranges / moments and flag NaN or Inf values before training."""
    logger.info("Pre-training diagnostics")
    logger.info(f"X_train range {X_train.min():.4f}, {X_train.max():.4f}")
    logger.info(f"y_train range {y_train.min():.4f}, {y_train.max():.4f}")
    logger.info(f"X_train mean {X_train.mean():.4f}, std {X_train.std():.4f}")
    logger.info(f"y_train mean {y_train.mean():.4f}, std {y_train.std():.4f}")

    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        logger.error("WARNING: X_train contains NaN or Inf values!")
    if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
        logger.error("WARNING: y_train contains NaN or Inf values!")


def log_training_summary(logger: logging.Logger, history: Dict[str, list], min_improvement_pct: float) -> float:
    """Report how much the training loss improved; warn if below ``min_improvement_pct``."""
    final_train_loss = float(history["loss"][-1])
    initial_train_loss = float(history["loss"][0])
    improvement = (initial_train_loss - final_train_loss) / max(initial_train_loss, 1e-8) * 100.0

    logger.info(f"Initial training loss {initial_train_loss:.6f}")
    logger.info(f"Final training loss {final_train_loss:.6f}")
    logger.info(f"Training loss improvement {improvement:.2f}%")
    if improvement < min_improvement_pct:
        logger.warning(
            f"Training loss improved by less than {min_improvement_pct:.0f}%. "
            "Consider more epochs or checking preprocessing."
        )
    return improvement


def plot_training_history(history: Dict[str, list], output_path: PathLike) -> Path:
    """Save training / validation MSE loss and MAE curves (log scale)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["loss"], label="Training Loss", linewidth=2)
    ax1.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Model Training Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    if "mae" in history and "val_mae" in history:
        ax2.plot(history["mae"], label="Training MAE", linewidth=2)
        ax2.plot(history["val_mae"], label="Validation MAE", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MAE")
        ax2.set_title("Mean Absolute Error")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
