"""Shared helpers: logging, reproducibility and artefact I/O."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np

from .config import PATHS

PathLike = Union[str, Path]

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(
    name: str,
    log_file: PathLike | None = None,
    level: int = logging.INFO,
    fmt: str = LOG_FORMAT,
    reset_handlers: bool = False,
) -> logging.Logger:
    """Create (or reuse) a logger that writes to the console and, optionally, a file.

    Relative ``log_file`` names are placed in the project ``logs/`` folder.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if reset_handlers:
        logger.handlers = []

    if not logger.handlers:
        formatter = logging.Formatter(fmt)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file is not None:
            log_path = Path(log_file)
            if not log_path.is_absolute():
                log_path = PATHS.logs_dir / log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Fix NumPy and TensorFlow random seeds (as in the original study)."""
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:  # pragma: no cover - TF is not needed for Task 1
        pass


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_json(data: Any, path: PathLike, indent: int = 4) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=_to_builtin)
    return path


def load_json(path: PathLike) -> Any:
    with open(Path(path)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Task 1 artefacts  ->  data/processed/
# ---------------------------------------------------------------------------
def save_task1_results(
    X: np.ndarray,
    y: np.ndarray,
    scaler: Dict[str, Any],
    summary_stats: Dict[str, Any],
    output_dir: PathLike = PATHS.processed_dir,
) -> Path:
    """Persist Task 1 outputs (sequences, scaler, summary) for use in Task 2."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_sequences.npy", X)
    np.save(output_dir / "y_targets.npy", y)
    save_json(scaler, output_dir / "scaler.json")
    save_json(summary_stats, output_dir / "summary_stats.json")

    logging.getLogger("Task1Save").info("Task 1 results saved successfully to %s", output_dir)
    return output_dir


def load_task1_results(
    input_dir: PathLike = PATHS.processed_dir,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load the sequences ``X``, targets ``y`` and normalisation scaler from Task 1."""
    input_dir = Path(input_dir)
    X = np.load(input_dir / "X_sequences.npy")
    y = np.load(input_dir / "y_targets.npy")
    scaler = load_json(input_dir / "scaler.json")
    return X, y, scaler


# ---------------------------------------------------------------------------
# Task 2 artefacts  ->  outputs/predictions/
# ---------------------------------------------------------------------------
def save_test_predictions(
    path: PathLike,
    X_test: np.ndarray,
    y_test_physical: np.ndarray,
    persistence: np.ndarray,
    convlstm: np.ndarray,
) -> Path:
    """Store the held-out test set and both models' forecasts (mm/h) in one ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X_test=X_test,
        y_test_physical=y_test_physical,
        persistence=persistence,
        convlstm=convlstm,
    )
    return path


def load_test_predictions(path: PathLike) -> Dict[str, np.ndarray]:
    """Load the arrays written by :func:`save_test_predictions`."""
    with np.load(Path(path)) as data:
        return {k: data[k] for k in data.files}
