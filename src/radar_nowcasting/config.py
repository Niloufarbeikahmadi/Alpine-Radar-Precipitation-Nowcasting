"""
Central configuration for the radar precipitation-nowcasting pipeline.

Every numeric value below is taken verbatim from the original research
script (``code.py``). Gathering them here makes the experimental set-up
easy to audit without changing any of the underlying science.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Filesystem layout used by the pipeline (relative to the project root)."""

    root: Path = PROJECT_ROOT

    # Data
    raw_data_file: Path = PROJECT_ROOT / "data" / "raw" / "radar_data.nc"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    # Outputs
    eda_figures_dir: Path = PROJECT_ROOT / "outputs" / "figures" / "eda"
    training_figures_dir: Path = PROJECT_ROOT / "outputs" / "figures" / "training"
    evaluation_figures_dir: Path = PROJECT_ROOT / "outputs" / "figures" / "evaluation"
    metrics_dir: Path = PROJECT_ROOT / "outputs" / "metrics"
    models_dir: Path = PROJECT_ROOT / "outputs" / "models"
    predictions_dir: Path = PROJECT_ROOT / "outputs" / "predictions"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # Individual artefacts
    @property
    def best_model_file(self) -> Path:
        return self.models_dir / "best_nowcasting_model.h5"

    @property
    def test_predictions_file(self) -> Path:
        return self.predictions_dir / "test_predictions.npz"

    @classmethod
    def at(cls, root: Path | str) -> "Paths":
        """Same folder layout under another root (e.g. a demo run that must not
        overwrite the reported results)."""
        root = Path(root).resolve()
        return cls(
            root=root,
            raw_data_file=root / "data" / "raw" / "radar_data.nc",
            processed_dir=root / "data" / "processed",
            eda_figures_dir=root / "outputs" / "figures" / "eda",
            training_figures_dir=root / "outputs" / "figures" / "training",
            evaluation_figures_dir=root / "outputs" / "figures" / "evaluation",
            metrics_dir=root / "outputs" / "metrics",
            models_dir=root / "outputs" / "models",
            predictions_dir=root / "outputs" / "predictions",
            logs_dir=PROJECT_ROOT / "logs",
        )


# ---------------------------------------------------------------------------
# Task 1 - EDA & pre-processing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessingConfig:
    # Physical dBZ range used to "un-scale" reflectivity delivered in [0, 1]
    target_dbz_min: float = -10.0
    target_dbz_max: float = 60.0

    # Heuristic upper bound for detecting pre-scaled reflectivity
    scaled_max_threshold: float = 1.5

    # Expected radar scan interval (minutes)
    expected_interval_min: int = 5

    # Meteorologically plausible reflectivity bounds (outlier detection)
    valid_min_dbz: float = -10.0
    valid_max_dbz: float = 75.0

    # Marshall-Palmer type Z-R relationship: Z = a * R^b
    zr_a: float = 200.0
    zr_b: float = 1.6

    # Reflectivity below this value (dBZ) is treated as clutter / no rain
    clutter_threshold_dbz: float = 15.0

    # Sliding-window framing: 5 input frames -> 3 target frames
    input_sequence_length: int = 5
    target_sequence_length: int = 3


# Rain-rate classes (mm/h) used in the EDA logs
EDA_LOG_THRESHOLDS: Dict[str, float] = {
    "Moderate rain (≥2.5 mm/h)": 2.5,
    "Heavy rain (≥7.5 mm/h)": 7.5,
    "Violent rain (≥50 mm/h)": 50.0,
}

# Rain-rate classes (mm/h) used for the storm-cell summary statistics
STORM_CELL_THRESHOLDS: Dict[str, float] = {
    "moderate_rain": 2.5,
    "heavy_rain": 7.5,
    "violent_rain": 25.0,
    "extreme_rain": 50.0,
}

# Visualisation thresholds (mm/h)
HEAVY_RAIN_PLOT_THRESHOLD: float = 7.5
STORM_CORE_THRESHOLD: float = 10.0
MIN_STORM_CELL_PIXELS: int = 10
SIGNIFICANT_RAIN_THRESHOLD: float = 0.1


# ---------------------------------------------------------------------------
# Task 2 - Model development
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 5  # np.random.seed / tf.random.set_seed
SPLIT_RANDOM_STATE: int = 42  # sklearn train_test_split
TEST_VAL_FRACTION: float = 0.3  # 70 % train, 30 % held out ...
VAL_TEST_SPLIT: float = 0.5  # ... split equally into validation / test

# Architecture adopted in the final study (best, most stable results)
DEFAULT_ARCHITECTURE: str = "light"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 70
    batch_size: int = 4
    learning_rate: float = 1e-3
    clipnorm: float = 1.0
    early_stopping_patience: int = 10
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 4
    min_lr: float = 1e-5
    # Log a warning if training loss improves by less than this percentage
    min_improvement_pct: float = 30.0
    predict_batch_size: int = 4


# The heavy U-Net variant was originally trained with slightly different
# callback settings - preserved here for full reproducibility.
TRAINING_CONFIGS: Dict[str, TrainingConfig] = {
    "unet": TrainingConfig(
        early_stopping_patience=5,
        reduce_lr_patience=5,
        min_lr=1e-6,
        min_improvement_pct=50.0,
    ),
    "moderate": TrainingConfig(),
    "light": TrainingConfig(),
}


# ---------------------------------------------------------------------------
# Verification thresholds (mm/h)
# ---------------------------------------------------------------------------
# Task 2 quick evaluation
TASK2_CSI_THRESHOLDS: Dict[str, float] = {
    "moderate": 2.5,
    "heavy": 7.5,
    "violent": 25.0,
}

# Task 3 comprehensive evaluation
TASK3_THRESHOLDS: Dict[str, float] = {
    "light": 0.1,
    "moderate": 2.5,
    "heavy": 7.5,
    "violent": 25.0,
    "extreme": 50.0,
}

EPS: float = 1e-9


# Convenience singletons
PATHS = Paths()
PREPROCESSING = PreprocessingConfig()
