"""
Task 2 - ``NowcastingModels``: persistence baseline + ConvLSTM deep model.

Works on the rainfall-intensity sequences prepared in Task 1
(``log(1 + R)`` -> z-score normalised) and reports results in mm/h.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from . import metrics as verification
from .baseline import persistence_forecast
from .config import DEFAULT_ARCHITECTURE, PATHS, TRAINING_CONFIGS, TrainingConfig
from .evaluation_plots import plot_prediction_comparison
from .models import build_model
from .preprocessing import inverse_transform_predictions
from .training import (
    build_callbacks,
    log_data_diagnostics,
    log_training_summary,
    plot_training_history,
)
from .utils import setup_logger

PathLike = Union[str, Path]

DISPLAY_NAMES = {
    "unet": "U-NET CONVLSTM MODEL",
    "moderate": "MODERATE CONVLSTM MODEL",
    "light": "LIGHT CONVLSTM MODEL",
}


class NowcastingModels:
    """
    Nowcasting workflow with:

    * a persistence baseline, and
    * a ConvLSTM deep model (``architecture`` = ``"light"`` | ``"moderate"`` | ``"unet"``).

    The adopted configuration is the lightweight ConvLSTM (16-16-8 filters,
    L2 regularisation, dropout), designed for small datasets and stable training.
    """

    def __init__(
        self,
        architecture: str = DEFAULT_ARCHITECTURE,
        log_file: str = "task2_nowcasting.log",
        training_config: Optional[TrainingConfig] = None,
    ):
        self.architecture = architecture
        self.training_config = training_config or TRAINING_CONFIGS[architecture]
        self.logger = setup_logger("NowcastingModels", log_file)
        self.logger.info(f"NowcastingModels ({architecture}) logger initialized.")

        self.persistence_model = None
        self.model = None
        self.history = None
        self.scaler = None

    # ------------------------------------------------------------------
    # Persistence baseline
    # ------------------------------------------------------------------
    def create_persistence_baseline(self) -> None:
        """Configure the persistence forecast as a callable "model"."""
        self.logger.info("Creating persistence baseline model")
        self.persistence_model = persistence_forecast
        self.logger.info("Persistence baseline created successfully")

    # ------------------------------------------------------------------
    # Deep model
    # ------------------------------------------------------------------
    def build_model(self, input_shape, forecast_horizon: int = 3):
        """Build and compile the selected ConvLSTM architecture."""
        self.logger.info(f"Building {self.architecture.upper()} ConvLSTM model with input shape {input_shape}")
        self.model = build_model(self.architecture, tuple(input_shape), forecast_horizon)
        self.logger.info(f"Model built successfully. Total parameters: {self.model.count_params()}")
        return self.model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        save_path: PathLike = PATHS.best_model_file,
        history_plot_path: Optional[PathLike] = PATHS.training_figures_dir / "training_history.png",
    ) -> Dict[str, list]:
        """Train with early stopping, LR scheduling and best-model checkpointing."""
        cfg = self.training_config
        epochs = cfg.epochs if epochs is None else epochs
        batch_size = cfg.batch_size if batch_size is None else batch_size

        self.logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        log_data_diagnostics(self.logger, X_train, y_train)

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=build_callbacks(cfg, save_path),
            verbose=1,
        )

        history_dict = {k: [float(v) for v in vals] for k, vals in self.history.history.items()}
        log_training_summary(self.logger, history_dict, cfg.min_improvement_pct)

        if history_plot_path is not None:
            plot_training_history(history_dict, history_plot_path)
            self.logger.info(f"Training history plot saved as {history_plot_path}")
        return history_dict

    # ------------------------------------------------------------------
    # Inverse transform & metrics (thin wrappers kept for a familiar API)
    # ------------------------------------------------------------------
    @staticmethod
    def inverse_transform_predictions(data_scaled: np.ndarray, scaler) -> np.ndarray:
        """Normalised ``log(1 + R)`` -> rainfall intensity (mm/h)."""
        return inverse_transform_predictions(data_scaled, scaler)

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Continuous + categorical metrics in physical units."""
        return verification.calculate_metrics(y_true, y_pred)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def generate_predictions(self, X_test: np.ndarray, scaler) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with both models and return ``(persistence, convlstm)`` in mm/h."""
        self.logger.info("Generating predictions from both models...")
        forecast_horizon = self.model.output_shape[1]

        # Persistence (normalised space)
        pers_preds_scaled = np.array(
            [self.persistence_model(X_test[i], nsteps=forecast_horizon) for i in range(len(X_test))]
        )
        self.logger.info(
            f"Persistence predictions (scaled) range: {pers_preds_scaled.min():.4f} to {pers_preds_scaled.max():.4f}"
        )

        # ConvLSTM (normalised space)
        self.logger.info("Generating ConvLSTM predictions...")
        model_preds_scaled = self.model.predict(X_test, batch_size=self.training_config.predict_batch_size, verbose=1)
        self.logger.info(
            f"ConvLSTM predictions (scaled) range: {model_preds_scaled.min():.4f} to {model_preds_scaled.max():.4f}"
        )
        if self.architecture == "unet":
            self._log_saturation_diagnostics(model_preds_scaled)

        # Back to physical units (mm/h)
        return (
            self.inverse_transform_predictions(pers_preds_scaled, scaler),
            self.inverse_transform_predictions(model_preds_scaled, scaler),
        )

    def _log_saturation_diagnostics(self, preds_scaled: np.ndarray) -> None:
        """Extra diagnostics used while debugging the heavy U-Net variant."""
        near_zero = np.sum(preds_scaled < 0.01) / preds_scaled.size * 100.0
        near_one = np.sum(preds_scaled > 0.99) / preds_scaled.size * 100.0
        self.logger.info(f"U-Net: {near_zero:.1f}% predictions near 0, {near_one:.1f}% near 1.")
        if near_zero > 90.0:
            self.logger.warning("WARNING: U-Net predictions are mostly near 0. Model may be underpredicting.")
        if near_one > 10.0:
            self.logger.warning("WARNING: U-Net predictions have many values near 1 (saturation).")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_models(
        self,
        y_test_physical: np.ndarray,
        pers_preds_physical: np.ndarray,
        model_preds_physical: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Score both models (mm/h) and print a console summary."""
        self.logger.info("Calculating evaluation metrics...")
        results = {
            "persistence": self.calculate_metrics(y_test_physical, pers_preds_physical),
            "convlstm": self.calculate_metrics(y_test_physical, model_preds_physical),
        }
        print_task2_summary(results, DISPLAY_NAMES.get(self.architecture, "CONVLSTM MODEL"))
        return results

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def create_comparison_plot(
        self,
        y_test_physical: np.ndarray,
        pers_preds_physical: np.ndarray,
        model_preds_physical: np.ndarray,
        idx: int = 0,
        output_path: PathLike = PATHS.training_figures_dir / "prediction_comparison.png",
    ) -> Path:
        """True vs persistence vs ConvLSTM for one test sample across the horizon."""
        path = plot_prediction_comparison(y_test_physical, pers_preds_physical, model_preds_physical, idx, output_path)
        self.logger.info(f"Prediction comparison plot saved as {path}")
        return path


def print_task2_summary(results: Dict[str, Dict[str, float]], model_label: str) -> None:
    """Pretty-print the Task 2 metrics and the improvement over persistence."""
    line = "=" * 70
    print(line)
    print("MODEL EVALUATION RESULTS (Physical Units - mm/h)")
    print(line)

    print("\n--- PERSISTENCE BASELINE ---")
    for metric, value in results["persistence"].items():
        print(f"{metric.upper():15s}: {value:10.4f}")

    print(f"\n--- {model_label} ---")
    for metric, value in results["convlstm"].items():
        print(f"{metric.upper():15s}: {value:10.4f}")

    print("\n" + line)
    print("IMPROVEMENT OVER BASELINE")
    print(line)
    mse_improvement = verification.improvement_pct(results["persistence"]["mse"], results["convlstm"]["mse"])
    print(f"MSE Improvement: {mse_improvement:.2f}%")
    heavy_csi_diff = results["convlstm"].get("heavy_csi", 0.0) - results["persistence"].get("heavy_csi", 0.0)
    print(f"Heavy Rain CSI Improvement: {heavy_csi_diff:.4f}")
    print(line)
