"""
End-to-end orchestration of the three study tasks.

* :func:`run_task1` - EDA & pre-processing  (radar NetCDF -> training sequences)
* :func:`run_task2` - model development     (persistence vs ConvLSTM)
* :func:`run_task3` - comprehensive evaluation (metrics, figures, reports)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from sklearn.model_selection import train_test_split

from .config import (
    DEFAULT_ARCHITECTURE,
    PATHS,
    RANDOM_SEED,
    SPLIT_RANDOM_STATE,
    TEST_VAL_FRACTION,
    VAL_TEST_SPLIT,
)
from .utils import (
    load_task1_results,
    load_test_predictions,
    save_json,
    save_task1_results,
    save_test_predictions,
    set_global_seed,
)

PathLike = Union[str, Path]
BANNER = "=" * 70


# ===========================================================================
# Task 1
# ===========================================================================
def run_task1(
    data_file: PathLike = PATHS.raw_data_file,
    create_dummy_if_missing: bool = False,
    dummy_timesteps: int = 200,
    dummy_grid_size: int = 64,
    figures_dir: PathLike = PATHS.eda_figures_dir,
    processed_dir: PathLike = PATHS.processed_dir,
):
    """Run EDA & pre-processing and save the training sequences.

    By default an existing radar file is required. Synthetic data can be
    generated instead (``create_dummy_if_missing=True``) for smoke tests.
    """
    from .preprocessing import StormCellNowcastingEDA

    logger = logging.getLogger("Task1Main")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

    data_file = Path(data_file)

    # 1) Radar data (optionally synthetic)
    if not data_file.exists():
        if not create_dummy_if_missing:
            logger.error(f"Data file '{data_file}' not found and create_dummy_if_missing=False.")
            raise FileNotFoundError(f"Radar data file '{data_file}' not found.")

        from .synthetic import create_dummy_radar_data

        logger.warning(f"Data file '{data_file}' not found. Creating synthetic dataset...")
        create_dummy_radar_data(data_file, n_timesteps=dummy_timesteps, grid_size=dummy_grid_size)

    # 2) Initialise processor and 3) load / validate
    eda = StormCellNowcastingEDA(data_file)
    if not eda.load_and_validate_netcdf():
        eda.logger.error("Failed to load and validate NetCDF data. Exiting.")
        return None

    # 4) Missing values & outliers
    eda.detect_and_handle_missing_values()

    # 5) Reflectivity -> rainfall intensity (Z = 200 R^1.6)
    eda.convert_reflectivity_to_rainfall_intensity()

    # 6) Visualisations
    eda.create_visualizations(figures_dir)

    # 7) Sequences for model training (5 -> 3 frames)
    X, y, scaler = eda.prepare_sequences_for_training()

    # 8) Summary statistics and 9) persistence for Task 2
    summary_stats = eda.generate_summary_statistics()
    save_task1_results(X, y, scaler, summary_stats, processed_dir)

    eda.logger.info("EDA and Pre-processing pipeline completed successfully!")
    return eda, X, y, scaler, summary_stats


# ===========================================================================
# Task 2
# ===========================================================================
def run_task2(
    processed_dir: PathLike = PATHS.processed_dir,
    architecture: str = DEFAULT_ARCHITECTURE,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    model_path: PathLike = PATHS.best_model_file,
    training_figures_dir: PathLike = PATHS.training_figures_dir,
    metrics_dir: PathLike = PATHS.metrics_dir,
    predictions_file: PathLike = PATHS.test_predictions_file,
    seed: int = RANDOM_SEED,
):
    """Train / evaluate persistence and ConvLSTM models and save outputs for Task 3."""
    set_global_seed(seed)
    from .nowcaster import NowcastingModels  # imports TensorFlow

    training_figures_dir = Path(training_figures_dir)

    print(f"\n{BANNER}\nTASK 2: PRECIPITATION NOWCASTING MODEL DEVELOPMENT\n{BANNER}\n")

    # Load Task 1 results
    print("Loading preprocessed data from Task 1...")
    try:
        X, y, scaler = load_task1_results(processed_dir)
    except FileNotFoundError:
        print("ERROR: Task 1 results not found. Please run Task 1 first.")
        return None
    print(f"✓ Data loaded successfully\n  Input sequences: {X.shape}\n  Target sequences: {y.shape}")

    # Split 70 / 15 / 15
    print("\nSplitting data into train/validation/test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_VAL_FRACTION, random_state=SPLIT_RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_TEST_SPLIT, random_state=SPLIT_RANDOM_STATE
    )
    print(f"  Training:   {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test:       {X_test.shape[0]} samples")

    # Models
    model_dev = NowcastingModels(architecture=architecture)
    model_dev.scaler = scaler

    print("\nCreating persistence baseline model...")
    model_dev.create_persistence_baseline()

    print(f"\nBuilding {architecture} ConvLSTM model...")
    model = model_dev.build_model(X_train.shape[1:], forecast_horizon=y_train.shape[1])
    print("\nModel Architecture:")
    model.summary()

    # Train
    print(f"\n{BANNER}\nTRAINING CONVLSTM MODEL\n{BANNER}")
    history = model_dev.train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        save_path=model_path,
        history_plot_path=training_figures_dir / "training_history.png",
    )

    # Predict & evaluate in physical units (mm/h)
    print("\nConverting test data to physical units...")
    y_test_physical = model_dev.inverse_transform_predictions(y_test, scaler)

    print("\nGenerating predictions from both models...")
    pers_preds, conv_preds = model_dev.generate_predictions(X_test, scaler)

    print("\nEvaluating models...")
    results = model_dev.evaluate_models(y_test_physical, pers_preds, conv_preds)

    print("\nCreating prediction comparison visualization...")
    model_dev.create_comparison_plot(
        y_test_physical,
        pers_preds,
        conv_preds,
        idx=0,
        output_path=training_figures_dir / "prediction_comparison.png",
    )

    # Persist outputs for Task 3
    print("\nSaving results for Task 3...")
    save_test_predictions(predictions_file, X_test, y_test_physical, pers_preds, conv_preds)
    save_json({"architecture": architecture, **results}, Path(metrics_dir) / "task2_metrics.json")
    save_json(history, Path(metrics_dir) / "training_history.json")

    print(f"\n{BANNER}\nTASK 2 COMPLETED SUCCESSFULLY!\n{BANNER}")
    print(f"  ✓ {model_path}")
    print(f"  ✓ {training_figures_dir / 'training_history.png'}")
    print(f"  ✓ {training_figures_dir / 'prediction_comparison.png'}")
    print(f"  ✓ {predictions_file}")
    print(f"{BANNER}\n")
    return model_dev, results, history


# ===========================================================================
# Task 3
# ===========================================================================
def run_task3(
    predictions_file: PathLike = PATHS.test_predictions_file,
    figures_dir: PathLike = PATHS.evaluation_figures_dir,
    metrics_dir: PathLike = PATHS.metrics_dir,
):
    """Comprehensive evaluation of the Task 2 forecasts."""
    from .evaluation import StormNowcastingEvaluation

    print(f"\n{BANNER}\nTASK 3: COMPREHENSIVE EVALUATION\n{BANNER}")
    print("\nLoading Task 2 predictions...")
    try:
        preds = load_test_predictions(predictions_file)
    except FileNotFoundError:
        print(f"ERROR: {predictions_file} not found. Please run Task 2 first.")
        return None, None

    y_true, y_pers, y_conv = preds["y_test_physical"], preds["persistence"], preds["convlstm"]
    print("✓ Task 2 predictions loaded successfully")
    print(f"  Ground truth shape: {y_true.shape}")
    print(f"  Persistence predictions shape: {y_pers.shape}")
    print(f"  ConvLSTM predictions shape: {y_conv.shape}")

    evaluator = StormNowcastingEvaluation(y_true, y_pers, y_conv)
    metrics = evaluator.calculate_comprehensive_metrics()

    p, c = metrics["persistence"], metrics["convlstm"]
    print(f"\n{'-' * 60}\nEVALUATION SUMMARY\n{'-' * 60}")
    print(
        f"\nRMSE  Persistence: {p['rmse']:.4f} mm/h | ConvLSTM: {c['rmse']:.4f} mm/h "
        f"| Improvement: {(p['rmse'] - c['rmse']) / p['rmse'] * 100:.2f}%"
    )
    print(f"CSI (>7.5 mm/h)  Persistence: {p['heavy_csi']:.4f} | ConvLSTM: {c['heavy_csi']:.4f}")
    print(f"Bias  Persistence: {p['bias']:+.4f} mm/h | ConvLSTM: {c['bias']:+.4f} mm/h")

    print("\nCreating comprehensive visualizations...")
    evaluator.create_comparative_visualizations(figures_dir)
    evaluator.save_reports(metrics_dir)

    print(f"\n{BANNER}\nTASK 3 COMPLETED SUCCESSFULLY!\n{BANNER}")
    print(f"  Figures: {figures_dir}\n  Reports: {metrics_dir}\n{BANNER}\n")
    return evaluator, metrics
