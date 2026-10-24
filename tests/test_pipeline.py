"""End-to-end checks: reported results are reproducible, and the pipeline runs."""

import json

import numpy as np
import pytest

from radar_nowcasting.pipeline import run_task1, run_task3
from radar_nowcasting.utils import load_task1_results


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def test_task3_reproduces_reported_metrics(project_root, tmp_path):
    """Re-evaluating the shipped predictions gives the published numbers."""
    run_task3(
        project_root / "outputs/predictions/test_predictions.npz",
        figures_dir=tmp_path / "figs",
        metrics_dir=tmp_path / "metrics",
    )
    reported = _flatten(json.loads((project_root / "outputs/metrics/comprehensive_metrics.json").read_text()))
    recomputed = _flatten(json.loads((tmp_path / "metrics/comprehensive_metrics.json").read_text()))

    assert reported.keys() == recomputed.keys()
    for key, value in reported.items():
        # float32 reductions may differ in the last digits across NumPy versions
        assert recomputed[key] == pytest.approx(value, rel=1e-5, abs=1e-6), key

    expected_report = (project_root / "outputs/metrics/evaluation_report.txt").read_text()
    assert (tmp_path / "metrics/evaluation_report.txt").read_text() == expected_report
    for fig in ["metrics_dashboard", "spatial_error_maps", "temporal_analysis", "statistical_distributions"]:
        assert (tmp_path / "figs" / f"{fig}.png").exists()


def test_shipped_processed_data_is_consistent(project_root):
    X, y, scaler = load_task1_results(project_root / "data/processed")
    assert X.shape == (43, 5, 64, 64, 1)
    assert y.shape == (43, 3, 64, 64, 1)
    assert scaler["type"] == "zscore_log_rain"
    # Each target window starts right after its input window (sliding by one frame)
    np.testing.assert_array_equal(X[1, -1], y[0, 0])


def test_task1_to_task2_to_task3_smoke(tmp_path):
    """Synthetic radar -> sequences -> 1-epoch ConvLSTM -> evaluation."""
    pytest.importorskip("tensorflow")
    from radar_nowcasting.pipeline import run_task2

    out = run_task1(
        data_file=tmp_path / "raw/demo.nc",
        create_dummy_if_missing=True,
        dummy_timesteps=30,
        figures_dir=tmp_path / "eda",
        processed_dir=tmp_path / "processed",
    )
    assert out is not None

    model_dev, results, history = run_task2(
        processed_dir=tmp_path / "processed",
        epochs=1,
        model_path=tmp_path / "models/model.h5",
        training_figures_dir=tmp_path / "training",
        metrics_dir=tmp_path / "metrics",
        predictions_file=tmp_path / "predictions/test_predictions.npz",
    )
    assert set(results) == {"persistence", "convlstm"}
    assert len(history["loss"]) == 1
    assert (tmp_path / "models/model.h5").exists()

    evaluator, metrics = run_task3(
        tmp_path / "predictions/test_predictions.npz",
        figures_dir=tmp_path / "evaluation",
        metrics_dir=tmp_path / "metrics",
    )
    assert metrics["convlstm"]["rmse"] >= 0
