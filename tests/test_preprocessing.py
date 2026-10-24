"""Task 1: Z-R conversion, normalisation, sequences, QC and the EDA class."""

import numpy as np
import pytest
import xarray as xr

from radar_nowcasting.preprocessing import (
    StormCellNowcastingEDA,
    build_sequences,
    fit_log_zscore,
    inverse_transform_predictions,
)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------
def test_log_zscore_round_trip(rng):
    rain = rng.gamma(0.3, 5.0, size=(20, 16, 16)).astype(np.float32)
    norm, scaler = fit_log_zscore(rain)

    assert scaler["type"] == "zscore_log_rain"
    assert norm.mean() == pytest.approx(0.0, abs=1e-5)
    assert norm.std() == pytest.approx(1.0, abs=1e-4)
    np.testing.assert_allclose(inverse_transform_predictions(norm, scaler), rain, rtol=1e-4, atol=1e-4)


def test_inverse_transform_without_scaler_assumes_log_space():
    x = np.array([0.0, np.log1p(10.0)])
    np.testing.assert_allclose(inverse_transform_predictions(x, scaler=None), [0.0, 10.0])


def test_build_sequences_shapes_and_alignment():
    frames = np.arange(50, dtype=np.float32)[:, None, None] * np.ones((50, 4, 4), np.float32)
    X, y = build_sequences(frames, input_length=5, target_length=3)

    assert X.shape == (43, 5, 4, 4, 1)
    assert y.shape == (43, 3, 4, 4, 1)
    # sample i: inputs t = i..i+4, targets t = i+5..i+7
    assert X[7, :, 0, 0, 0].tolist() == [7, 8, 9, 10, 11]
    assert y[7, :, 0, 0, 0].tolist() == [12, 13, 14]


# ---------------------------------------------------------------------------
# EDA class
# ---------------------------------------------------------------------------
def _eda_from_dbz(tmp_path, dbz_values) -> StormCellNowcastingEDA:
    """Build an EDA object around a tiny dBZ cube."""
    data = np.asarray(dbz_values, dtype=np.float32)
    ds = xr.Dataset(
        {"reflectivity": (("time", "y", "x"), data)},
        coords={
            "time": np.arange(data.shape[0]) * np.timedelta64(5, "m") + np.datetime64("2025-06-01T10:00"),
            "y": np.arange(data.shape[1]),
            "x": np.arange(data.shape[2]),
        },
    )
    path = tmp_path / "tiny.nc"
    ds.to_netcdf(path)
    eda = StormCellNowcastingEDA(path, log_file=str(tmp_path / "eda.log"))
    assert eda.load_and_validate_netcdf()
    return eda


def test_zr_conversion_marshall_palmer(tmp_path):
    # 40 dBZ -> Z = 1e4 -> R = (1e4 / 200) ** (1 / 1.6) ≈ 11.53 mm/h ; 10 dBZ is clutter -> 0
    eda = _eda_from_dbz(tmp_path, np.full((3, 2, 2), 40.0) * np.array([1, 1, 0.25])[:, None, None])
    assert eda.reflectivity_metadata["is_scaled"] is False

    eda.convert_reflectivity_to_rainfall_intensity()
    rain = eda.data.rainfall_intensity.values

    assert rain[0, 0, 0] == pytest.approx((1e4 / 200.0) ** (1 / 1.6), rel=1e-5)
    assert np.all(rain[2] == 0.0)  # 10 dBZ < 15 dBZ clutter threshold
    assert eda.data.rainfall_intensity.attrs["units"] == "mm/h"


def test_missing_values_are_interpolated_in_time(tmp_path):
    cube = np.full((5, 2, 2), 30.0, dtype=np.float32)
    cube[:, 0, 0] = [20.0, np.nan, 40.0, np.nan, np.nan]
    cube[2, 1, 1] = 99.0  # implausible (> 75 dBZ) -> NaN -> interpolated
    eda = _eda_from_dbz(tmp_path, cube)

    eda.detect_and_handle_missing_values()
    refl = eda.data.reflectivity.values

    assert refl[1, 0, 0] == pytest.approx(30.0)  # mean of both neighbours
    assert refl[3, 0, 0] == pytest.approx(40.0)  # forward-filled from previous
    assert np.isnan(refl[4, 0, 0])  # last frame left untouched
    assert refl[2, 1, 1] == pytest.approx(30.0)


def test_full_task1_on_synthetic_data(synthetic_nc, tmp_path):
    eda = StormCellNowcastingEDA(synthetic_nc, log_file=str(tmp_path / "eda.log"))
    assert eda.load_and_validate_netcdf()
    assert eda.reflectivity_metadata["is_scaled"] is True

    eda.detect_and_handle_missing_values()
    eda.convert_reflectivity_to_rainfall_intensity()
    eda.create_visualizations(tmp_path / "figs")
    X, y, scaler = eda.prepare_sequences_for_training()
    stats = eda.generate_summary_statistics()

    assert X.shape == (43, 5, 64, 64, 1) and y.shape == (43, 3, 64, 64, 1)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    assert stats["data_shape"] == (50, 64, 64)
    assert set(stats["storm_cells_detected"]) == {"moderate_rain", "heavy_rain", "violent_rain", "extreme_rain"}
    for name in ["temporal_evolution", "spatial_patterns", "statistical_distributions", "storm_cell_tracking"]:
        assert (tmp_path / "figs" / f"{name}.png").stat().st_size > 0


def test_invalid_file_fails_validation(tmp_path):
    xr.Dataset({"precip": (("t", "y", "x"), np.zeros((2, 2, 2)))}).to_netcdf(tmp_path / "bad.nc")
    eda = StormCellNowcastingEDA(tmp_path / "bad.nc", log_file=str(tmp_path / "eda.log"))
    assert eda.load_and_validate_netcdf() is False
