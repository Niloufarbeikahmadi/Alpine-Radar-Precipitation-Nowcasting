"""
Task 1 - Exploratory data analysis & pre-processing of radar reflectivity.

Pipeline
--------
1. Load the NetCDF radar cube and validate its structure.
2. Detect implausible values / NaNs and fill them by temporal interpolation.
3. Map (possibly [0, 1]-scaled) reflectivity back to dBZ and convert it to
   rainfall intensity with the Z-R power law ``Z = a * R^b``.
4. Build sliding-window sequences (5 input -> 3 target frames) on
   ``log(1 + R)`` z-score normalised rainfall.
5. Compute summary statistics for the report.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import xarray as xr

from . import eda_plots
from .config import (
    EDA_LOG_THRESHOLDS,
    PATHS,
    PREPROCESSING,
    STORM_CELL_THRESHOLDS,
    PreprocessingConfig,
)
from .utils import setup_logger

PathLike = Union[str, Path]

SCALER_TYPE = "zscore_log_rain"


# ===========================================================================
# Normalisation helpers (shared by Task 1 and Task 2)
# ===========================================================================
def fit_log_zscore(rainfall: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply ``log(1 + R)`` followed by z-score normalisation.

    Returns the normalised array and the scaler dictionary needed to invert it.
    """
    rainfall_log = np.log1p(rainfall)
    mean = float(np.mean(rainfall_log))
    std = float(np.std(rainfall_log) + 1e-6)
    rainfall_norm = (rainfall_log - mean) / std

    scaler = {"type": SCALER_TYPE, "mean": mean, "std": std}
    return rainfall_norm, scaler


def inverse_transform_predictions(data_scaled: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    """Transform normalised values back to physical rainfall (mm/h).

    For scaler type ``'zscore_log_rain'``:
      1. Undo z-score: ``x_log = x_scaled * std + mean``
      2. Undo log1p:   ``x = expm1(x_log)``
    """
    original_shape = data_scaled.shape
    data_flat = data_scaled.reshape(-1)

    if isinstance(scaler, dict) and scaler.get("type") == SCALER_TYPE:
        data_log = data_flat * scaler["std"] + scaler["mean"]
    else:
        # Fallback: assume values are already in log(1 + R) space
        data_log = data_flat

    data_physical = np.expm1(data_log)
    return data_physical.reshape(original_shape)


def build_sequences(data: np.ndarray, input_length: int, target_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a window over ``(T, H, W)`` frames.

    Returns
    -------
    X : (N, input_length, H, W, 1)
    y : (N, target_length, H, W, 1)
    """
    n_samples = data.shape[0]
    seqs, targets = [], []
    for i in range(n_samples - input_length - target_length + 1):
        seqs.append(data[i : i + input_length])
        targets.append(data[i + input_length : i + input_length + target_length])

    X = np.array(seqs)[..., np.newaxis]
    y = np.array(targets)[..., np.newaxis]
    return X, y


# ===========================================================================
# EDA / pre-processing class
# ===========================================================================
class StormCellNowcastingEDA:
    """
    EDA and pre-processing for precipitation nowcasting from radar data.

    Handles both physical dBZ reflectivity and data pre-scaled to [0, 1].
    """

    def __init__(
        self,
        data_path: PathLike,
        log_file: str = "storm_nowcasting_eda.log",
        config: PreprocessingConfig = PREPROCESSING,
    ):
        self.data_path = str(data_path)
        self.config = config
        self.data: xr.Dataset = None
        self.logger = self._setup_logging(log_file)

        # Metadata describing how reflectivity is interpreted / transformed
        self.reflectivity_metadata: Dict[str, Any] = {
            "is_scaled": None,  # True if input reflectivity looked scaled
            "scale_min": None,  # original min
            "scale_max": None,  # original max
            "target_dbz_min": config.target_dbz_min,  # physical dBZ range used
            "target_dbz_max": config.target_dbz_max,  # when un-scaling
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    @staticmethod
    def _setup_logging(log_file: str) -> logging.Logger:
        logger = setup_logger("StormNowcastingEDA", log_file)
        logger.info("Storm Cell Nowcasting EDA Logger Initialized")
        return logger

    # ------------------------------------------------------------------
    # Loading & validation
    # ------------------------------------------------------------------
    def load_and_validate_netcdf(self) -> bool:
        """Load NetCDF data and validate its structure. Returns ``True`` if valid."""
        try:
            self.logger.info(f"Loading NetCDF file: {self.data_path}")
            self.data = xr.open_dataset(self.data_path)

            validation_passed = True
            validation_notes: List[str] = []

            # Primary variable
            if "reflectivity" not in self.data.variables:
                validation_passed = False
                validation_notes.append("Missing required variable 'reflectivity'")
            else:
                expected_dims = ["time", "y", "x"]
                actual_dims = list(self.data.reflectivity.dims)
                if set(expected_dims) != set(actual_dims):
                    validation_passed = False
                    validation_notes.append(f"Dimension mismatch. Expected {expected_dims}, got {actual_dims}")

                reflectivity_data = self.data.reflectivity.values
                rmin = float(np.nanmin(reflectivity_data))
                rmax = float(np.nanmax(reflectivity_data))
                self.logger.info(f"Reflectivity data shape: {reflectivity_data.shape}")
                self.logger.info(f"Reflectivity range: {rmin:.2f} to {rmax:.2f} (native units)")

                # Infer whether data looks scaled
                self._infer_reflectivity_scaling(rmin, rmax)

            # Required coordinates
            for coord in ["time", "x", "y"]:
                if coord not in self.data.coords:
                    validation_passed = False
                    validation_notes.append(f"Missing coordinate: {coord}")

            # Time spacing (5 min expected, but only warn if different)
            if "time" in self.data.coords and len(self.data.time) > 1:
                time_diff = np.diff(self.data.time.values)
                expected_interval = np.timedelta64(self.config.expected_interval_min, "m")
                if not np.allclose(
                    time_diff.astype("timedelta64[m]").astype(float),
                    expected_interval.astype("timedelta64[m]").astype(float),
                ):
                    validation_notes.append("Time intervals are not consistently 5 minutes")
                    self.logger.warning("Time intervals are not exactly 5 minutes; continuing anyway.")

            if validation_passed:
                self.logger.info("✅ NetCDF file structure validation PASSED - File follows suitable structure")
            else:
                self.logger.warning("❌ NetCDF file structure validation FAILED")
                for note in validation_notes:
                    self.logger.warning(f"   - {note}")

            return validation_passed

        except Exception as e:  # noqa: BLE001 - report any loading problem
            self.logger.error(f"Error loading NetCDF file: {str(e)}")
            return False

    def _infer_reflectivity_scaling(self, rmin: float, rmax: float) -> None:
        """Decide whether reflectivity is physical dBZ or scaled to [0, 1].

        Heuristic: values in a narrow ``[0, 1.x]`` range that are never
        negative are treated as scaled; anything else is treated as dBZ.
        """
        meta = self.reflectivity_metadata
        meta["scale_min"] = rmin
        meta["scale_max"] = rmax

        if rmin >= 0.0 and rmax <= self.config.scaled_max_threshold:
            meta["is_scaled"] = True
            self.logger.info(
                "Reflectivity appears to be pre-scaled (≈[0, 1]). "
                f"Will map back to dBZ using [{meta['target_dbz_min']}, "
                f"{meta['target_dbz_max']}] as physical range."
            )
        else:
            meta["is_scaled"] = False
            self.logger.info("Reflectivity values look like physical dBZ; no inverse scaling needed.")

    # ------------------------------------------------------------------
    # Missing values / outliers
    # ------------------------------------------------------------------
    def detect_and_handle_missing_values(self) -> None:
        """Detect implausible values / NaNs and fill them by temporal interpolation."""
        self.logger.info("Starting missing value and outlier detection")
        reflectivity = self.data.reflectivity.values

        original_stats = {
            "mean": np.nanmean(reflectivity),
            "std": np.nanstd(reflectivity),
            "min": np.nanmin(reflectivity),
            "max": np.nanmax(reflectivity),
            "nan_count": np.sum(np.isnan(reflectivity)),
            "total_pixels": reflectivity.size,
        }
        self.logger.info("Original data statistics:")
        for stat, value in original_stats.items():
            self.logger.info(f"  {stat}: {value:.4f}")

        # Outliers are assessed in "dBZ-like" units. With scaled data the
        # range is [0, 1], so these thresholds will normally not flag anything.
        outlier_mask = (reflectivity < self.config.valid_min_dbz) | (reflectivity > self.config.valid_max_dbz)
        outlier_count = int(np.sum(outlier_mask))
        if outlier_count > 0:
            self.logger.warning(f"Found {outlier_count} meteorologically implausible values; setting to NaN.")
            reflectivity[outlier_mask] = np.nan

        nan_mask = np.isnan(reflectivity)
        total_missing = int(np.sum(nan_mask))

        if total_missing > 0:
            missing_percentage = (total_missing / reflectivity.size) * 100
            self.logger.info(f"Total missing values (NaNs + outliers): {total_missing} ({missing_percentage:.4f}%)")
            _interpolate_missing_in_time(reflectivity)
            self.data["reflectivity"].values = reflectivity
            self.logger.info("Missing values handled using temporal interpolation")
        else:
            self.logger.info("No missing values detected")

    # ------------------------------------------------------------------
    # Reflectivity -> rainfall intensity
    # ------------------------------------------------------------------
    def _get_reflectivity_dbz(self) -> np.ndarray:
        """Return reflectivity in physical dBZ, applying inverse scaling if needed."""
        refl_native = self.data.reflectivity.values.astype(np.float32)
        meta = self.reflectivity_metadata

        if meta["is_scaled"]:
            src_min, src_max = meta["scale_min"], meta["scale_max"]
            tgt_min, tgt_max = meta["target_dbz_min"], meta["target_dbz_max"]

            if src_max <= src_min + 1e-6:
                self.logger.warning("Scaled reflectivity has near-constant range; falling back to zeros in dBZ.")
                refl_dbz = np.zeros_like(refl_native)
            else:
                # Linear mapping [src_min, src_max] -> [tgt_min, tgt_max]
                refl_norm = (refl_native - src_min) / (src_max - src_min)
                refl_dbz = tgt_min + refl_norm * (tgt_max - tgt_min)

            self.logger.info(
                f"Converted scaled reflectivity [{src_min:.3f}, {src_max:.3f}] "
                f"to approximate dBZ range [{tgt_min:.1f}, {tgt_max:.1f}]."
            )
        else:
            refl_dbz = refl_native  # already dBZ

        return refl_dbz

    def convert_reflectivity_to_rainfall_intensity(self, a: float | None = None, b: float | None = None) -> None:
        """Convert reflectivity to rainfall intensity (mm/h) using ``Z = a * R^b``.

        Scaled ``[0, 1]`` reflectivity is first mapped back to a dBZ range.
        Defaults: ``a = 200``, ``b = 1.6`` (Marshall-Palmer).
        """
        a = self.config.zr_a if a is None else a
        b = self.config.zr_b if b is None else b

        self.logger.info("Converting reflectivity to rainfall intensity using Z-R relationship")

        # 1) reflectivity in dBZ
        reflectivity_dbz = self._get_reflectivity_dbz()

        # 2) clutter suppression in dBZ space
        mask_no_rain = reflectivity_dbz < self.config.clutter_threshold_dbz

        # 3) dBZ -> Z (mm^6 m^-3)
        Z = 10.0 ** (reflectivity_dbz / 10.0)

        # 4) Z = a * R^b  =>  R = (Z / a)^(1/b)   [mm/h]
        R = (Z / a) ** (1.0 / b)
        R[mask_no_rain] = 0.0

        self.data["rainfall_intensity"] = xr.DataArray(
            R,
            dims=self.data.reflectivity.dims,
            coords=self.data.reflectivity.coords,
            attrs={
                "units": "mm/h",
                "long_name": "Rainfall intensity",
                "ZR_relationship": f"Z = {a} * R^{b}",
                "description": (
                    "Converted from reflectivity using power-law Z-R "
                    "relationship; reflectivity auto-scaled to dBZ if needed."
                ),
            },
        )

        self.logger.info(f"Rainfall intensity range: {np.nanmin(R):.4f} to {np.nanmax(R):.4f} mm/h")
        for name, thr in EDA_LOG_THRESHOLDS.items():
            pixels = int(np.sum(R >= thr))
            self.logger.info(f"  {name}: {pixels} pixels")

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    def create_visualizations(self, output_dir: PathLike = PATHS.eda_figures_dir) -> None:
        """Create the four EDA figures (see :mod:`radar_nowcasting.eda_plots`)."""
        self.logger.info("Creating comprehensive visualizations")
        eda_plots.create_all(self.data, output_dir, logger=self.logger)

    # ------------------------------------------------------------------
    # Sequences & summary
    # ------------------------------------------------------------------
    def prepare_sequences_for_training(
        self,
        input_sequence_length: int | None = None,
        target_sequence_length: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Normalise rainfall and frame it as (input -> target) sequences."""
        input_len = input_sequence_length or self.config.input_sequence_length
        target_len = target_sequence_length or self.config.target_sequence_length

        rainfall = self.data.rainfall_intensity.values
        rainfall_norm, scaler = fit_log_zscore(rainfall)
        X, y = build_sequences(rainfall_norm, input_len, target_len)
        return X, y, scaler

    def _detect_storm_cells(self) -> Dict[str, Any]:
        rainfall = self.data.rainfall_intensity.values
        out: Dict[str, Any] = {}
        for name, thr in STORM_CELL_THRESHOLDS.items():
            cnt = int(np.sum(rainfall > thr))
            perc = float(cnt / rainfall.size * 100.0)
            out[name] = {"threshold_mm_h": thr, "pixel_count": cnt, "percentage": perc}
        return out

    def generate_summary_statistics(self) -> dict:
        self.logger.info("Generating summary statistics")
        refl = self.data.reflectivity.values
        rain = self.data.rainfall_intensity.values
        stats = {
            "data_shape": self.data.reflectivity.shape,
            "time_range": {
                "start": str(self.data.time.values[0]),
                "end": str(self.data.time.values[-1]),
                "n_timesteps": len(self.data.time),
            },
            "spatial_extent": {
                "x_range": (float(self.data.x.min()), float(self.data.x.max())),
                "y_range": (float(self.data.y.min()), float(self.data.y.max())),
            },
            "reflectivity_stats": {
                "min": float(np.nanmin(refl)),
                "max": float(np.nanmax(refl)),
                "mean": float(np.nanmean(refl)),
                "std": float(np.nanstd(refl)),
                "is_scaled": self.reflectivity_metadata["is_scaled"],
                "scale_min": self.reflectivity_metadata["scale_min"],
                "scale_max": self.reflectivity_metadata["scale_max"],
            },
            "rainfall_stats": {
                "min": float(np.nanmin(rain)),
                "max": float(np.nanmax(rain)),
                "mean": float(np.nanmean(rain)),
                "std": float(np.nanstd(rain)),
            },
            "storm_cells_detected": self._detect_storm_cells(),
        }

        self.logger.info("=== SUMMARY STATISTICS ===")
        self.logger.info(f"Data shape: {stats['data_shape']}")
        self.logger.info(f"Time range: {stats['time_range']['start']} to {stats['time_range']['end']}")
        rs = stats["reflectivity_stats"]
        self.logger.info(f"Reflectivity (native) - Min: {rs['min']:.2f}, Max: {rs['max']:.2f}, Mean: {rs['mean']:.2f}")
        ps = stats["rainfall_stats"]
        self.logger.info(f"Rainfall - Min: {ps['min']:.4f}, Max: {ps['max']:.4f}, Mean: {ps['mean']:.4f}")
        return stats


# ===========================================================================
# Internal helpers
# ===========================================================================
def _interpolate_missing_in_time(reflectivity: np.ndarray) -> None:
    """Fill NaNs pixel-wise along the time axis (in place).

    Interior gaps take the mean of both neighbours when available, otherwise
    the single valid neighbour. The first and last frames are left untouched.
    """
    for i in range(reflectivity.shape[1]):  # y
        for j in range(reflectivity.shape[2]):  # x
            series = reflectivity[:, i, j]
            nan_idx = np.where(np.isnan(series))[0]
            if len(nan_idx) == 0:
                continue
            for k in nan_idx:
                if 0 < k < len(series) - 1:
                    if not np.isnan(series[k - 1]) and not np.isnan(series[k + 1]):
                        series[k] = 0.5 * (series[k - 1] + series[k + 1])
                    elif not np.isnan(series[k - 1]):
                        series[k] = series[k - 1]
                    elif not np.isnan(series[k + 1]):
                        series[k] = series[k + 1]
