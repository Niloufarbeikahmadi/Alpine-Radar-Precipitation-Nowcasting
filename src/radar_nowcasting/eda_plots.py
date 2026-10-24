"""Task 1 - Exploratory visualisations of radar reflectivity and rainfall."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import ndimage

from .config import (
    HEAVY_RAIN_PLOT_THRESHOLD,
    MIN_STORM_CELL_PIXELS,
    SIGNIFICANT_RAIN_THRESHOLD,
    STORM_CORE_THRESHOLD,
)

PathLike = Union[str, Path]
_LOG = logging.getLogger("StormNowcastingEDA")


def create_all(data: xr.Dataset, output_dir: PathLike, logger: Optional[logging.Logger] = None) -> None:
    """Generate every Task 1 figure into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_temporal_evolution(data, output_dir, logger)
    plot_spatial_patterns(data, output_dir, logger)
    plot_statistical_distributions(data, output_dir, logger)
    plot_storm_cell_tracking(data, output_dir, logger)


# ---------------------------------------------------------------------------
def plot_temporal_evolution(data: xr.Dataset, output_dir: PathLike, logger: Optional[logging.Logger] = None) -> Path:
    """Domain-mean reflectivity / rainfall, heavy-rain area and max rainfall over time."""
    logger = logger or _LOG
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    time_coords = data.time.values
    mean_reflectivity = data.reflectivity.mean(dim=["x", "y"]).values
    mean_rainfall = data.rainfall_intensity.mean(dim=["x", "y"]).values

    axes[0, 0].plot(time_coords, mean_reflectivity, "b-", linewidth=2)
    axes[0, 0].set_title("Temporal Evolution of Mean Reflectivity")
    axes[0, 0].set_ylabel("Reflectivity (native units)")

    axes[0, 1].plot(time_coords, mean_rainfall, "r-", linewidth=2)
    axes[0, 1].set_title("Temporal Evolution of Mean Rainfall Intensity")
    axes[0, 1].set_ylabel("Rainfall Intensity (mm/h)")

    heavy_area = (data.rainfall_intensity > HEAVY_RAIN_PLOT_THRESHOLD).sum(dim=["x", "y"]).values
    axes[1, 0].plot(time_coords, heavy_area, "g-", linewidth=2)
    axes[1, 0].set_title("Area with Heavy Rainfall (>7.5 mm/h) Over Time")
    axes[1, 0].set_ylabel("Pixel Count")
    axes[1, 0].set_xlabel("Time")

    max_rainfall = data.rainfall_intensity.max(dim=["x", "y"]).values
    axes[1, 1].plot(time_coords, max_rainfall, "purple", linewidth=2)
    axes[1, 1].set_title("Maximum Rainfall Intensity Over Time")
    axes[1, 1].set_ylabel("Max Rainfall (mm/h)")
    axes[1, 1].set_xlabel("Time")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    path = Path(output_dir) / "temporal_evolution.png"
    _save(fig, path)
    logger.info("Temporal evolution plots saved")
    return path


# ---------------------------------------------------------------------------
def plot_spatial_patterns(data: xr.Dataset, output_dir: PathLike, logger: Optional[logging.Logger] = None) -> Path:
    """Rainfall snapshots at four key timestamps with 10 mm/h storm-core contours."""
    logger = logger or _LOG
    n_time = len(data.time)
    time_indices = [0, n_time // 3, 2 * n_time // 3, -1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, t_idx in enumerate(time_indices):
        rainfall = data.rainfall_intensity.isel(time=t_idx).values
        t_str = str(data.time.isel(time=t_idx).values)

        im = axes[idx].imshow(
            rainfall,
            cmap="viridis",
            vmin=0,
            vmax=max(1.0, np.nanmax(rainfall)),
            origin="lower",
        )
        axes[idx].set_title(f"Rainfall Intensity at {t_str[:19]}")
        axes[idx].set_xlabel("X Coordinate")
        axes[idx].set_ylabel("Y Coordinate")
        plt.colorbar(im, ax=axes[idx], label="Rainfall Intensity (mm/h)")

        if np.nanmax(rainfall) > STORM_CORE_THRESHOLD:
            cs = axes[idx].contour(rainfall, levels=[STORM_CORE_THRESHOLD], colors="red", linewidths=2)
            axes[idx].clabel(cs, inline=True, fontsize=8)

    path = Path(output_dir) / "spatial_patterns.png"
    _save(fig, path)
    logger.info("Spatial pattern plots saved")
    return path


# ---------------------------------------------------------------------------
def plot_statistical_distributions(
    data: xr.Dataset, output_dir: PathLike, logger: Optional[logging.Logger] = None
) -> Path:
    """Histograms, Z-R scatter and CDF of rainfall intensity."""
    logger = logger or _LOG
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    refl_flat = data.reflectivity.values.flatten()
    axes[0, 0].hist(refl_flat, bins=50, alpha=0.7, color="blue", edgecolor="black")
    axes[0, 0].set_title("Distribution of Radar Reflectivity")
    axes[0, 0].set_xlabel("Reflectivity (native units)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(True, alpha=0.3)

    rain_flat = data.rainfall_intensity.values.flatten()
    sig_mask = rain_flat > SIGNIFICANT_RAIN_THRESHOLD
    rain_sig = rain_flat[sig_mask]
    refl_sig = refl_flat[sig_mask]
    has_rain = len(rain_sig) > 0

    # Rainfall histogram
    if has_rain:
        axes[0, 1].hist(rain_sig, bins=50, alpha=0.7, color="green", edgecolor="black", log=True)
    else:
        axes[0, 1].hist(rain_flat, bins=10, alpha=0.7, color="green", edgecolor="black")
    axes[0, 1].set_title("Distribution of Rainfall Intensity")
    axes[0, 1].set_xlabel("Rainfall Intensity (mm/h)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    # Reflectivity vs rainfall scatter
    if has_rain:
        sample_size = min(10000, len(rain_sig))
        idx = np.random.choice(len(rain_sig), sample_size, replace=False)
        axes[1, 0].scatter(refl_sig[idx], rain_sig[idx], alpha=0.5, s=1, color="purple")
        axes[1, 0].set_ylabel("Rainfall (mm/h)")
    else:
        _no_data(axes[1, 0])
    axes[1, 0].set_title("Reflectivity vs Rainfall Intensity")
    axes[1, 0].set_xlabel("Reflectivity (native units)")
    axes[1, 0].grid(True, alpha=0.3)

    # Cumulative distribution
    if has_rain:
        srt = np.sort(rain_sig)
        cdf = np.arange(1, len(srt) + 1) / len(srt)
        axes[1, 1].plot(srt, cdf, "r-", linewidth=2)
        axes[1, 1].set_xscale("log")
    else:
        _no_data(axes[1, 1])
    axes[1, 1].set_title("Cumulative Distribution of Rainfall Intensity")
    axes[1, 1].set_xlabel("Rainfall Intensity (mm/h)")
    axes[1, 1].set_ylabel("Cumulative Probability")
    axes[1, 1].grid(True, alpha=0.3)

    path = Path(output_dir) / "statistical_distributions.png"
    _save(fig, path)
    logger.info("Statistical distribution plots saved")
    return path


# ---------------------------------------------------------------------------
def plot_storm_cell_tracking(data: xr.Dataset, output_dir: PathLike, logger: Optional[logging.Logger] = None) -> Path:
    """20-panel time-lapse with connected-component storm cells (> 10 mm/h) labelled."""
    logger = logger or _LOG
    n_timesteps = min(20, len(data.time))
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()

    for i in range(n_timesteps):
        rainfall = data.rainfall_intensity.isel(time=i).values
        axes[i].imshow(
            rainfall,
            cmap="YlOrRd",
            vmin=0,
            vmax=max(1.0, np.nanmax(rainfall)),
            origin="lower",
        )
        axes[i].set_title(f"T+{i * 5} min")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")

        storm_mask = rainfall > STORM_CORE_THRESHOLD
        if np.any(storm_mask):
            labeled, num = ndimage.label(storm_mask)
            for lbl in range(1, num + 1):
                ys, xs = np.where(labeled == lbl)
                if len(ys) <= MIN_STORM_CELL_PIXELS:
                    continue
                cy, cx = np.mean(ys), np.mean(xs)
                axes[i].text(
                    cx,
                    cy,
                    f"SC{lbl}",
                    fontsize=8,
                    color="blue",
                    weight="bold",
                    bbox=dict(boxstyle="circle,pad=0.3", fc="white", alpha=0.7),
                )

    for i in range(n_timesteps, len(axes)):
        fig.delaxes(axes[i])

    path = Path(output_dir) / "storm_cell_tracking.png"
    _save(fig, path)
    logger.info("Storm cell tracking visualization saved")
    return path


# ---------------------------------------------------------------------------
def _no_data(ax) -> None:
    ax.text(0.5, 0.5, "No significant rainfall data", ha="center", va="center", transform=ax.transAxes)


def _save(fig, path: Path, dpi: int = 300) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
