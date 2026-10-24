"""
Visual diagnostics comparing the persistence baseline with the ConvLSTM model.

* :func:`plot_prediction_comparison` - Task 2 sample forecast panel
* :func:`plot_metrics_dashboard`     - Task 3 six-panel skill dashboard
* :func:`plot_spatial_error_maps`    - Task 3 truth / forecasts / error difference
* :func:`plot_temporal_analysis`     - Task 3 lead-time skill & extreme events
* :func:`plot_statistical_distributions` - Task 3 error & scatter diagnostics
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np

from .metrics import csi_by_lead_time

PathLike = Union[str, Path]
Metrics = Dict[str, Dict]

PERS, CONV = "persistence", "convlstm"


def _save(fig, path: PathLike, dpi: int = 300) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# Task 2
# ===========================================================================
def plot_prediction_comparison(
    y_true: np.ndarray,
    y_pers: np.ndarray,
    y_conv: np.ndarray,
    idx: int,
    output_path: PathLike,
) -> Path:
    """Rows: truth, persistence, ConvLSTM; columns: forecast steps of sample ``idx``."""
    sample_true, sample_pers, sample_conv = y_true[idx], y_pers[idx], y_conv[idx]
    vmax = max(np.nanmax(sample_true), np.nanmax(sample_pers), np.nanmax(sample_conv), 1.0)

    t_out = sample_true.shape[0]
    fig, axes = plt.subplots(3, t_out, figsize=(4 * t_out, 9))
    if t_out == 1:
        axes = np.expand_dims(axes, axis=1)

    cmap = "inferno"
    rows = [("True", sample_true), ("Persistence", sample_pers), ("ConvLSTM", sample_conv)]
    for t in range(t_out):
        for r, (label, sample) in enumerate(rows):
            axes[r, t].imshow(sample[t, :, :, 0], origin="lower", vmin=0, vmax=vmax, cmap=cmap)
            axes[r, t].set_title(f"{label} t+{t + 1}")
            axes[r, t].set_xticks([])
            axes[r, t].set_yticks([])

    fig.colorbar(
        axes[0, 0].images[0],
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.04,
        pad=0.2,
        label="Rainfall (mm/h)",
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    return _save(fig, output_path)


# ===========================================================================
# Task 3
# ===========================================================================
def plot_metrics_dashboard(metrics: Metrics, output_path: PathLike) -> Path:
    """Six-panel dashboard: errors, improvements, lead time, CSI, POD-FAR, volume bias."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    p, c = metrics[PERS], metrics[CONV]

    # 1. RMSE & MAE comparison
    names = ["RMSE", "MAE"]
    x = np.arange(len(names))
    width = 0.35
    axes[0, 0].bar(x - width / 2, [p["rmse"], p["mae"]], width, label="Persistence", alpha=0.8, color="skyblue")
    axes[0, 0].bar(x + width / 2, [c["rmse"], c["mae"]], width, label="ConvLSTM", alpha=0.8, color="coral")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names)
    axes[0, 0].set_ylabel("Error (mm/h)")
    axes[0, 0].set_title("Error Metrics Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # 2. Improvement percentages (sign reversed for CSI: higher is better)
    improvements = [
        (pv - cv) / pv * 100 if pv != 0 else 0
        for pv, cv in zip([p["rmse"], p["mae"], p["heavy_csi"]], [c["rmse"], c["mae"], c["heavy_csi"]])
    ]
    improvements[2] = -improvements[2]
    colors = ["green" if i > 0 else "red" for i in improvements]
    axes[0, 1].bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
    axes[0, 1].set_xticks(range(len(improvements)))
    axes[0, 1].set_xticklabels(["RMSE", "MAE", "CSI (Heavy)"])
    axes[0, 1].set_ylabel("Improvement (%)")
    axes[0, 1].set_title("ConvLSTM Improvement over Persistence")
    axes[0, 1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(improvements):
        axes[0, 1].text(
            i, v + (2 if v > 0 else -2), f"{v:.1f}%", ha="center", va="bottom" if v > 0 else "top", fontweight="bold"
        )

    # 3. Lead-time performance
    n_lead = len(p["lead_time_metrics"])
    lead_labels = [f"T+{i + 1}" for i in range(n_lead)]
    pers_rmse = [p["lead_time_metrics"][f"lead_time_{i + 1}"]["rmse"] for i in range(n_lead)]
    conv_rmse = [c["lead_time_metrics"][f"lead_time_{i + 1}"]["rmse"] for i in range(n_lead)]
    axes[0, 2].plot(lead_labels, pers_rmse, "o-", label="Persistence", linewidth=2.5, markersize=8)
    axes[0, 2].plot(lead_labels, conv_rmse, "s-", label="ConvLSTM", linewidth=2.5, markersize=8)
    axes[0, 2].set_xlabel("Lead Time")
    axes[0, 2].set_ylabel("RMSE (mm/h)")
    axes[0, 2].set_title("Error Growth with Lead Time")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. CSI by threshold
    thresholds = ["light", "moderate", "heavy", "violent"]
    axes[1, 0].plot(
        thresholds, [p[f"{t}_csi"] for t in thresholds], "o-", label="Persistence", linewidth=2.5, markersize=8
    )
    axes[1, 0].plot(
        thresholds, [c[f"{t}_csi"] for t in thresholds], "s-", label="ConvLSTM", linewidth=2.5, markersize=8
    )
    axes[1, 0].set_xlabel("Rainfall Threshold")
    axes[1, 0].set_ylabel("Critical Success Index")
    axes[1, 0].set_title("Detection Skill by Rainfall Intensity")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # 5. POD-FAR performance diagram
    pod_p = [p[f"{t}_pod"] for t in thresholds]
    far_p = [p[f"{t}_far"] for t in thresholds]
    pod_c = [c[f"{t}_pod"] for t in thresholds]
    far_c = [c[f"{t}_far"] for t in thresholds]
    axes[1, 1].scatter(far_p, pod_p, s=150, label="Persistence", alpha=0.7, marker="o")
    axes[1, 1].scatter(far_c, pod_c, s=150, label="ConvLSTM", alpha=0.7, marker="s")
    for i, t in enumerate(thresholds):
        axes[1, 1].annotate(t.capitalize(), (far_p[i], pod_p[i]), fontsize=9, ha="right", va="bottom")
        axes[1, 1].annotate(t.capitalize(), (far_c[i], pod_c[i]), fontsize=9, ha="left", va="bottom")
    axes[1, 1].set_xlabel("False Alarm Ratio")
    axes[1, 1].set_ylabel("Probability of Detection")
    axes[1, 1].set_title("POD–FAR Performance Diagram")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])

    # 6. Volume bias
    volume_bias = [p["volume_bias"], c["volume_bias"]]
    bars = axes[1, 2].bar(
        ["Persistence", "ConvLSTM"], volume_bias, color=["red" if b > 0 else "blue" for b in volume_bias], alpha=0.7
    )
    axes[1, 2].set_ylabel("Volume Bias (%)")
    axes[1, 2].set_title("Total Rainfall Volume Bias")
    axes[1, 2].axhline(y=0, color="black", linestyle="-", alpha=0.5)
    axes[1, 2].grid(True, alpha=0.3, axis="y")
    for bar in bars:
        height = bar.get_height()
        axes[1, 2].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontweight="bold",
        )

    plt.suptitle("Comprehensive Model Evaluation Dashboard", fontsize=16, y=0.995)
    plt.tight_layout()
    return _save(fig, output_path)


def plot_spatial_error_maps(
    y_true: np.ndarray,
    y_pers: np.ndarray,
    y_conv: np.ndarray,
    output_path: PathLike,
    sample_idx: int = 0,
) -> Path:
    """Per lead time: truth, persistence, ConvLSTM and |error| difference (ConvLSTM - Pers)."""
    sample_idx = min(sample_idx, len(y_true) - 1)
    n_lead = y_true.shape[1]
    fig, axes = plt.subplots(n_lead, 4, figsize=(20, 5 * n_lead))

    for t in range(n_lead):
        vmax = np.max(y_true[sample_idx])
        panels = [
            (y_true, f"Ground Truth T+{t + 1}"),
            (y_pers, f"Persistence T+{t + 1}"),
            (y_conv, f"ConvLSTM T+{t + 1}"),
        ]
        for col, (arr, title) in enumerate(panels):
            im = axes[t, col].imshow(arr[sample_idx, t, :, :, 0], cmap="YlGnBu", vmin=0, vmax=vmax)
            axes[t, col].set_title(title)
            plt.colorbar(im, ax=axes[t, col], label="Rain (mm/h)")

        truth = y_true[sample_idx, t, :, :, 0]
        pers_error = np.abs(y_pers[sample_idx, t, :, :, 0] - truth)
        conv_error = np.abs(y_conv[sample_idx, t, :, :, 0] - truth)
        error_max = max(pers_error.max(), conv_error.max())
        im3 = axes[t, 3].imshow(conv_error - pers_error, cmap="RdBu_r", vmin=-error_max, vmax=error_max)
        axes[t, 3].set_title(f"Error Diff (ConvLSTM − Pers) T+{t + 1}")
        plt.colorbar(im3, ax=axes[t, 3]).set_label("Error Diff (mm/h)")

    plt.suptitle("Spatial Prediction Comparison", fontsize=16, y=0.995)
    plt.tight_layout()
    return _save(fig, output_path)


def plot_temporal_analysis(
    metrics: Metrics,
    y_true: np.ndarray,
    y_pers: np.ndarray,
    y_conv: np.ndarray,
    output_path: PathLike,
    heavy_threshold: float = 7.5,
) -> Path:
    """MAE & heavy-rain CSI vs lead time, temporal correlations, extreme-event errors."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    n_lead = y_true.shape[1]
    lead_times = list(range(1, n_lead + 1))

    # Error growth
    pers_mae = [metrics[PERS]["lead_time_metrics"][f"lead_time_{i}"]["mae"] for i in lead_times]
    conv_mae = [metrics[CONV]["lead_time_metrics"][f"lead_time_{i}"]["mae"] for i in lead_times]
    axes[0, 0].plot(lead_times, pers_mae, "o-", label="Persistence", linewidth=2.5, markersize=8)
    axes[0, 0].plot(lead_times, conv_mae, "s-", label="ConvLSTM", linewidth=2.5, markersize=8)
    axes[0, 0].set_xlabel("Lead Time (steps)")
    axes[0, 0].set_ylabel("MAE (mm/h)")
    axes[0, 0].set_title("Error Growth with Lead Time")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Heavy-rain CSI degradation
    axes[0, 1].plot(
        lead_times,
        csi_by_lead_time(y_true, y_pers, heavy_threshold),
        "o-",
        label="Persistence",
        linewidth=2.5,
        markersize=8,
    )
    axes[0, 1].plot(
        lead_times,
        csi_by_lead_time(y_true, y_conv, heavy_threshold),
        "s-",
        label="ConvLSTM",
        linewidth=2.5,
        markersize=8,
    )
    axes[0, 1].set_xlabel("Lead Time (steps)")
    axes[0, 1].set_ylabel("CSI (>7.5 mm/h)")
    axes[0, 1].set_title("Detection Skill Degradation")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Temporal correlation of domain-mean series
    corrs_pers, corrs_conv = [], []
    for i in range(min(20, len(y_true))):
        true_series = y_true[i].mean(axis=(1, 2)).flatten()
        pers_series = y_pers[i].mean(axis=(1, 2)).flatten()
        conv_series = y_conv[i].mean(axis=(1, 2)).flatten()
        if true_series.std() > 0:
            cp = np.corrcoef(true_series, pers_series)[0, 1]
            cc = np.corrcoef(true_series, conv_series)[0, 1]
            if not np.isnan(cp):
                corrs_pers.append(cp)
            if not np.isnan(cc):
                corrs_conv.append(cc)
    if corrs_pers and corrs_conv:
        axes[1, 0].hist(corrs_pers, alpha=0.6, label="Persistence", bins=15, color="skyblue")
        axes[1, 0].hist(corrs_conv, alpha=0.6, label="ConvLSTM", bins=15, color="coral")
        axes[1, 0].set_xlabel("Temporal Correlation")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Temporal Correlation Distribution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Extreme events (top 10 % of observed rainfall)
    threshold = np.percentile(y_true, 90)
    extreme_mask = y_true >= threshold
    if np.sum(extreme_mask) > 0:
        pers_err = np.abs(y_pers[extreme_mask] - y_true[extreme_mask])
        conv_err = np.abs(y_conv[extreme_mask] - y_true[extreme_mask])
        axes[1, 1].boxplot(
            [pers_err.flatten(), conv_err.flatten()], tick_labels=["Persistence", "ConvLSTM"], whis=5.0, showfliers=True
        )
        axes[1, 1].set_ylabel("Absolute Error (mm/h)")
        axes[1, 1].set_title("Extreme Events Error\n(Top 10% rainfall)")
        axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Temporal Evolution Analysis", fontsize=14)
    plt.tight_layout()
    return _save(fig, output_path)


def plot_statistical_distributions(
    y_true: np.ndarray, y_pers: np.ndarray, y_conv: np.ndarray, output_path: PathLike
) -> Path:
    """Error histograms, predicted-vs-observed scatter and residuals."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pers_errors = (y_pers - y_true).flatten()
    conv_errors = (y_conv - y_true).flatten()

    axes[0, 0].hist(pers_errors, bins=50, alpha=0.6, label="Persistence", color="skyblue", density=True)
    axes[0, 0].hist(conv_errors, bins=50, alpha=0.6, label="ConvLSTM", color="coral", density=True)
    axes[0, 0].axvline(0, color="black", linestyle="--", alpha=0.5)
    axes[0, 0].set_xlabel("Prediction Error (mm/h)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Error Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    n_pixels = len(y_true.flatten())
    indices = np.random.choice(n_pixels, min(5000, n_pixels), replace=False)
    true_s = y_true.flatten()[indices]
    pers_s = y_pers.flatten()[indices]
    conv_s = y_conv.flatten()[indices]

    for ax, pred_s, title, color in [
        (axes[0, 1], pers_s, "Persistence: Predicted vs True", None),
        (axes[1, 0], conv_s, "ConvLSTM: Predicted vs True", "coral"),
    ]:
        ax.scatter(true_s, pred_s, alpha=0.3, s=5, color=color)
        ax.plot([0, true_s.max()], [0, true_s.max()], "r--", alpha=0.5)
        ax.set_xlabel("True Rainfall (mm/h)")
        ax.set_ylabel("Predicted Rainfall (mm/h)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[1, 1].scatter(true_s, pers_errors[indices], alpha=0.3, s=5, label="Persistence")
    axes[1, 1].scatter(true_s, conv_errors[indices], alpha=0.3, s=5, label="ConvLSTM")
    axes[1, 1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("True Rainfall (mm/h)")
    axes[1, 1].set_ylabel("Prediction Error (mm/h)")
    axes[1, 1].set_title("Residual Plot")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Statistical Analysis", fontsize=14)
    plt.tight_layout()
    return _save(fig, output_path)
