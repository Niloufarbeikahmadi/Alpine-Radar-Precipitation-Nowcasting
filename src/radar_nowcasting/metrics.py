"""
Forecast-verification metrics for precipitation nowcasting.

All functions expect arrays in **physical units (mm/h)** with shape
``(N, T_out, H, W, 1)``.

Continuous scores
    MSE, RMSE, MAE, bias, normalised RMSE/MAE, Pearson correlation,
    spatial correlation, extreme-value errors, volume bias.
Categorical scores (2x2 contingency table at a rain-rate threshold)
    POD = hits / (hits + misses)
    FAR = false alarms / (hits + false alarms)
    CSI = hits / (hits + misses + false alarms)
    HSS = Heidke Skill Score
"""

from __future__ import annotations

from typing import Dict, Mapping, NamedTuple

import numpy as np

from .config import EPS, TASK2_CSI_THRESHOLDS, TASK3_THRESHOLDS


class Contingency(NamedTuple):
    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int


def contingency_table(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Contingency:
    """2x2 contingency table for the event ``rain >= threshold``."""
    obs_binary = (y_true >= threshold).astype(int)
    pred_binary = (y_pred >= threshold).astype(int)
    return Contingency(
        hits=np.sum((obs_binary == 1) & (pred_binary == 1)),
        misses=np.sum((obs_binary == 1) & (pred_binary == 0)),
        false_alarms=np.sum((obs_binary == 0) & (pred_binary == 1)),
        correct_negatives=np.sum((obs_binary == 0) & (pred_binary == 0)),
    )


def critical_success_index(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """CSI (threat score) for the event ``rain >= threshold``."""
    c = contingency_table(y_true, y_pred, threshold)
    return c.hits / (c.hits + c.misses + c.false_alarms + EPS)


def categorical_scores(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    """POD, FAR, CSI and HSS for the event ``rain >= threshold``."""
    h, m, f, cn = contingency_table(y_true, y_pred, threshold)
    pod = h / (h + m + EPS)
    far = f / (h + f + EPS)
    csi = h / (h + m + f + EPS)
    hss = 2 * (h * cn - m * f) / ((h + m) * (m + cn) + (h + f) * (f + cn) + EPS)
    return {"pod": float(pod), "far": float(far), "csi": float(csi), "hss": float(hss)}


# ===========================================================================
# Task 2 - headline metrics
# ===========================================================================
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Mapping[str, float] = TASK2_CSI_THRESHOLDS,
) -> Dict[str, float]:
    """MSE, MAE, RMSE, bias and CSI at moderate / heavy / violent thresholds."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    metrics = {
        "mse": mse,
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(mse)),
        "bias": float(np.mean(y_pred - y_true)),
    }
    for name, thr in thresholds.items():
        metrics[f"{name}_csi"] = float(critical_success_index(y_true, y_pred, thr))
    return metrics


# ===========================================================================
# Task 3 - comprehensive verification
# ===========================================================================
def comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Mapping[str, float] = TASK3_THRESHOLDS,
) -> Dict[str, object]:
    """Full meteorological verification of one model against observations."""
    m: Dict[str, object] = {}

    # --- Continuous error metrics -----------------------------------------
    error = y_pred - y_true
    abs_error = np.abs(error)
    m["mse"] = float(np.mean(error**2))
    m["rmse"] = float(np.sqrt(m["mse"]))
    m["mae"] = float(np.mean(abs_error))
    m["bias"] = float(np.mean(error))

    # --- Normalised metrics -----------------------------------------------
    mean_true = np.mean(y_true)
    m["nrmse"] = float(m["rmse"] / (mean_true + EPS))
    m["nmae"] = float(m["mae"] / (mean_true + EPS))

    # --- Correlation on co-located wet pixels -----------------------------
    flat_true, flat_pred = y_true.flatten(), y_pred.flatten()
    valid_mask = (flat_true > 0) & (flat_pred > 0)
    if np.sum(valid_mask) > 10:
        corr = np.corrcoef(flat_true[valid_mask], flat_pred[valid_mask])[0, 1]
        m["correlation"] = float(corr if not np.isnan(corr) else 0)
    else:
        m["correlation"] = 0.0

    # --- Categorical verification ----------------------------------------
    for thresh_name, threshold in thresholds.items():
        for score, value in categorical_scores(y_true, y_pred, threshold).items():
            m[f"{thresh_name}_{score}"] = value

    # --- Lead-time (temporal) metrics ------------------------------------
    m["lead_time_metrics"] = lead_time_metrics(y_true, y_pred)

    # --- Spatial skill ----------------------------------------------------
    m["mean_spatial_correlation"] = mean_spatial_correlation(y_true, y_pred)

    # --- Extreme values (top 5 % of observed rainfall) --------------------
    top_5_percentile = np.percentile(y_true, 95)
    extreme_mask = y_true >= top_5_percentile
    if np.sum(extreme_mask) > 0:
        extreme_error = y_pred[extreme_mask] - y_true[extreme_mask]
        m["extreme_bias"] = float(np.mean(extreme_error))
        m["extreme_mae"] = float(np.mean(np.abs(extreme_error)))
    else:
        m["extreme_bias"] = 0.0
        m["extreme_mae"] = 0.0

    # --- Domain-integrated volume bias (%) --------------------------------
    total_observed_volume = np.sum(y_true)
    total_predicted_volume = np.sum(y_pred)
    m["volume_bias"] = float((total_predicted_volume - total_observed_volume) / (total_observed_volume + EPS) * 100)
    return m


def lead_time_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """MSE / MAE / RMSE for each forecast step (T+1 ... T+T_out)."""
    out: Dict[str, Dict[str, float]] = {}
    for t in range(y_pred.shape[1]):
        t_error = y_pred[:, t] - y_true[:, t]
        t_mse = np.mean(t_error**2)
        out[f"lead_time_{t + 1}"] = {
            "mse": float(t_mse),
            "mae": float(np.mean(np.abs(t_error))),
            "rmse": float(np.sqrt(t_mse)),
        }
    return out


def mean_spatial_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average pattern correlation of the time-mean forecast field per sample."""
    spatial_corrs = []
    for i in range(len(y_true)):
        true_frame = y_true[i].mean(axis=0)
        pred_frame = y_pred[i].mean(axis=0)
        if true_frame.std() > 0 and pred_frame.std() > 0:
            corr = np.corrcoef(true_frame.flatten(), pred_frame.flatten())[0, 1]
            if not np.isnan(corr):
                spatial_corrs.append(corr)
    return float(np.mean(spatial_corrs) if spatial_corrs else 0)


def csi_by_lead_time(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> list:
    """CSI at ``threshold`` for each forecast step."""
    return [critical_success_index(y_true[:, t], y_pred[:, t], threshold) for t in range(y_true.shape[1])]


def improvement_pct(baseline: float, model: float) -> float:
    """Relative error reduction (%) of ``model`` vs ``baseline`` (lower is better)."""
    return (baseline - model) / max(baseline, 1e-8) * 100.0
