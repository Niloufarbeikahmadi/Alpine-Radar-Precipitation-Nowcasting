"""Verification scores and the persistence baseline."""

import numpy as np
import pytest

from radar_nowcasting.baseline import persistence_forecast, persistence_predictions
from radar_nowcasting.metrics import (
    calculate_metrics,
    categorical_scores,
    comprehensive_metrics,
    contingency_table,
    critical_success_index,
    improvement_pct,
)


@pytest.fixture
def fields(rng):
    y_true = rng.gamma(0.5, 6.0, size=(4, 3, 8, 8, 1)).astype(np.float32)
    y_pred = np.clip(y_true + rng.normal(0, 1.0, size=y_true.shape), 0, None).astype(np.float32)
    return y_true, y_pred


def test_perfect_forecast(fields):
    y_true, _ = fields
    m = calculate_metrics(y_true, y_true)
    assert m["mse"] == m["mae"] == m["rmse"] == m["bias"] == 0.0
    assert m["heavy_csi"] == pytest.approx(1.0)


def test_contingency_table_counts():
    obs = np.array([10.0, 10.0, 0.0, 0.0, 8.0])
    fc = np.array([10.0, 0.0, 10.0, 0.0, 9.0])
    c = contingency_table(obs, fc, threshold=7.5)
    assert (c.hits, c.misses, c.false_alarms, c.correct_negatives) == (2, 1, 1, 1)

    s = categorical_scores(obs, fc, threshold=7.5)
    assert s["pod"] == pytest.approx(2 / 3)
    assert s["far"] == pytest.approx(1 / 3)
    assert s["csi"] == pytest.approx(2 / 4)
    # HSS = 2(H·CN − M·FA) / [(H+M)(M+CN) + (H+FA)(FA+CN)] = 2(2−1)/(3·2 + 3·2) = 1/6
    assert s["hss"] == pytest.approx(1 / 6)
    assert critical_success_index(obs, fc, 7.5) == pytest.approx(0.5)


def test_comprehensive_metrics_keys(fields):
    y_true, y_pred = fields
    m = comprehensive_metrics(y_true, y_pred)
    for thr in ["light", "moderate", "heavy", "violent", "extreme"]:
        for score in ["pod", "far", "csi", "hss"]:
            assert f"{thr}_{score}" in m
    assert list(m["lead_time_metrics"]) == ["lead_time_1", "lead_time_2", "lead_time_3"]
    assert -1.0 <= m["correlation"] <= 1.0
    assert m["rmse"] == pytest.approx(np.sqrt(m["mse"]))


def test_persistence_repeats_last_frame(rng):
    seq = rng.random((5, 8, 8, 1))
    fc = persistence_forecast(seq, nsteps=3)
    assert fc.shape == (3, 8, 8, 1)
    for t in range(3):
        np.testing.assert_array_equal(fc[t], seq[-1])

    batch = persistence_predictions(rng.random((2, 5, 8, 8, 1)), nsteps=3)
    assert batch.shape == (2, 3, 8, 8, 1)


def test_improvement_pct():
    assert improvement_pct(4.0, 3.0) == pytest.approx(25.0)
