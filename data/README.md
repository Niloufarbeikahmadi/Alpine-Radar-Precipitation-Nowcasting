# Data

## `raw/` — radar reflectivity (not included)

The study uses a single convective episode of weather-radar reflectivity over
the Alpine region. The raw file was provided by a third party and is **not
redistributed** in this repository. To run Task 1, place it at:

```
data/raw/radar_data.nc
```

Expected NetCDF structure:

| Item | Value |
|---|---|
| Variable | `reflectivity(time, y, x)` |
| Coordinates | `time`, `y`, `x` |
| Grid | 64 × 64 pixels |
| Time steps | 50 frames, every 5 min (2025-06-01 10:00 → 14:05) |
| Units | Min–max scaled to [0, 1] (≈ 0.052 – 1.0). Physical dBZ is also supported. |

No radar file? Generate a synthetic cube with the same structure to try the pipeline:

```bash
python scripts/run_eda.py --data data/raw/demo.nc --synthetic
```

## `processed/` — model-ready sequences (included)

Produced by Task 1 (`scripts/run_eda.py`) and consumed by Task 2.

| File | Shape / content |
|---|---|
| `X_sequences.npy` | `(43, 5, 64, 64, 1)` float32 — 5 input frames (25 min) per sample |
| `y_targets.npy` | `(43, 3, 64, 64, 1)` float32 — next 3 frames (15 min) to forecast |
| `scaler.json` | `{"type": "zscore_log_rain", "mean": 0.0904, "std": 0.5108}` — z-score of `log(1 + R)` |
| `summary_stats.json` | Data shape, time range, reflectivity & rainfall statistics, storm-pixel counts |

Values are z-scored `log(1 + R)` rainfall, where `R` (mm/h) is derived from
reflectivity with the Marshall–Palmer relation `Z = 200 · R^1.6`.
Invert with `radar_nowcasting.preprocessing.inverse_transform_predictions`.
