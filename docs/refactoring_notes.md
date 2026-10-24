# Engineering notes: from research script to package

The study was first written as one 3,000-line script (`code.py`) that ran the
three tasks top to bottom. This repository reorganises that code into a small
Python package. **The models, hyper-parameters, pre-processing and evaluation
logic are unchanged.** This document lists where each piece now lives and how
the equivalence was checked.

## Where things moved

| Original `code.py` | New location |
|---|---|
| Hard-coded constants spread through the script | `src/radar_nowcasting/config.py` (one place, same values) |
| `StormCellNowcastingEDA` (loading, QC, Z–R, sequences, statistics) | `preprocessing.py` |
| `StormCellNowcastingEDA._plot_*` | `eda_plots.py` |
| `save_task1_results`, logging set-up | `utils.py` |
| `NowcastingModels` — **3 copies** (heavy U-Net, moderate, light), the last one overriding the others | One `NowcastingModels` class in `nowcaster.py` with `architecture="light" \| "moderate" \| "unet"` |
| `build_unet_convlstm` (three versions) | `models.py`: `build_unet_convlstm`, `build_moderate_convlstm`, `build_light_convlstm` |
| `persistence_forecast` | `baseline.py` |
| Callbacks, training diagnostics, `plot_training_history` | `training.py` |
| `calculate_metrics`, `calculate_comprehensive_metrics` | `metrics.py` |
| `StormNowcastingEvaluation` | `evaluation.py` (figures in `evaluation_plots.py`) |
| `main`, `main_task2`, `main_task3` | `pipeline.py` → `run_task1`, `run_task2`, `run_task3` |
| `create_dummy_radar_data` (called but not defined) | `synthetic.py` (used only for tests and demos) |

## Changes to how files are saved (no effect on results)

* **No more pickles of live objects.** Task 2 used to pickle the whole
  `NowcastingModels` object (Keras model and logger included). That file can
  only be re-loaded from the exact script that wrote it. Outputs are now:
  * `outputs/predictions/test_predictions.npz` — test inputs, truth and both forecasts (mm/h)
  * `outputs/metrics/task2_metrics.json`, `outputs/metrics/training_history.json`
  * `data/processed/scaler.json`, `data/processed/summary_stats.json` (were `.pkl`)
* The deep-learning model's result key is `convlstm` (it was `unet`, left over
  from the first architecture). The published metric files were re-keyed;
  their values are unchanged.
* Every output goes to a fixed folder (`data/processed`, `outputs/...`, `logs/`)
  instead of the current working directory.
* Unused imports (`cartopy`, `dill`, `plotly`, `confusion_matrix`, …) were
  removed, so the environment is much lighter.

## Equivalence checks

The original classes were loaded from `code.py` and run side by side with the
package on the same inputs:

| Check | Result |
|---|---|
| Task 1 on a synthetic cube with injected NaNs: rainfall field, `X`, `y`, scaler, summary statistics | identical |
| Task 2, all three architectures: layer-by-layer config, parameter count, initial weights (same seed), optimiser | identical (light 35,465 · moderate 140,049 · U-Net 5,758,273 parameters) |
| Task 2, light model trained for 2 epochs on the real sequences with the same seed | identical loss history and predictions |
| Train/val/test split (`random_state = 42`) | reproduces the published test set exactly |
| Task 3 metrics recomputed from the published predictions (same environment) | identical, all 80 values |
| Published weights (`best_nowcasting_model.h5`) → test forecasts | match the published forecasts to < 1e-4 mm/h |

Recomputing the Task 3 metrics with a newer NumPy release gives differences
of about 1e-7 (relative) from the stored files. This comes from float32
summation order, not from the code. `tests/test_pipeline.py` checks this
with a 1e-5 tolerance.

## One fix in a rejected variant

The **heavy U-Net** copy of `inverse_transform_predictions` checked for the
scaler type `"zscore_lograin"`, while Task 1 writes `"zscore_log_rain"`. That
variant therefore never undid the z-score before `expm1`. All variants now
share the correct inverse transform. **The adopted light ConvLSTM and every
reported number are unaffected.**

## Notes on the published results

These are kept exactly as reported. They are listed here so readers can
interpret them correctly:

* The `extreme_bias` and `extreme_mae` scores are defined on the top 5 % of
  observed pixels. In this test set more than 95 % of pixels are dry, so the
  95th percentile is 0 mm/h and these two scores equal the overall bias and MAE.
* The 43 sliding windows overlap and come from one storm. They are shuffled
  before the train/val/test split, so neighbouring windows can appear in
  different subsets. A blocked (chronological) split across several events
  would be the next step for an operational assessment.
