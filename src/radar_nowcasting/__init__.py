"""
radar_nowcasting
================

Short-term precipitation nowcasting from weather-radar reflectivity.

Modules
-------
config            Paths and every hyper-parameter of the study
preprocessing     Task 1 - loading, QC, Z-R conversion, normalisation, sequences
eda_plots         Task 1 - exploratory figures
baseline          Eulerian persistence forecast
models            Task 2 - ConvLSTM architectures (light / moderate / U-Net)
training          Task 2 - callbacks, diagnostics, learning curves
nowcaster         Task 2 - ``NowcastingModels`` workflow class
metrics           Verification scores (RMSE, MAE, CSI, POD, FAR, HSS, ...)
evaluation        Task 3 - ``StormNowcastingEvaluation`` and reports
evaluation_plots  Task 2/3 - forecast comparison and diagnostic figures
pipeline          ``run_task1`` / ``run_task2`` / ``run_task3`` orchestration
synthetic         Synthetic radar cube for smoke tests

TensorFlow is imported lazily, so Task 1 and Task 3 run without it.
"""

__version__ = "1.0.0"
__author__ = "Niloufar Beikahmadi"
