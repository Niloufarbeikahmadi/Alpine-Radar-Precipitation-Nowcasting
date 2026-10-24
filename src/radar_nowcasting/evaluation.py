"""
Task 3 - Comprehensive evaluation of the nowcasting models.

``StormNowcastingEvaluation`` computes the full verification suite
(:mod:`radar_nowcasting.metrics`), renders the diagnostic figures
(:mod:`radar_nowcasting.evaluation_plots`) and writes JSON / TXT / CSV reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from . import evaluation_plots as plots
from .config import PATHS
from .metrics import comprehensive_metrics
from .utils import setup_logger

PathLike = Union[str, Path]
PERS, CONV = "persistence", "convlstm"


class StormNowcastingEvaluation:
    """Comprehensive evaluation pipeline for precipitation nowcasting models."""

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred_pers: np.ndarray,
        y_pred_conv: np.ndarray,
        results: Optional[Dict[str, Any]] = None,
        log_file: str = "storm_nowcasting_evaluation.log",
    ):
        self.y_true = y_true
        self.y_pred_pers = y_pred_pers
        self.y_pred_conv = y_pred_conv
        self.results = results  # Task 2 headline metrics (optional)
        self.logger = setup_logger(
            "StormNowcastingEvaluation",
            log_file,
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            reset_handlers=True,
        )
        self.logger.info("Storm Nowcasting Evaluation Logger Initialized")
        self.evaluation_results: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def calculate_comprehensive_metrics(self) -> Dict[str, Dict]:
        """Continuous, categorical, lead-time, spatial, extreme and volume metrics."""
        self.logger.info("Calculating comprehensive evaluation metrics")
        self.evaluation_results = {
            PERS: comprehensive_metrics(self.y_true, self.y_pred_pers),
            CONV: comprehensive_metrics(self.y_true, self.y_pred_conv),
        }
        self.logger.info("Comprehensive metrics calculation completed")
        return self.evaluation_results

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    def create_comparative_visualizations(self, output_dir: PathLike = PATHS.evaluation_figures_dir) -> None:
        """Render the dashboard, spatial maps, temporal and statistical diagnostics."""
        output_dir = Path(output_dir)
        self.logger.info(f"Creating visualizations in {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        plots.plot_metrics_dashboard(self.evaluation_results, output_dir / "metrics_dashboard.png")
        self.logger.info("Metrics dashboard created")

        plots.plot_spatial_error_maps(
            self.y_true, self.y_pred_pers, self.y_pred_conv, output_dir / "spatial_error_maps.png"
        )
        self.logger.info("Spatial error maps created")

        plots.plot_temporal_analysis(
            self.evaluation_results,
            self.y_true,
            self.y_pred_pers,
            self.y_pred_conv,
            output_dir / "temporal_analysis.png",
        )
        self.logger.info("Temporal analysis created")

        plots.plot_statistical_distributions(
            self.y_true,
            self.y_pred_pers,
            self.y_pred_conv,
            output_dir / "statistical_distributions.png",
        )
        self.logger.info("Statistical distributions created")
        self.logger.info(f"All visualizations saved to {output_dir}")

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    def build_summary(self) -> Dict[str, Any]:
        """Headline numbers used by the JSON / TXT reports."""
        p, c = self.evaluation_results[PERS], self.evaluation_results[CONV]
        return {
            "Overall Performance": {
                "Persistence": {
                    "RMSE (mm/h)": p["rmse"],
                    "MAE (mm/h)": p["mae"],
                    "Bias (mm/h)": p["bias"],
                    "Correlation": p["correlation"],
                },
                "ConvLSTM": {
                    "RMSE (mm/h)": c["rmse"],
                    "MAE (mm/h)": c["mae"],
                    "Bias (mm/h)": c["bias"],
                    "Correlation": c["correlation"],
                },
            },
            "Heavy Rainfall Detection (>7.5 mm/h)": {
                "Persistence": {"CSI": p["heavy_csi"], "POD": p["heavy_pod"], "FAR": p["heavy_far"]},
                "ConvLSTM": {"CSI": c["heavy_csi"], "POD": c["heavy_pod"], "FAR": c["heavy_far"]},
            },
            "Improvements": {
                "RMSE_improvement_%": (p["rmse"] - c["rmse"]) / p["rmse"] * 100,
                "MAE_improvement_%": (p["mae"] - c["mae"]) / p["mae"] * 100,
                "CSI_improvement_%": (c["heavy_csi"] - p["heavy_csi"]) / (p["heavy_csi"] + 1e-9) * 100,
            },
        }

    def save_reports(self, output_dir: PathLike = PATHS.metrics_dir) -> Path:
        """Write ``evaluation_summary.json``, ``evaluation_report.txt``,
        ``comprehensive_metrics.json`` and ``metrics_comparison.csv``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = self.build_summary()

        with open(output_dir / "evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        with open(output_dir / "evaluation_report.txt", "w") as f:
            f.write(format_text_report(self.evaluation_results, summary))

        with open(output_dir / "comprehensive_metrics.json", "w") as f:
            json.dump(self.evaluation_results, f, indent=4, default=str)

        pd.DataFrame(self.evaluation_results).to_csv(output_dir / "metrics_comparison.csv")

        self.logger.info(f"Summary reports created in {output_dir}")
        return output_dir


def format_text_report(metrics: Dict[str, Dict], summary: Dict[str, Any]) -> str:
    """Human-readable evaluation report."""
    p, c = metrics[PERS], metrics[CONV]
    imp = summary["Improvements"]
    line = "=" * 70
    lines = [
        line,
        "PRECIPITATION NOWCASTING EVALUATION REPORT",
        line,
        "",
        "OVERALL PERFORMANCE METRICS",
        "-" * 40,
        "",
        "Persistence Baseline:",
        f"  RMSE: {p['rmse']:.4f} mm/h",
        f"  MAE:  {p['mae']:.4f} mm/h",
        f"  Bias: {p['bias']:+.4f} mm/h",
        f"  Correlation: {p['correlation']:.4f}",
        "",
        "ConvLSTM Model:",
        f"  RMSE: {c['rmse']:.4f} mm/h",
        f"  MAE:  {c['mae']:.4f} mm/h",
        f"  Bias: {c['bias']:+.4f} mm/h",
        f"  Correlation: {c['correlation']:.4f}",
        "",
        "IMPROVEMENTS:",
        f"  RMSE Improvement: {imp['RMSE_improvement_%']:.2f}%",
        f"  MAE Improvement:  {imp['MAE_improvement_%']:.2f}%",
        f"  CSI Improvement:  {imp['CSI_improvement_%']:.2f}%",
        "",
        line,
        "",
    ]
    return "\n".join(lines)
