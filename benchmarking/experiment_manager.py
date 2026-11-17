"""Experiment manager for tabular processing and export of empirical results."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from benchmarking.experiment_config import ExperimentConfig
from benchmarking.metrics import (
    AggregatedMetric,
    RawMeasurement,
    aggregated_metrics_to_dataframe,
    raw_measurements_to_dataframe,
)


class ExperimentManager:
    """Manages storage, tabular conversion, and serialization of benchmark results."""

    def __init__(
        self,
        config: ExperimentConfig,
        raw_measurements: Optional[List[RawMeasurement]] = None,
        aggregated_metrics: Optional[List[AggregatedMetric]] = None,
    ) -> None:
        self.config = config
        self.raw_measurements: List[RawMeasurement] = raw_measurements or []
        self.aggregated_metrics: List[AggregatedMetric] = aggregated_metrics or []

    def get_raw_dataframe(self) -> pd.DataFrame:
        """Return raw measurements as a Pandas DataFrame."""
        return raw_measurements_to_dataframe(self.raw_measurements)

    def get_aggregated_dataframe(self) -> pd.DataFrame:
        """Return aggregated summary metrics as a Pandas DataFrame."""
        return aggregated_metrics_to_dataframe(self.aggregated_metrics)

    def export_raw_to_csv(self, filepath: str | Path) -> Path:
        """Export raw measurement records to CSV."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.get_raw_dataframe()
        df.to_csv(path, index=False)
        return path

    def export_raw_to_json(self, filepath: str | Path) -> Path:
        """Export raw measurement records to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [m.to_dict() for m in self.raw_measurements]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def export_aggregated_to_csv(self, filepath: str | Path) -> Path:
        """Export aggregated metrics to CSV."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.get_aggregated_dataframe()
        df.to_csv(path, index=False)
        return path

    def export_aggregated_to_json(self, filepath: str | Path) -> Path:
        """Export aggregated metrics to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [m.to_dict() for m in self.aggregated_metrics]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def export_all(self, output_dir: str | Path) -> Dict[str, Path]:
        """Convenience method to export both raw and aggregated results to CSV and JSON."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        exp_id = self.config.experiment_id

        raw_csv = self.export_raw_to_csv(out_dir / f"raw_measurements_{exp_id}.csv")
        raw_json = self.export_raw_to_json(out_dir / f"raw_measurements_{exp_id}.json")
        agg_csv = self.export_aggregated_to_csv(out_dir / f"aggregated_metrics_{exp_id}.csv")
        agg_json = self.export_aggregated_to_json(out_dir / f"aggregated_metrics_{exp_id}.json")

        return {
            "raw_csv": raw_csv,
            "raw_json": raw_json,
            "aggregated_csv": agg_csv,
            "aggregated_json": agg_json,
        }
