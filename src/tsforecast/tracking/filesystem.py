from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tsforecast.utils.paths import get_runs_dir

logger = logging.getLogger(__name__)


class RunTracker:
    def __init__(self, run_id: str, base_dir: str | Path = ".") -> None:
        self.run_id = run_id
        self.run_dir = Path(get_runs_dir(str(base_dir))) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_file(self) -> Path:
        return self.run_dir / "logs.txt"

    def ticker_dir(self, ticker: str) -> Path:
        """Return (and create) the per-ticker subdirectory: tickers/{ticker}/."""
        d = self.run_dir / "tickers" / ticker
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ticker_plot_path(self, ticker: str) -> Path:
        """Return the path where the forecast plot for *ticker* will be saved."""
        return self.ticker_dir(ticker) / "plot.png"

    def ticker_return_plot_path(self, ticker: str) -> Path:
        """Return the path where the return plot for *ticker* will be saved."""
        return self.ticker_dir(ticker) / "plot_returns.png"

    # Config
    def save_config(self, config: dict) -> None:
        """Write config dict to config.yaml (atomic write)."""
        dest = self.run_dir / "config.yaml"
        tmp = Path(str(dest) + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, default_flow_style=False, allow_unicode=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, dest)
        except OSError as exc:
            logger.error(f"Failed to save config to {dest}: {exc}")
            tmp.unlink(missing_ok=True)
            raise

    # Global metrics
    def save_global_metrics(self, metrics: dict) -> None:
        """Save global run metrics as metrics.json (atomic write)."""
        rounded = {
            k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
            for k, v in metrics.items()
        }
        dest = self.run_dir / "metrics.json"
        tmp = Path(str(dest) + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(json.dumps(rounded, indent=2))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, dest)
        except OSError as exc:
            logger.error(f"Failed to save metrics.json to {dest}: {exc}")
            tmp.unlink(missing_ok=True)
            raise


    # Per-ticker artifacts
    def save_ticker_metrics(self, ticker: str, metrics: dict) -> None:
        """Write per-ticker metrics to tickers/{ticker}/metrics.csv (atomic write)."""
        dest = self.ticker_dir(ticker) / "metrics.csv"
        rounded = {
            k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
            for k, v in metrics.items()
        }
        tmp = Path(str(dest) + ".tmp")
        try:
            pd.DataFrame([rounded]).to_csv(tmp, index=False)
            os.replace(tmp, dest)
        except OSError as exc:
            logger.error(f"Failed to save ticker metrics to {dest}: {exc}")
            tmp.unlink(missing_ok=True)
            raise

    def save_ticker_predictions(
        self,
        ticker: str,
        dates: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anchors: np.ndarray,
    ) -> None:
        """Save per-ticker predictions to tickers/{ticker}/predictions.csv (atomic write)."""
        H = y_true.shape[1]
        date_strings = pd.to_datetime(dates).strftime("%Y-%m-%d")
        data: dict = {"date": date_strings, "ticker": ticker, "anchor": anchors}
        for h in range(H):
            data[f"y_true_{h}"] = y_true[:, h]
        for h in range(H):
            data[f"y_pred_{h}"] = y_pred[:, h]
        dest = self.ticker_dir(ticker) / "predictions.csv"
        tmp = Path(str(dest) + ".tmp")
        try:
            pd.DataFrame(data).to_csv(tmp, index=False)
            os.replace(tmp, dest)
        except OSError as exc:
            logger.error(f"Failed to save predictions to {dest}: {exc}")
            tmp.unlink(missing_ok=True)
            raise

    def save_ticker_model(self, model, ticker: str) -> Path:
        """Call model.save(tickers/{ticker}/model) and return the path."""
        path = self.ticker_dir(ticker) / "model"
        model.save(path)
        return path
