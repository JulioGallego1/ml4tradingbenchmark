"""
Main CLI entrypoint for the tsforecast pipeline.

Usage:
    python -m tsforecast.cli.train --model rf --regime bear --L 96 --H 21
    tsforecast-train --model lstm --regime bull --L 48 --H 63
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tsforecast.tracking.run_id import make_run_id
from tsforecast.tracking.filesystem import RunTracker
from tsforecast.utils.logging import get_logger
from tsforecast.training.reproducibility import set_seed

from tsforecast.evaluation.metrics import (
        directional_accuracy,
        final_return_mae,
        mae,
        mape,
        rmse,
        smape,
    )

from tsforecast.evaluation.plots import plot_ticker_forecast, plot_ticker_returns, plot_training_curves

from tsforecast.data.loaders import load_price_data
from tsforecast.data.scalers import KINDS as GLOBAL_SCALER_KINDS, GlobalScaler
from tsforecast.data.splits import make_time_splits
from tsforecast.data.windows import generate_windows_mimo

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a forecasting model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["rf", "lstm", "patchtst"],
        required=True,
        help="Model to train.",
    )
    parser.add_argument(
        "--regime",
        choices=["bear", "bull", "newest"],
        required=True,
        help="Market regime to use.",
    )
    parser.add_argument(
        "--L",
        type=int,
        required=True,
        help="Context length in trading days.",
    )
    parser.add_argument(
        "--H",
        type=int,
        required=True,
        help="Forecast horizon in trading days.",
    )
    parser.add_argument(
        "--strategy",
        choices=["mimo", "recursive"],
        default="mimo",
        help="Forecasting strategy: mimo (direct) or recursive (iterative in --step blocks).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Block size for recursive strategy (ignored if strategy=mimo).",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Project root for runs/ output.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override. Overrides the seed value from YAML configs.",
    )
    parser.add_argument(
        "--global-scaler",
        choices=list(GLOBAL_SCALER_KINDS),
        default="none",
        help=(
            "Global per-ticker scaler fitted on the FULL series (train+val+test). "
            "This intentionally introduces test-set leakage and is an experimental knob. "
            "Independent of any model-internal per-window scaling."
        ),
    )
    parser.add_argument(
        "--hparams",
        default=None,
        help=(
            "JSON string of hyperparameter overrides. Keys override values from YAML "
            "configs (e.g., '{\"n_estimators\": 200, \"max_depth\": 10}'). "
            "Used by sweep/Slurm launchers to inject hyperparameters."
        ),
    )
    args = parser.parse_args()
    if args.model == "rf" and args.strategy == "recursive":
        parser.error("RF only supports strategy=mimo (no recursive rollout).")
    return args


def load_configs(model: str, regime: str) -> dict:
    """Load and merge model, splits, and train YAML configs."""
    config_dir = Path("configs")
    model_cfg_path = config_dir / "model" / f"{model}.yaml"
    splits_cfg_path = config_dir / "splits.yaml"
    train_cfg_path = config_dir / "train.yaml"

    config = {}

    if train_cfg_path.exists():
        with open(train_cfg_path) as f:
            train_cfg = yaml.safe_load(f) or {}
        config.update(train_cfg)

    if model_cfg_path.exists():
        with open(model_cfg_path) as f:
            model_cfg = yaml.safe_load(f) or {}
        config.update(model_cfg)

    if splits_cfg_path.exists():
        with open(splits_cfg_path) as f:
            splits_cfg = yaml.safe_load(f) or {}
        regimes = splits_cfg.get("regimes", {})
        regime_cfg = regimes.get(regime, {})
        config.update(regime_cfg)

    return config


def build_model(
    model_name: str,
    config: dict,
    L: int,
    H: int,
    run_dir: Path,
    strategy: str = "mimo",
    step: int = 1,
):
    """Instantiate the requested model using config parameters."""
    if model_name == "rf":
        from tsforecast.models.rf import RandomForestModel
        return RandomForestModel(
            n_estimators=config.get("n_estimators", 200),
            max_features=config.get("max_features", "sqrt"),
            max_depth=config.get("max_depth", None),
            min_samples_leaf=config.get("min_samples_leaf", 1),
            random_state=config.get("seed", 2024),
        )

    elif model_name == "lstm":
        from tsforecast.models.lstm import LSTMModel

        return LSTMModel(
            context_length=L,
            horizon=H,
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
            lr=config.get("lr", 1e-3),
            max_epochs=config.get("max_epochs", 100),
            batch_size=config.get("batch_size", 32),
            patience=config.get("patience", 10),
            random_state=config.get("seed", 2024),
            strategy=strategy,
            step=step,
            use_window_scaling=config.get("use_window_scaling", True),
        )

    elif model_name == "patchtst":
        from tsforecast.models.patchtst import PatchTSTModel

        output_dir = str(run_dir / "patchtst_tmp")
        return PatchTSTModel(
            context_length=L,
            horizon=H,
            patch_length=config.get("patch_length", 16),
            patch_stride=config.get("patch_stride", 8),
            d_model=config.get("d_model", 128),
            num_attention_heads=config.get("num_attention_heads", 4),
            num_hidden_layers=config.get("num_hidden_layers", 3),
            ffn_dim=config.get("ffn_dim", 256),
            dropout=config.get("dropout", 0.2),
            lr=config.get("lr", 1e-4),
            max_epochs=config.get("max_epochs", 100),
            batch_size=config.get("batch_size", 32),
            patience=config.get("patience", 10),
            random_state=config.get("seed", 2024),
            output_dir=output_dir,
            strategy=strategy,
            step=step,
            use_window_scaling=config.get("use_window_scaling", True),
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


def process_ticker(
    parquet_path: Path,
    config: dict,
    L: int,
    H: int,
    logger,
    global_scaler_kind: str = "none",
) -> dict | None:
    """
    Load, split, and window data for a single ticker file.

    When ``global_scaler_kind`` is not ``"none"``, a per-ticker scaler is fitted
    on the FULL series (train+val+test, intentional leakage) and applied to the
    values before window generation; the fitted scaler is returned so callers
    can inverse-transform predictions back to the raw price scale.

    Returns a dict with keys:
        ticker, scaler,
        X_train, Y_train, anchors_train, dates_train,
        X_val,   Y_val,   anchors_val,   dates_val,
        X_test,  Y_test,  anchors_test,  dates_test,
    or None if the ticker cannot be processed.
    """
    ticker = parquet_path.stem
    test_start = config["test_start"]
    test_days = config.get("test_days", 252)
    val_days = config.get("val_days", 252)

    logger.info(f"  [{ticker}] Generating windows.")
    try:
        df = load_price_data(parquet_path)
    except Exception as exc:
        logger.warning(f"  [{ticker}] Failed to load: {exc}")
        return None

    values = df["Close"].values.astype(np.float32)
    dates_arr = df["Date"].values

    scaler = GlobalScaler(kind=global_scaler_kind).fit(values)
    values_scaled = scaler.transform(values).astype(np.float32, copy=False)

    try:
        splits = make_time_splits(
            df,
            test_start=test_start,
            test_days=test_days,
            val_days=val_days,
            context_length=L,
        )
    except Exception as exc:
        logger.warning(f"  [{ticker}] make_time_splits failed: {exc}")
        return None

    train_start, train_end = splits["train"]
    val_start, val_end = splits["val"]
    test_start_idx, test_end_idx = splits["test"]

    try:
        X_train, Y_train, anchors_train, dates_train = generate_windows_mimo(
            values_scaled, dates_arr, train_start, train_end, L, H
        )
        X_val, Y_val, anchors_val, dates_val = generate_windows_mimo(
            values_scaled, dates_arr, val_start, val_end, L, H
        )
        X_test, Y_test, anchors_test, dates_test = generate_windows_mimo(
            values_scaled, dates_arr, test_start_idx, test_end_idx, L, H
        )
    except Exception as exc:
        logger.warning(f"  [{ticker}] Window generation failed: {exc}")
        return None

    if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
        logger.warning(f"  [{ticker}] Empty split — skipping.")
        return None

    return {
        "ticker": ticker,
        "scaler": scaler,
        "X_train": X_train,
        "Y_train": Y_train,
        "anchors_train": anchors_train,
        "dates_train": dates_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "anchors_val": anchors_val,
        "dates_val": dates_val,
        "X_test": X_test,
        "Y_test": Y_test,
        "anchors_test": anchors_test,
        "dates_test": dates_test,
    }


def _aggregate_ticker_metrics(ticker_metrics: list[dict]) -> dict:
    """Compute arithmetic mean of numeric metrics across tickers."""
    if not ticker_metrics:
        return {}
    all_keys: set[str] = set()
    for m in ticker_metrics:
        all_keys.update(m.keys())
    result: dict = {}
    for key in sorted(all_keys):
        values = [
            m[key]
            for m in ticker_metrics
            if key in m and isinstance(m[key], (int, float))
        ]
        if values:
            result[key] = float(np.mean(values))
    return result


def _run_per_ticker(
    all_results: list[dict],
    tracker,
    model_name: str,
    config: dict,
    L: int,
    H: int,
    strategy: str,
    step: int,
    logger,
    n_failed_load: int = 0,
) -> tuple[dict, list[str]]:
    """Train one independent model per ticker and aggregate metrics.

    Saves per ticker inside runs/<run_id>/tickers/<T>/:
      metrics.csv, predictions.csv, plot.png, model/

    Also saves:
      runs/<run_id>/metrics.json  — mean across successful tickers
    """
    ticker_metrics_list: list[dict] = []
    failed_tickers: list[str] = []

    for r in all_results:
        ticker = r["ticker"]
        try:
            ticker_dir = tracker.ticker_dir(ticker)
            model = build_model(model_name, config, L, H, ticker_dir, strategy=strategy, step=step)
            logger.info(f"  [{ticker}] Training {type(model).__name__}...")
            model.fit(r["X_train"], r["Y_train"], r["X_val"], r["Y_val"])
            logger.info(f"  [{ticker}] Training complete.")
            if getattr(model, "history", None):
                plot_training_curves(
                    model.history,
                    save_path=tracker.ticker_dir(ticker) / "training_curves.png",
                )
                logger.info(f"  [{ticker}] Training curves saved.")

            y_pred = model.predict(r["X_test"])

            # Inverse-transform back to raw price scale before any metric, plot,
            # or saved prediction. Recursive rollout (if any) ran entirely in
            # scaled space inside the model, so this is the single point of
            # de-normalization. When kind="none" these calls are identity.
            scaler = r["scaler"]
            y_true_raw = scaler.inverse_transform(r["Y_test"])
            y_pred_raw = scaler.inverse_transform(y_pred)
            anchors_raw = scaler.inverse_transform(r["anchors_test"])

            ticker_metrics = {
                "mae": mae(y_true_raw, y_pred_raw),
                "rmse": rmse(y_true_raw, y_pred_raw),
                "mape": mape(y_true_raw, y_pred_raw),
                "smape": smape(y_true_raw, y_pred_raw),
                "directional_accuracy": directional_accuracy(
                    y_true_raw, y_pred_raw, anchors_raw
                ),
                "final_return_mae": final_return_mae(
                    y_true_raw, y_pred_raw, anchors_raw
                ),
            }

            tracker.save_ticker_metrics(ticker, ticker_metrics)
            tracker.save_ticker_predictions(
                ticker=ticker,
                dates=r["dates_test"],
                y_true=y_true_raw,
                y_pred=y_pred_raw,
                anchors=anchors_raw,
            )
            plot_ticker_forecast(
                dates=r["dates_test"],
                y_true=y_true_raw,
                y_pred=y_pred_raw,
                ticker=ticker,
                save_path=tracker.ticker_plot_path(ticker),
            )
            plot_ticker_returns(
                dates=r["dates_test"],
                y_true=y_true_raw,
                y_pred=y_pred_raw,
                anchors=anchors_raw,
                ticker=ticker,
                save_path=tracker.ticker_return_plot_path(ticker),
            )
            tracker.save_ticker_model(model, ticker)

            ticker_metrics_list.append(ticker_metrics)
            logger.info(
                f"  [{ticker}] MAE={ticker_metrics['mae']:.4f}  "
                f"RMSE={ticker_metrics['rmse']:.4f}  "
                f"MAPE={ticker_metrics['mape']:.4f}  "
                f"SMAPE={ticker_metrics['smape']:.4f}  "
                f"Dir={ticker_metrics['directional_accuracy']:.2f}%  "
                f"FinalRetMAE={ticker_metrics['final_return_mae']:.4f}pp"
            )

        except Exception as exc:
            logger.error(
                f"  [{ticker}] Failed during training/evaluation: {exc}",
                exc_info=True,
            )
            failed_tickers.append(ticker)

    if not ticker_metrics_list:
        return {}, failed_tickers

    global_metrics = _aggregate_ticker_metrics(ticker_metrics_list)
    # Raw-price MAE / RMSE are scale-dependent and not meaningful when averaged
    # across tickers of different price levels — keep them per ticker only.
    global_metrics.pop("mae", None)
    global_metrics.pop("rmse", None)
    global_metrics["n_tickers_ok"] = len(ticker_metrics_list)
    global_metrics["n_tickers_failed"] = len(failed_tickers) + n_failed_load

    tracker.save_global_metrics(global_metrics)

    return global_metrics, failed_tickers


def main():
    args = parse_args()

    model_name = args.model
    regime = args.regime
    L = args.L
    H = args.H
    strategy = args.strategy
    step = args.step
    global_scaler_kind = args.global_scaler
    data_dir = Path("data/raw")
    base_dir = Path(args.base_dir)

    if strategy == "mimo":
        step = 0

    config = load_configs(model_name, regime)

    if args.hparams:
        try:
            hparam_overrides = json.loads(args.hparams)
        except Exception as exc:
            print(f"ERROR: --hparams is not valid JSON: {exc}", file=sys.stderr)
            sys.exit(1)
        config.update(hparam_overrides)

    if args.seed is not None:
        config["seed"] = args.seed

    extra_tags: list[str] = []
    if model_name in ("patchtst", "lstm"):
        use_window_scaling = config.get("use_window_scaling", True)
        extra_tags.append("windowscaling" if use_window_scaling else "nowindowscaling")
    extra_tags.append(f"globalscaler-{global_scaler_kind}")

    run_id = make_run_id(model_name, regime, L, H, strategy=strategy, step=step, extra_tags=extra_tags or None)

    tracker = RunTracker(run_id, base_dir=str(base_dir))
    logger = get_logger("tsforecast", log_file=tracker.log_file)

    set_seed(config.get("seed", 2024))

    logger.info(
        f"Starting run {run_id} | model={model_name} regime={regime} "
        f"L={L} H={H} strategy={strategy} step={step} "
        f"global_scaler={global_scaler_kind} (raw prices)"
    )

    parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        logger.warning(f"No parquet files found in '{data_dir}'. Nothing to do — exiting.")
        sys.exit(0)

    logger.info(f"Found {len(parquet_files)} parquet file(s) in '{data_dir}'.")

    # Preprocessing
    all_results: list[dict] = []
    failed_load: list[str] = []

    for pf in parquet_files:
        ticker = pf.stem
        logger.info(f"Preprocessing ticker: {ticker}")
        result = process_ticker(
            parquet_path=pf,
            config=config,
            L=L,
            H=H,
            logger=logger,
            global_scaler_kind=global_scaler_kind,
        )
        if result is None:
            failed_load.append(ticker)
        else:
            all_results.append(result)

    if not all_results:
        logger.warning("No valid tickers could be processed. Exiting.")
        sys.exit(0)

    logger.info(f"Preprocessed {len(all_results)} ticker(s) successfully.")

    # Training and evaluation
    global_metrics, failed_train = _run_per_ticker(
        all_results, tracker, model_name, config, L, H, strategy, step, logger,
        n_failed_load=len(failed_load),
    )
    failed_tickers = failed_load + failed_train

    if not global_metrics:
        logger.warning("No tickers completed training successfully. No global metrics to save.")
        sys.exit(1)

    if failed_tickers:
        logger.warning(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")

    #Save config
    full_config = {
        **config,
        "model": model_name,
        "regime": regime,
        "L": L,
        "H": H,
        "run_id": run_id,
        "strategy": strategy,
        "step": step,
        "global_scaler": global_scaler_kind,
        "tickers_ok": global_metrics.get("n_tickers_ok", 0),
        "tickers_failed": global_metrics.get("n_tickers_failed", 0),
    }
    tracker.save_config(full_config)

    logger.info(
        f"Run {run_id} complete — "
        f"{global_metrics.get('n_tickers_ok', 0)} tickers ok, "
        f"{global_metrics.get('n_tickers_failed', 0)} failed."
    )
    logger.info(f"  Global MAPE:           {global_metrics['mape']:.4f}")
    logger.info(f"  Global SMAPE:          {global_metrics['smape']:.4f}")
    logger.info(f"  Global Dir:            {global_metrics['directional_accuracy']:.2f}%")
    logger.info(f"  Global FinalRet MAE:   {global_metrics['final_return_mae']:.4f} pp")
    logger.info(f"  Saved to: {tracker.run_dir}")


if __name__ == "__main__":
    main()
