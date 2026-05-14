from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd

def generate_windows_mimo(
        series: np.ndarray,
        dates: np.ndarray,
        start: int,
        end: int,
        context_length: int,
        horizon: int,
        stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding (context, target) window pairs from a 1-D price series.

    Uses numpy.lib.stride_tricks.sliding_window_view for vectorised extraction.
    Returns empty arrays (not an error) when the slice is too short to form any window.

    Returns (X, Y, anchors, start_dates) where:
      - X shape (n_windows, context_length), float32
      - Y shape (n_windows, horizon), float32
      - anchors shape (n_windows,) — last observed value before each window
      - start_dates shape (n_windows,) — datetime64[ns] of the first predicted step
    """
    if series.ndim != 1:
        raise ValueError("Input series must be one-dimensional.")
    if dates.ndim != 1 or len(dates) != len(series):
        raise ValueError("Dates array must be one-dimensional and the same length as series.")
    
    series_len = len(series)
    start = max(start, 0)
    end = min(end, series_len)

    # First and last valid prediction start positions
    first_pred = start + context_length
    last_pred = end - horizon + 1

    if last_pred <= first_pred or context_length <= 0 or horizon <= 0:
        return (
            np.empty((0, context_length), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    # sliding_window_view index i covers series[i : i + window_size];
    # window offset = pred_position - context_length.
    window_size = context_length + horizon
    all_windows = np.lib.stride_tricks.sliding_window_view(
        series.astype(np.float64), window_size
    )
    window_offsets = np.arange(first_pred - context_length, last_pred - context_length, stride)
    windows = all_windows[window_offsets]

    X = windows[:, :context_length].astype(np.float32)
    Y = windows[:, context_length:].astype(np.float32)

    pred_positions = window_offsets + context_length  # series index of first predicted step
    anchors = series[pred_positions - 1].astype(np.float32)
    start_dates = dates[pred_positions].astype("datetime64[ns]")

    return X, Y, anchors, start_dates

