from __future__ import annotations

import numpy as np
from typing import Union

def _check_shapes(y_true: np.ndarray, y_pred: np.ndarray):
    """Validate that y_true and y_pred have matching shapes."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} do not match.")

def mae(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Mean Absolute Error: mean of |y_true - y_pred|."""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    _check_shapes(y_true, y_pred)

    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Root Mean Squared Error: sqrt(mean((y_true - y_pred)^2))."""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    _check_shapes(y_true, y_pred)
    
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (in %). Uses ``eps`` to avoid division by zero."""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    _check_shapes(y_true, y_pred)

    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], eps: float = 1e-8) -> float:
    """Symmetric MAPE (in %), bounded 0–200%. Normalises by sum of absolute values."""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    _check_shapes(y_true, y_pred)

    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)

def directional_accuracy(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    anchors: Union[np.ndarray, list],
) -> float:
    """Directional accuracy over the full horizon (0–100 %).

    For every step h in [1, H], compares sign(y_true[:, h] - anchor) against
    sign(y_pred[:, h] - anchor), both relative to the anchor (last observed
    value before the window). Accuracy is averaged over all windows and steps.
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    _check_shapes(y_true, y_pred)

    if y_true.ndim == 1:
        y_true = y_true[np.newaxis, :]
        y_pred = y_pred[np.newaxis, :]

    anchors = np.array(anchors, dtype=np.float64).ravel()[:, np.newaxis]
    dir_true = np.sign(y_true - anchors)
    dir_pred = np.sign(y_pred - anchors)

    return float(np.mean(dir_true == dir_pred) * 100.0)


def final_return_mae(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    anchors: Union[np.ndarray, list],
    eps: float = 1e-12,
) -> float:
    """Mean absolute error of the final-horizon return, in percentage points.

    For every window the final-step return is taken relative to the anchor
    (last observed price before the window):

        r_true_final = (y_true[:, -1] - anchor) / anchor
        r_pred_final = (y_pred[:, -1] - anchor) / anchor

    The metric returns ``mean(|r_pred_final - r_true_final|) * 100`` so the
    value is interpretable as "average percentage-point error on the final
    holding-period return".
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    _check_shapes(y_true, y_pred)

    if y_true.ndim == 1:
        y_true = y_true[np.newaxis, :]
        y_pred = y_pred[np.newaxis, :]

    anchors = np.array(anchors, dtype=np.float64).ravel()
    denom = np.where(np.abs(anchors) < eps, eps, anchors)
    r_true_final = (y_true[:, -1] - anchors) / denom
    r_pred_final = (y_pred[:, -1] - anchors) / denom
    return float(np.mean(np.abs(r_pred_final - r_true_final)) * 100.0)