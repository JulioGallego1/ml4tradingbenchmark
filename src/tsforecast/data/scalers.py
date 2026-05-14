"""Global per-series scalers used optionally before window generation.

The user explicitly opts-in to fitting on the *full* available series — train,
validation, and test — even though this leaks test-set information into the
preprocessing step. This is an intentional experimental knob and must NOT be
silently changed to a leakage-free variant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KINDS = ("none", "minmax", "zscore")


@dataclass
class GlobalScaler:
    """Per-series affine scaler with three modes.

    - ``none``: identity transform.
    - ``minmax``: maps ``[min, max] -> [0, 1]``.
    - ``zscore``: maps to zero-mean, unit-variance (biased std, ddof=0).
    """

    kind: str = "none"
    min_: float | None = None
    max_: float | None = None
    mean_: float | None = None
    std_: float | None = None
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Unknown scaler kind {self.kind!r}; expected one of {KINDS}.")

    def fit(self, series: np.ndarray) -> "GlobalScaler":
        if self.kind == "none":
            return self
        x = np.asarray(series, dtype=np.float64).ravel()
        if x.size == 0:
            raise ValueError("Cannot fit GlobalScaler on an empty series.")
        if self.kind == "minmax":
            self.min_ = float(x.min())
            self.max_ = float(x.max())
        elif self.kind == "zscore":
            self.mean_ = float(x.mean())
            self.std_ = float(x.std())
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        if self.kind == "none":
            return np.asarray(arr)
        x = np.asarray(arr, dtype=np.float64)
        if self.kind == "minmax":
            span = max(self.max_ - self.min_, self.eps)
            out = (x - self.min_) / span
        else:  # zscore
            scale = max(self.std_, self.eps)
            out = (x - self.mean_) / scale
        return out.astype(np.float32, copy=False)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        if self.kind == "none":
            return np.asarray(arr)
        x = np.asarray(arr, dtype=np.float64)
        if self.kind == "minmax":
            span = max(self.max_ - self.min_, self.eps)
            out = x * span + self.min_
        else:  # zscore
            scale = max(self.std_, self.eps)
            out = x * scale + self.mean_
        return out.astype(np.float32, copy=False)
