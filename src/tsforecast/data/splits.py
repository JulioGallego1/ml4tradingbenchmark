from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def make_time_splits(df: pd.DataFrame,
                     test_start: str,
                     test_days: int,
                     val_days: int,
                     context_length: int,
                    ) -> Dict[str, Tuple[int, int]]:
    """Compute train/val/test integer index ranges for a time-ordered DataFrame.

    Each split includes a context_length prefix so that windows can be
    formed without running off the start of the series. Returns a dict with
    keys "train", "val", "test", each mapping to (start, end)
    for use with df.iloc[start:end].

    Raises ValueError if test_start is out of range or there is
    insufficient history before the validation set.
    """

    date_col = "Date"
    date_arr = df[date_col].to_numpy(dtype='datetime64[ns]')

    test_start_dt = np.datetime64(pd.to_datetime(test_start), "ns")
    if not (date_arr[0] <= test_start_dt <= date_arr[-1]):
        raise ValueError(
            f"test_start={test_start} is outside the available date range "
            f"[{date_arr[0]}, {date_arr[-1]}]."
        )

    test_start_idx = int(np.searchsorted(date_arr, test_start_dt))
    test_end_idx = min(test_start_idx + test_days, len(df))
    if test_end_idx < test_start_idx + test_days:
        logger.warning(
            "Requested test_days=%d but only %d days available from test_start. Trimming.",
            test_days, test_end_idx - test_start_idx,
        )

    val_end_idx = test_start_idx
    val_start_idx = val_end_idx - val_days
    if val_start_idx < context_length:
        raise ValueError(
            f"Not enough history for validation set: "
            f"val_start_idx={val_start_idx} < context_length={context_length}."
        )

    val_context_start = val_start_idx - context_length
    splits = {
        "train": (0, val_context_start),
        "val": (val_context_start, val_end_idx),
        "test": (test_start_idx - context_length, test_end_idx),
    }

    for name, (start, end) in splits.items():
        logger.info(
            "%s: idx [%d -> %d] | %s -> %s | rows=%d",
            name.upper(), start, end,
            df.iloc[start][date_col].date(), df.iloc[end - 1][date_col].date(), end - start,
        )

    return splits