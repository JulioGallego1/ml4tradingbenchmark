from __future__ import annotations
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

def read_file(file_path: str) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame. Raises on missing file or unsupported extension."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path) # Get file extension
    ext = ext.lower()

    if ext == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Supported extension is .parquet.")
    
def load_price_data(path: str) -> pd.DataFrame:
    """Load a price series from a pipeline Parquet file, returning only ["Date", "Close"].

    Raises ValueError if either column is missing.
    """
    df = read_file(path)

    missing = {"Date", "Close"} - set(df.columns)
    
    if missing:
        raise ValueError(f"Missing expected columns {missing} in {path}.")

    logger.debug(f"Loaded {len(df)} records from {path}.")
    return df[["Date", "Close"]]