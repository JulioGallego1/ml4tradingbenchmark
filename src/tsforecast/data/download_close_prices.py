import argparse
import os

import pandas as pd
import yfinance as yf


def safe_name(ticker: str) -> str:
    """Replace . and / with _ to make a ticker name safe as a filename component."""
    return ticker.replace(".", "_").replace("/", "_")

def read_tickers(tickers_file: str) -> list[str]:
    """Return tickers from a file (one per line, # lines are ignored)."""
    with open(tickers_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def download_close(ticker: str, start: str = "2015-01-01", end: str = "2026-01-01") -> pd.DataFrame:
    """Download daily close prices from Yahoo Finance. Returns ["Date", "Close"], sorted.

    Raises ValueError if yfinance returns no data.
    """
    df = yf.download(ticker, 
                     start=start, 
                     end=end, 
                     interval="1d", 
                     progress=False, 
                     multi_level_index=False,
                     auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker or date range.")

    df = df.reset_index()
    out = df[["Date", "Close"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="raise")
    out = out.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    return out

OUT_DIR = "data/raw"


def main():
    """CLI entrypoint: download close prices for each ticker and write Parquet files to data/raw."""
    p = argparse.ArgumentParser(description="Download daily Close prices and save to Parquet.")
    p.add_argument("--tickers-file", default="configs/tickers.txt", help="File with 1 ticker per line.")
    p.add_argument("--start", default="2015-01-01", help="Start date: YYYY-MM-DD")
    p.add_argument("--end", default="2025-01-01", help="End date YYYY-MM-DD")
    args = p.parse_args()

    tickers = read_tickers(args.tickers_file)
    os.makedirs(OUT_DIR, exist_ok=True)

    succeeded: list[str] = []
    failed: list[str] = []

    for t in tickers:
        path = os.path.join(OUT_DIR, f"{safe_name(t)}.parquet")
        try:
            close_df = download_close(t, args.start, args.end)
            close_df.to_parquet(path, index=False)
            print(f"[OK] {t}: {len(close_df)} rows -> {path}")
            succeeded.append(t)
        except Exception as exc:
            print(f"[ERROR] {t}: download failed – {exc}")
            failed.append(t)

    print(f"\nSummary: {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed tickers: {', '.join(failed)}")


if __name__ == "__main__":
    main()