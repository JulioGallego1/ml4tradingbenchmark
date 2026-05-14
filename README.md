# ml4trading

ML pipeline for stock price forecasting. Benchmarks **Random Forest**, **LSTM**, and **PatchTST** across market regimes, forecasting strategies, horizons, and context lengths. Runs locally or on a Slurm cluster (VRAIN).

## Table of Contents

- [Project Overview](#project-overview)
- [Experiment Design](#experiment-design)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Run Outputs](#run-outputs)
- [HPC / Slurm](#hpc--slurm)
- [Tests](#tests)
- [Aggregating Results](#aggregating-results)

---

## Project Overview

Each experiment is an independent, reproducible run that:

1. Loads daily close prices from pre-downloaded Parquet files.
2. Builds time-ordered train / val / test splits per ticker.
3. Generates sliding windows on raw prices (context length L → forecast horizon H).
4. Trains a separate model **per ticker**.
5. Evaluates on the held-out test set. Per-ticker metrics: MAE, RMSE, MAPE, SMAPE, directional accuracy, and final-return MAE (percentage points). Global metrics: MAPE, SMAPE, directional accuracy, and final-return MAE only — MAE and RMSE are scale-dependent in raw prices and are intentionally not aggregated across tickers.
6. Saves config, per-ticker metrics, predictions, plots, model checkpoints, and a global aggregate to `runs/<run_id>/`.

Sweeps are generated from `configs/sweep/grid.yaml` as a Cartesian product of regimes × horizons × context lengths × strategies × per-model hyperparameters. Each Slurm task receives its parameters as a JSON config line and launches a single training run.

---

## Experiment Design

| Dimension | Values |
|---|---|
| **Market regime** | `bear` (test from 2022-03-11), `bull` (test from 2023-01-06) |
| **Horizon H** | 10 (≈2w), 21 (≈1m), 63 (≈3m) trading days |
| **Context length L** | 32, 48, 96 trading days |
| **Forecasting strategy** | `mimo` (direct multi-output), `recursive` (iterative with block step) |
| **Model** | `rf`, `lstm`, `patchtst` |

**MIMO** — the model maps the L-step context directly to an H-step forecast vector in a single shot.

**Recursive** — the model is trained to predict `--step` steps at a time. At inference, predictions are appended back into the context and the model is called repeatedly until H steps have been produced.

### Run ID Format

| Strategy | Format |
|---|---|
| MIMO | `{MODEL}_mimo_{regime}_L{L}_H{H}[_tag]_{timestamp}` |
| Recursive | `{MODEL}_rec_step{step}_{regime}_L{L}_H{H}[_tag]_{timestamp}` |

For LSTM and PatchTST, an extra tag `windowscaling` or `nowindowscaling` is appended before the timestamp depending on whether per-window scaling is enabled (RevIN-style for PatchTST, per-window standardization for LSTM). A `globalscaler-{none|minmax|zscore}` tag follows. The timestamp is millisecond-precision plus a 2-digit random suffix to avoid collisions in Slurm array jobs.

---

## Project Structure

```
ml4trading/
├── configs/
│   ├── tickers.txt             # one ticker per line (20 European equities by default)
│   ├── splits.yaml             # regime definitions (test_start, val_days, test_days)
│   ├── train.yaml              # shared training settings (epochs, batch size, patience, seed)
│   ├── model/
│   │   ├── rf.yaml
│   │   ├── lstm.yaml
│   │   └── patchtst.yaml
│   └── sweep/
│       └── grid.yaml           # sweep axes and per-model hyperparameter grids
├── scripts/
│   ├── make_grid.py            # generates scripts/grid.jsonl from grid.yaml
│   ├── run_sweep.py            # one-command launcher (Slurm or local)
│   ├── run_config.py           # runs one experiment from a JSON config string
│   └── slurm/
│       ├── run_one.sh          # submit or run a single experiment
│       └── run_array.sh        # Slurm array job: one task per line of grid.jsonl
├── src/tsforecast/
│   ├── cli/train.py            # main entrypoint (also exposed as `tsforecast-train`)
│   ├── data/                   # loaders, splits, windows, downloader
│   ├── models/                 # base (ABC), rf, lstm, patchtst
│   ├── training/               # PyTorch fit loop, callbacks, reproducibility
│   ├── evaluation/             # metrics, plots
│   ├── tracking/               # RunTracker (atomic writes), run_id helpers
│   └── utils/                  # paths, logging
├── tests/
├── analyze_results.py          # best-of-family analysis across runs
├── data/
│   └── raw/                    # downloaded .parquet files (not versioned)
└── runs/                       # one directory per run (not versioned)
```

---

## Setup

### 1. Clone and create the environment

```bash
git clone <repo-url>
cd ml4trading
conda env create -f environment.yml
conda activate tsforecast-env
```

### 2. Install the package

```bash
pip install -e ".[dev]"
```

### 3. Verify

```bash
python -c "import tsforecast; print('OK')"
pytest tests/ -m "not slow"
```

---

## Usage

### Step 1 — Add tickers

Edit `configs/tickers.txt` (one ticker per line, lines starting with `#` are ignored):

```
AIR.PA
ASML
BMW.DE
```

### Step 2 — Download price data

```bash
# Use tickers from configs/tickers.txt and the default date range (2015-01-01 → 2025-01-01)
python -m tsforecast.data.download_close_prices

# Custom file or date range
python -m tsforecast.data.download_close_prices \
  --tickers-file configs/tickers.txt \
  --start 2016-01-01 --end 2024-12-31
```

Files are saved as `data/raw/<safe_ticker>.parquet` (dots and slashes in the ticker symbol are replaced with underscores).

### Step 3 — Run a single experiment

```bash
# MIMO strategy (default)
python -m tsforecast.cli.train --model rf --regime bear --L 96 --H 21

# Recursive strategy
python -m tsforecast.cli.train --model lstm --regime bear --L 96 --H 63 \
  --strategy recursive --step 16

# Override hyperparameters
python -m tsforecast.cli.train --model rf --regime bear --L 96 --H 21 \
  --hparams '{"n_estimators": 200, "max_depth": 10}'
```

The package also installs a console entry point: `tsforecast-train --model rf --regime bear --L 96 --H 21`.

**CLI flags:**

| Flag | Required | Default | Description |
|---|---|---|---|
| `--model` | Yes | — | `rf`, `lstm`, or `patchtst` |
| `--regime` | Yes | — | `bear` or `bull` |
| `--L` | Yes | — | Context length in trading days |
| `--H` | Yes | — | Forecast horizon in trading days |
| `--strategy` | No | `mimo` | `mimo` or `recursive` |
| `--step` | No | `1` | Block size for recursive strategy (forced to `0` when `strategy=mimo`) |
| `--seed` | No | from yaml | Random seed override |
| `--hparams` | No | — | JSON string of hyperparameter overrides |
| `--base-dir` | No | `.` | Project root for `runs/` output |

### Step 4 — Inspect results

```
runs/RF_mimo_bear_L96_H21_<timestamp>/
├── config.yaml
├── metrics.json
├── metrics_global.csv
├── logs.txt
└── tickers/<TICKER>/
    ├── plot.png                # forecast windows over the test span
    ├── plot_returns.png        # horizon-end returns: real vs predicted
    ├── training_curves.png     # LSTM and PatchTST only
    ├── predictions.csv
    ├── metrics.csv
    └── model/                  # serialized checkpoint (joblib for RF, state_dict for torch models)
```

---

## Configuration

YAML configs are merged in this order at runtime: `train.yaml` → `model/<model>.yaml` → regime block from `splits.yaml` → JSON overrides from `--hparams` → `--seed`. Later sources win.

### `configs/train.yaml`

Shared training settings:

```yaml
max_epochs: 250
batch_size: 32
patience: 25
seed: 2024
```

### `configs/splits.yaml`

Date boundaries for each regime. `test_start` is the first test day; `val_days` and `test_days` are counted in trading days backward / forward from there.

```yaml
regimes:
  bear:
    test_start: "2022-03-11"
    val_days: 456
    test_days: 252
  bull:
    test_start: "2023-01-06"
    val_days: 504
    test_days: 126
```

### `configs/model/*.yaml`

Per-model defaults. Examples:

```yaml
# rf.yaml
model: rf
n_estimators: 180
max_features: sqrt
max_depth: null
min_samples_leaf: 1

# lstm.yaml
model: lstm
hidden_size: 128
num_layers: 2
dropout: 0.1
lr: 0.001
patience: 25

# patchtst.yaml
model: patchtst
patch_length: 8
patch_stride: 4
d_model: 64
num_attention_heads: 4
num_hidden_layers: 3
ffn_dim: 256
dropout: 0.25
norm_type: layernorm
head_dropout: 0.1
attention_dropout: 0.1
ff_dropout: 0.2
lr: 0.001
patience: 25
use_window_scaling: true
```

### `configs/sweep/grid.yaml`

Sweep axes and per-model hyperparameter grids. `make_grid.py` produces the Cartesian product over all listed values.

```yaml
regimes: [bear, bull]
horizons: [10, 21, 63]
context_lengths: [32, 48, 96]
strategies: [mimo]            # add `recursive` to also sweep recursive runs
steps: [1]                    # only consumed when strategy = recursive
training_mode: [per_ticker]
seeds: [2024]
global_scalers: [none]        # none | minmax | zscore (intentional leakage knob)

models:
  rf:
    n_estimators: [200]
    max_depth: [null, 10]
    max_features: ["sqrt"]
    min_samples_leaf: [1, 2]
  lstm:
    hidden_size: [64, 128]
    num_layers: [1, 2]
    dropout: [0.1]
    lr: [0.001]
    batch_size: [32]
    use_window_scaling: [false, true]
  patchtst:
    d_model: [64, 128]
    num_attention_heads: [4, 8]
    num_hidden_layers: [2, 3]
    patch_length: [16]
    patch_stride: [8]
    dropout: [0.2]
    lr: [0.0015, 0.0003]
    use_window_scaling: [true]
```

For each (regime, H, L, strategy) tuple, MIMO rows have `step=0` and recursive rows expand to one row per value in `steps`.

### `--hparams` overrides

Any key passed as JSON via `--hparams` overrides the merged YAML value. `null` becomes Python `None`:

```bash
python -m tsforecast.cli.train --model rf --regime bear --L 96 --H 21 \
  --hparams '{"n_estimators": 200, "max_depth": null}'
```

---

## Run Outputs

All artifacts are written under `runs/<run_id>/`. Writes are atomic (write-to-`.tmp` + `os.replace`).

| Path | Contents |
|---|---|
| `config.yaml` | Full merged config used for the run |
| `metrics.json` | Global metrics: `mape`, `smape`, `directional_accuracy`, `final_return_mae`, `n_tickers_ok`, `n_tickers_failed`. Raw-price `mae` / `rmse` are deliberately omitted globally (scale-dependent across tickers). |
| `metrics_global.csv` | Same global metrics as a single-row CSV |
| `logs.txt` | Training log |
| `tickers/<T>/metrics.csv` | Per-ticker metrics: `mae`, `rmse`, `mape`, `smape`, `directional_accuracy`, `final_return_mae` |
| `tickers/<T>/predictions.csv` | Columns: `date`, `ticker`, `anchor`, `y_true_0..H-1`, `y_pred_0..H-1` |
| `tickers/<T>/plot.png` | Forecast windows over the test span |
| `tickers/<T>/plot_returns.png` | Horizon-end return per window: real vs predicted |
| `tickers/<T>/training_curves.png` | Train/val loss curves (LSTM and PatchTST only) |
| `tickers/<T>/model/` | Serialized model: `model.joblib` for RF, `model_state.pt` + `config.json` for LSTM, HuggingFace `save_pretrained` files + `hparams.json` for PatchTST |

Global metrics are arithmetic means of the per-ticker metrics over tickers that completed successfully.

---

## HPC / Slurm

### Scripts overview

| Script | When to use |
|---|---|
| `scripts/run_sweep.py` | Run all experiments for one model (recommended) |
| `scripts/slurm/run_one.sh` | Submit or run a single hand-crafted experiment |
| `scripts/slurm/run_array.sh` | Called by Slurm automatically — do not invoke directly |
| `scripts/run_config.py` | Run a single JSON config locally (used internally by the array script) |
| `scripts/make_grid.py` | Regenerate the full `grid.jsonl` manually |

### Full model sweep (recommended)

```bash
# 1. Edit the sweep config
vim configs/sweep/grid.yaml

# 2. Preview what will run
python scripts/run_sweep.py --model rf --dry-run

# 3. Submit to the cluster
python scripts/run_sweep.py --model rf
python scripts/run_sweep.py --model lstm
python scripts/run_sweep.py --model patchtst
```

`run_sweep.py` writes `scripts/grid_<model>.jsonl` and submits a Slurm array job (`run_array.sh`) where each task picks the line at `SLURM_ARRAY_TASK_ID` and forwards it through `run_config.py → tsforecast.cli.train`.

### Single experiment

```bash
# Submit to Slurm
sbatch scripts/slurm/run_one.sh rf bear 96 21
sbatch scripts/slurm/run_one.sh lstm bull 48 63 --strategy recursive --step 16

# Run locally (without sbatch)
bash scripts/slurm/run_one.sh rf bear 96 21
```

### Rerun specific failed tasks

```bash
sbatch --array=3,7 \
  --export=ALL,GRID_FILE=scripts/grid_rf.jsonl \
  scripts/slurm/run_array.sh
```

### Run locally (no cluster)

```bash
# Sequential sweep
python scripts/run_sweep.py --model rf --local

# Single config debug
python scripts/run_config.py '{"model":"rf","regime":"bear","L":96,"H":21,"strategy":"mimo","step":0,"seed":2024,"n_estimators":200,"max_depth":null}'
```

### Monitor jobs

```bash
squeue -u $USER
tail -f logs/slurm_<jobid>_<task>.out
```

> **Slurm prerequisites.** The submission scripts source `~/miniconda3/etc/profile.d/conda.sh` and activate `tsforecast-env`. The conda environment must exist with that name and the package must be installed inside it (`pip install -e .`). Each task requests 16 GB of memory, 4 CPUs, and 1 GPU by default — adjust the `#SBATCH` headers in `scripts/slurm/run_*.sh` to match your cluster.

---

## Tests

```bash
# Fast tests
pytest tests/ -m "not slow"

# All tests including the PatchTST smoke test
pytest tests/
```

| Test file | What it covers |
|---|---|
| `test_windows.py` | Window shapes, no overlap between X and Y, anchor values, `float32` dtype, stride |
| `test_splits.py` | Train/val/test boundaries, context-prefix preservation, error on insufficient history |
| `test_metrics.py` | MAE / RMSE / MAPE / SMAPE / directional accuracy / final-return MAE against known inputs |
| `test_models_smoke.py` | Fit + predict + save + load for RF, LSTM, PatchTST; `_LSTMNet` forward pass |
| `test_recursive.py` | Recursive output shapes, no future leakage, MIMO vs recursive equivalence on save/load |

---

## Aggregating Results

`analyze_results.py` performs a "best-of-family" analysis across all runs in `runs/`. For each `(regime, horizon)`, it picks the single best run per model family — independently for MAPE (lower is better) and directional accuracy (higher is better) — and produces comparison plots, summary tables, and symlinks pointing to the chosen run directories.

```bash
python analyze_results.py
python analyze_results.py --runs-dir runs --out-dir analysis
```

The script writes its output under `analysis/`:

```
analysis/
├── best_of_family/
│   ├── <regime>_H<H>/
│   │   ├── best_by_mape.csv
│   │   ├── best_by_da.csv
│   │   ├── mape_comparison.png
│   │   ├── da_comparison.png
│   │   ├── table_best_mape.png
│   │   └── table_best_da.png
│   └── overview/
│       ├── heatmap_mape.png
│       ├── heatmap_da.png
│       ├── best_by_mape_all.csv
│       └── best_by_da_all.csv
├── lines/                    # MAPE / DA line plots across horizons
├── best_selected/            # symlinks to the chosen run directories
├── all_runs_detail.csv
└── summary_report.txt
```

The model families compared are `RF`, `LSTM_mimo`, `LSTM_recursive`, `PATCHTST_mimo`, and `PATCHTST_recursive`. Runs with missing or unparseable `metrics.json` / `config.yaml` are skipped with a warning.
