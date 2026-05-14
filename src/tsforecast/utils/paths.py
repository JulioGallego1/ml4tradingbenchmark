from __future__ import annotations

import os


def get_runs_dir(base_dir: str) -> str:
    """Return (and create) ``{base_dir}/runs``."""
    runs_root = os.path.join(base_dir, "runs")
    os.makedirs(runs_root, exist_ok=True)
    return runs_root