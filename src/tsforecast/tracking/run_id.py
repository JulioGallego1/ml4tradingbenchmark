from __future__ import annotations

import random
import time


def _now_ts() -> str:
    """Return ``{ms_epoch}{2-digit-random}`` — collision-resistant timestamp for run IDs."""
    ms = int(time.time() * 1000)
    rnd = random.randint(0, 99)
    return f"{ms}{rnd:02d}"


def make_run_id(
    model: str,
    regime: str,
    L: int,
    H: int,
    strategy: str = "mimo",
    step: int = 1,
    extra_tags: list[str] | None = None,
) -> str:
    """Build a unique run ID string encoding model, strategy, regime, L, H, and timestamp.

    MIMO format:      ``{MODEL}_mimo_{regime}_L{L}_H{H}[_tag...]_{timestamp}``
    Recursive format: ``{MODEL}_rec_step{step}_{regime}_L{L}_H{H}[_tag...]_{timestamp}``
    """
    if strategy == "recursive":
        strat_tag = f"rec_step{step}"
    else:
        strat_tag = "mimo"
    tag_suffix = ("_" + "_".join(extra_tags)) if extra_tags else ""
    return f"{model.upper()}_{strat_tag}_{regime}_L{L}_H{H}{tag_suffix}_{_now_ts()}"
