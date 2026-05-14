"""Run-id tags for model-specific normalization options."""
from tsforecast.tracking.run_id import make_run_id


def test_make_run_id_includes_extra_tags():
    """extra_tags appear as ``_tag`` segments before the timestamp."""
    rid = make_run_id("lstm", "regimeA", 32, 8, extra_tags=["windowscaling"])
    assert "_windowscaling_" in rid


def test_lstm_window_scaling_distinguishes_run_ids():
    """Two LSTM runs with identical hparams but different scaling settings must differ."""
    rid_on = make_run_id("lstm", "regimeA", 32, 8, extra_tags=["windowscaling"])
    rid_off = make_run_id("lstm", "regimeA", 32, 8, extra_tags=["nowindowscaling"])
    on_core, off_core = rid_on.rsplit("_", 1)[0], rid_off.rsplit("_", 1)[0]
    assert on_core != off_core
    assert on_core.endswith("_windowscaling")
    assert off_core.endswith("_nowindowscaling")


def test_lstm_and_patchtst_share_window_scaling_tag():
    """LSTM and PatchTST use the same windowscaling/nowindowscaling tag convention."""
    lstm_on = make_run_id("lstm", "r", 16, 4, extra_tags=["windowscaling"])
    lstm_off = make_run_id("lstm", "r", 16, 4, extra_tags=["nowindowscaling"])
    patchtst_on = make_run_id("patchtst", "r", 16, 4, extra_tags=["windowscaling"])
    patchtst_off = make_run_id("patchtst", "r", 16, 4, extra_tags=["nowindowscaling"])
    for rid, tag in [
        (lstm_on, "windowscaling"), (lstm_off, "nowindowscaling"),
        (patchtst_on, "windowscaling"), (patchtst_off, "nowindowscaling"),
    ]:
        assert f"_{tag}_" in rid


def test_global_scaler_tag_distinguishes_runs():
    """Two runs identical except for global scaler must produce distinct IDs."""
    rid_none = make_run_id("rf", "r", 16, 4, extra_tags=["globalscaler-none"])
    rid_mm = make_run_id("rf", "r", 16, 4, extra_tags=["globalscaler-minmax"])
    rid_z = make_run_id("rf", "r", 16, 4, extra_tags=["globalscaler-zscore"])
    cores = {rid.rsplit("_", 1)[0] for rid in (rid_none, rid_mm, rid_z)}
    assert len(cores) == 3
    assert "_globalscaler-none" in rid_none
    assert "_globalscaler-minmax" in rid_mm
    assert "_globalscaler-zscore" in rid_z


def test_combined_window_and_global_scaler_tags_coexist():
    """Per-window and global-scaler tags should both appear on the same run ID."""
    rid = make_run_id(
        "lstm", "r", 16, 4,
        extra_tags=["windowscaling", "globalscaler-zscore"],
    )
    assert "_windowscaling_" in rid
    assert "_globalscaler-zscore_" in rid
