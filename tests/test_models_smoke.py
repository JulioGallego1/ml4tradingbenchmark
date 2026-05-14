import numpy as np
import pytest

SEED = 42
N_TRAIN = 80
N_VAL = 20
N_TEST = 10
L = 16
H = 4


def make_data(n, L, H, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.random((n, L)).astype(np.float32)
    Y = rng.random((n, H)).astype(np.float32)
    return X, Y


def test_rf_smoke():
    from tsforecast.models.rf import RandomForestModel
    X_tr, Y_tr = make_data(N_TRAIN, L, H)
    X_val, Y_val = make_data(N_VAL, L, H)
    X_te, _ = make_data(N_TEST, L, H)
    model = RandomForestModel(n_estimators=10, random_state=SEED)
    model.fit(X_tr, Y_tr, X_val, Y_val)
    preds = model.predict(X_te)
    assert preds.shape == (N_TEST, H)
    assert np.isfinite(preds).all()


def test_rf_save_load(tmp_path):
    from tsforecast.models.rf import RandomForestModel
    X_tr, Y_tr = make_data(N_TRAIN, L, H)
    model = RandomForestModel(n_estimators=5, random_state=SEED)
    model.fit(X_tr, Y_tr)
    model.save(tmp_path / "rf_model")
    loaded = RandomForestModel.load(tmp_path / "rf_model")
    X_te, _ = make_data(N_TEST, L, H)
    np.testing.assert_array_almost_equal(model.predict(X_te), loaded.predict(X_te))


def test_lstm_smoke():
    from tsforecast.models.lstm import LSTMModel
    X_tr, Y_tr = make_data(N_TRAIN, L, H)
    X_val, Y_val = make_data(N_VAL, L, H)
    X_te, _ = make_data(N_TEST, L, H)
    model = LSTMModel(
        context_length=L, horizon=H,
        hidden_size=16, num_layers=1, dropout=0.0,
        lr=1e-2, max_epochs=2, batch_size=16, patience=5,
        random_state=SEED,
    )
    model.fit(X_tr, Y_tr, X_val, Y_val)
    preds = model.predict(X_te)
    assert preds.shape == (N_TEST, H)
    assert np.isfinite(preds).all()


def test_lstm_save_load(tmp_path):
    from tsforecast.models.lstm import LSTMModel
    X_tr, Y_tr = make_data(N_TRAIN, L, H)
    X_val, Y_val = make_data(N_VAL, L, H)
    model = LSTMModel(
        context_length=L, horizon=H,
        hidden_size=16, num_layers=1, dropout=0.0,
        lr=1e-2, max_epochs=2, batch_size=16, patience=5,
        random_state=SEED,
    )
    model.fit(X_tr, Y_tr, X_val, Y_val)
    model.save(tmp_path / "lstm_model")
    loaded = LSTMModel.load(tmp_path / "lstm_model")
    X_te, _ = make_data(N_TEST, L, H)
    np.testing.assert_array_almost_equal(model.predict(X_te), loaded.predict(X_te), decimal=5)


def test_lstmnet_forward():
    """_LSTMNet constructor and forward pass with the current 4-arg signature."""
    import torch
    from tsforecast.models.lstm import _LSTMNet

    net = _LSTMNet(output_horizon=4, hidden_size=16, num_layers=1, dropout=0.0)
    x = torch.randn(3, L)       # (B, context_length)
    out = net(x)                # expected (B, H, 1)
    assert out.shape == (3, 4, 1)
    assert torch.isfinite(out).all()


def test_lstmnet_window_scaling_no_learnable_params():
    """use_window_scaling must not add learnable affine parameters."""
    import torch
    from tsforecast.models.lstm import _LSTMNet

    net_on = _LSTMNet(output_horizon=4, hidden_size=16, num_layers=1, dropout=0.0,
                      use_window_scaling=True)
    net_off = _LSTMNet(output_horizon=4, hidden_size=16, num_layers=1, dropout=0.0,
                       use_window_scaling=False)
    names = [n for n, _ in net_on.named_parameters()]
    assert not any("revin" in n.lower() or "affine" in n.lower() for n in names)
    # Param count must be identical regardless of the flag.
    assert sum(p.numel() for p in net_on.parameters()) == sum(p.numel() for p in net_off.parameters())


def test_lstmnet_window_scaling_is_affine_equivariant():
    """With scaling on, pred(a*x+b) == a*pred(x)+b — pure per-window standardization."""
    import torch
    from tsforecast.models.lstm import _LSTMNet

    torch.manual_seed(0)
    net = _LSTMNet(output_horizon=3, hidden_size=8, num_layers=1, dropout=0.0,
                   use_window_scaling=True).eval()
    x = torch.randn(4, 12) + 100.0
    a, b = 3.0, 5.0
    with torch.no_grad():
        p1 = net(x)
        p2 = net(a * x + b)
    torch.testing.assert_close(p2, a * p1 + b, rtol=1e-4, atol=1e-4)


def test_lstmnet_no_scaling_passes_raw_inputs():
    """With scaling off, the input scale propagates: pred(a*x) != a*pred(x) in general."""
    import torch
    from tsforecast.models.lstm import _LSTMNet

    torch.manual_seed(0)
    net = _LSTMNet(output_horizon=3, hidden_size=8, num_layers=1, dropout=0.0,
                   use_window_scaling=False).eval()
    x = torch.randn(4, 12)
    with torch.no_grad():
        p1 = net(x)
        p2 = net(100.0 * x)
    assert not torch.allclose(p2, 100.0 * p1, rtol=1e-3, atol=1e-3)


def test_lstmnet_predictions_denormalized_to_raw_scale():
    """When the head outputs ~0 in normalized space, denorm pred ≈ window mean (raw scale)."""
    import torch
    from tsforecast.models.lstm import _LSTMNet

    torch.manual_seed(0)
    net = _LSTMNet(output_horizon=4, hidden_size=16, num_layers=1, dropout=0.0,
                   use_window_scaling=True).eval()
    net.head.weight.data.zero_()
    net.head.bias.data.zero_()
    x = torch.randn(4, 12) * 5.0 + 100.0
    with torch.no_grad():
        pred = net(x).squeeze(-1)
    expected = x.mean(dim=1, keepdim=True).expand_as(pred)
    torch.testing.assert_close(pred, expected, rtol=1e-4, atol=1e-4)


def test_lstmnet_forward_uses_only_context():
    """Forward signature must accept only x — no target leakage is possible."""
    import inspect
    from tsforecast.models.lstm import _LSTMNet

    assert list(inspect.signature(_LSTMNet.forward).parameters) == ["self", "x"]


def test_lstm_save_load_preserves_window_scaling_flag(tmp_path):
    from tsforecast.models.lstm import LSTMModel
    X_tr, Y_tr = make_data(N_TRAIN, L, H)
    X_val, Y_val = make_data(N_VAL, L, H)
    for flag in (True, False):
        model = LSTMModel(
            context_length=L, horizon=H,
            hidden_size=8, num_layers=1, dropout=0.0,
            lr=1e-2, max_epochs=1, batch_size=16, patience=5,
            random_state=SEED, use_window_scaling=flag,
        )
        model.fit(X_tr, Y_tr, X_val, Y_val)
        out = tmp_path / f"lstm_ws_{flag}"
        model.save(out)
        loaded = LSTMModel.load(out)
        assert loaded.use_window_scaling is flag
        assert loaded._net.use_window_scaling is flag


# PatchTST smoke test is marked slow -- requires transformers library
@pytest.mark.slow
def test_patchtst_smoke(tmp_path):
    from tsforecast.models.patchtst import PatchTSTModel
    X_tr, Y_tr = make_data(N_TRAIN, L, H)
    X_val, Y_val = make_data(N_VAL, L, H)
    X_te, _ = make_data(N_TEST, L, H)
    model = PatchTSTModel(
        context_length=L, horizon=H,
        patch_length=4, patch_stride=2,
        d_model=16, num_attention_heads=2, num_hidden_layers=1,
        ffn_dim=32, dropout=0.0,
        lr=1e-3, max_epochs=1, batch_size=16, patience=5,
        random_state=SEED,
        output_dir=str(tmp_path / "patchtst_tmp"),
    )
    model.fit(X_tr, Y_tr, X_val, Y_val)
    preds = model.predict(X_te)
    assert preds.shape == (N_TEST, H)
    assert np.isfinite(preds).all()
