import numpy as np
import pytest

from tsforecast.data.scalers import GlobalScaler


def _series(seed=0, n=500):
    rng = np.random.default_rng(seed)
    return (100.0 + np.cumsum(rng.standard_normal(n))).astype(np.float32)


def test_none_is_identity():
    s = _series()
    sc = GlobalScaler(kind="none").fit(s)
    np.testing.assert_array_equal(sc.transform(s), s)
    np.testing.assert_array_equal(sc.inverse_transform(s), s)


def test_minmax_roundtrip_and_range():
    s = _series(1)
    sc = GlobalScaler(kind="minmax").fit(s)
    z = sc.transform(s)
    assert z.min() >= 0.0 - 1e-6 and z.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(sc.inverse_transform(z), s, atol=1e-4)


def test_zscore_roundtrip_and_stats():
    s = _series(2)
    sc = GlobalScaler(kind="zscore").fit(s)
    z = sc.transform(s)
    assert abs(float(z.mean())) < 1e-5
    assert abs(float(z.std(ddof=0)) - 1.0) < 1e-4
    np.testing.assert_allclose(sc.inverse_transform(z), s, atol=1e-4)


def test_inverse_transform_handles_2d_arrays():
    s = _series(3)
    sc = GlobalScaler(kind="zscore").fit(s)
    arr2d = np.stack([s[:10], s[10:20]])  # (2, 10)
    z = sc.transform(arr2d)
    back = sc.inverse_transform(z)
    np.testing.assert_allclose(back, arr2d, atol=1e-4)


def test_constant_series_does_not_explode():
    s = np.full(100, 7.0, dtype=np.float32)
    for kind in ("minmax", "zscore"):
        sc = GlobalScaler(kind=kind).fit(s)
        z = sc.transform(s)
        assert np.isfinite(z).all()
        back = sc.inverse_transform(z)
        np.testing.assert_allclose(back, s, atol=1e-4)


def test_rejects_unknown_kind():
    with pytest.raises(ValueError):
        GlobalScaler(kind="robust")
