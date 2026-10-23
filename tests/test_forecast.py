import numpy as np
import pytest

from carbonexus.forecast import baseline as fb


def series(n, noise=0.0, seed=0):
    t = np.arange(n)
    return 2 + 0.05 * t + np.sin(2 * np.pi * t / 12) + noise * np.random.default_rng(seed).normal(size=n)


def test_baseline_recovers_exact_trend_plus_season():
    y = series(48)
    mean, sd = fb.SeasonalTrend().fit(y).predict(12)
    assert np.allclose(mean, series(60)[48:], atol=1e-8)
    assert np.all(sd >= 0)


def test_baseline_interval_grows_with_horizon_for_noisy_series():
    m = fb.SeasonalTrend().fit(series(72, noise=0.2))
    _, sd = m.predict(60)
    assert sd[-1] > sd[0] > 0.15


def test_baseline_handles_gaps_and_rejects_short_series():
    y = series(48); y[10:14] = np.nan
    mean, _ = fb.SeasonalTrend().fit(y).predict(3)
    assert np.isfinite(mean).all()
    with pytest.raises(ValueError):
        fb.SeasonalTrend().fit(np.arange(5.0))


def test_metrics_known_values():
    m = fb.metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert m["rmse"] == pytest.approx(np.sqrt(4 / 3)) and m["mae"] == pytest.approx(2 / 3)
    assert m["r2"] == pytest.approx(1 - 4 / 2)


def test_backtest_does_not_peek_at_holdout():
    y = series(72, noise=0.05)
    y2 = y.copy(); y2[-12:] += 100            # corrupt the holdout only
    a, b = fb.backtest_baseline({"r": y}), fb.backtest_baseline({"r": y2})
    assert b["baseline"]["rmse"] > 90 and a["baseline"]["rmse"] < 0.5
    assert fb.backtest_baseline({"r": np.arange(10.0)}).get("error")


torch = pytest.importorskip("torch")


def test_bilstm_shapes_training_and_rollout():
    from carbonexus.forecast import bilstm
    t = np.arange(72)
    rng = np.random.default_rng(0)
    feats = {r: np.column_stack([np.sin(2 * np.pi * t / 12 + i) + 0.1 * rng.normal(size=72) for _ in range(13)]).astype(np.float32)
             for i, r in enumerate("abc")}
    X, Y, _ = bilstm.make_windows(feats["a"], 24, 12, 12, max_end=48)
    assert X.shape == (13, 24, 13) and Y.shape == (13, 12)       # 13 train windows, as in the paper split
    model, rep = bilstm.train_bilstm(feats, 12, epochs=3, hidden=8)
    assert rep["train_windows"] == 39
    assert model(torch.zeros(2, 24, 13)).shape == (2, 12)
    m = bilstm.test_metrics(model, feats, 12, n_samples=3)
    assert m["n"] == 36 and np.isfinite(m["rmse"])
    mean, std = bilstm.roll_forward(model, feats["a"][-24:], 12, np.zeros((12, 13)), 0, 2, 3)
    assert mean.shape == std.shape == (24,)
