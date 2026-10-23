import numpy as np
import pytest

from carbonexus import preprocess as pp


def test_robust_zscore_centres_and_scales():
    rng = np.random.default_rng(1)
    x = rng.normal(5, 2, 100_000)
    z, med, scale = pp.robust_zscore(x)
    assert med == pytest.approx(5, abs=0.05)
    assert scale == pytest.approx(2, abs=0.05)   # 1.4826*MAD ~ sigma for normal data
    assert np.median(z) == pytest.approx(0, abs=1e-9)


def test_robust_zscore_rejects_constant_channel():
    with pytest.raises(ValueError):
        pp.robust_zscore(np.ones(10))


def test_mad_clip_replaces_only_outlier_with_median():
    x = np.zeros((20, 1, 1)); x[:, 0, 0] = np.linspace(-1, 1, 20); x[7, 0, 0] = 100
    y, out = pp.mad_clip(x, 4.0)
    assert out[7, 0, 0] and out.sum() == 1
    assert y[7, 0, 0] == pytest.approx(np.nanmedian(x[:, 0, 0]))
    assert np.array_equal(np.delete(y, 7, 0), np.delete(x, 7, 0))


def test_kalman_single_step_matches_equations():
    # x0=1 observed at t=0 (P0=R), then z=3 at t=1.
    q, r = 1e-4, 1e-2
    x = np.array([1.0, 3.0]).reshape(2, 1, 1)
    out, filled = pp.kalman_fill(x, q, r)
    P_pred = r + q
    K = P_pred / (P_pred + r)
    assert out[1, 0, 0] == pytest.approx(1 + K * (3 - 1))
    assert not filled.any()


def test_kalman_fills_gap_with_prediction_and_respects_max_gap():
    x = np.full((10, 1, 1), 2.0)
    x[3:5] = np.nan          # 2-month gap -> filled
    x[7:] = np.nan           # 3-month trailing gap
    out, filled = pp.kalman_fill(x, max_gap=2)
    assert np.allclose(out[3:5], 2.0) and filled[3:5].all()
    assert np.isfinite(out[7:9]).all() and np.isnan(out[9, 0, 0])   # gap > 2 -> NaN again


def test_kalman_leading_nans_stay_nan_and_empty_pixels_stay_nan():
    x = np.full((6, 2, 1), np.nan); x[3:, 0, 0] = 1.0
    out, _ = pp.kalman_fill(x)
    assert np.isnan(out[:3, 0, 0]).all() and np.isfinite(out[3:, 0, 0]).all()
    assert np.isnan(out[:, 1, 0]).all()


def test_savgol_preserves_linear_trend_and_handles_nan_runs():
    t = np.arange(30, dtype=float)
    x = np.tile(t[:, None, None], (1, 2, 1)); x[10, 1, 0] = np.nan
    y = pp.savgol_time(x, 5, 2)
    assert np.allclose(y[:10, 1, 0], t[:10]) and np.allclose(y[:, 0, 0], t)
    assert np.isnan(y[10, 1, 0])


def test_fire_mask_nans_only_no2_and_co_in_burning_cells():
    a = {c: np.ones((2, 2, 2)) for c in ("no2", "co", "ch4", "so2")}
    fires = np.zeros((2, 2, 2), int); fires[1, 0, 1] = 2
    out = pp.apply_fire_mask(a, fires)
    assert np.isnan(out["no2"][1, 0, 1]) and np.isnan(out["co"][1, 0, 1])
    assert np.isfinite(out["ch4"]).all() and np.isfinite(out["so2"]).all()
    assert np.isfinite(out["no2"][0]).all()


def test_preprocess_channel_chain_runs_and_normalises():
    rng = np.random.default_rng(0)
    x = 1e-4 * (1 + 0.1 * rng.normal(size=(36, 4, 4))); x[5:7, 0, 0] = np.nan
    r = pp.preprocess_channel(x)
    assert abs(np.nanmedian(r["data"])) < 0.5 and r["filled"].sum() == 2
