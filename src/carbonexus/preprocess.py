"""Preprocessing steps from paper Section 2.2 on [T,H,W] monthly composites.

QA filtering, reprojection to the 0.05 deg grid and monthly median compositing are
performed upstream in the Earth Engine fetch (data/gee.py). This module implements the
remaining steps. Deviation from the paper's order: robust normalisation (2.2.7) is done
*first*, because Q=1e-4 / R=1e-2 of the Kalman filter are only meaningful on unit-scale
data (raw NO2 is ~1e-4 mol/m2).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import savgol_filter

MAD_SCALE = 1.4826


def robust_stats(x: np.ndarray) -> tuple[float, float]:
    """Median and scaled MAD over all finite values (global per channel)."""
    v = x[np.isfinite(x)]
    if v.size == 0:
        return np.nan, np.nan
    med = np.median(v)
    return float(med), float(MAD_SCALE * np.median(np.abs(v - med)))


def robust_zscore(x: np.ndarray, med: float | None = None, scale: float | None = None) -> tuple[np.ndarray, float, float]:
    """Eq. 7. Returns (normalised, median, 1.4826*MAD)."""
    if med is None or scale is None:
        med, scale = robust_stats(x)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Cannot normalise: channel has zero or undefined MAD")
    return (x - med) / scale, med, scale


def mad_clip(x: np.ndarray, k: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Eq. 1 applied to each pixel's time series (axis 0). Outliers -> pixel median.

    Returns (clipped, outlier_mask)."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
        med = np.nanmedian(x, axis=0)
        mad = MAD_SCALE * np.nanmedian(np.abs(x - med), axis=0)
        out = np.abs(x - med) > k * mad
    out &= (mad > 0)[None]
    y = np.where(out, med[None], x)
    return y, out


def kalman_fill(x: np.ndarray, q: float = 1e-4, r: float = 1e-2, max_gap: int | None = 6) -> tuple[np.ndarray, np.ndarray]:
    """Scalar random-walk Kalman filter per pixel (Eqs. 2-6), vectorised over pixels.

    Updates are skipped where the observation is NaN, so the prediction carries the state.
    Pixels start at their first valid observation (leading NaNs stay NaN). Gaps longer
    than max_gap months are left NaN. Returns (filtered, filled_mask)."""
    T = x.shape[0]
    out = np.full_like(x, np.nan, dtype=float)
    filled = np.zeros(x.shape, dtype=bool)
    state = np.full(x.shape[1:], np.nan)
    P = np.full(x.shape[1:], np.nan)
    gap = np.zeros(x.shape[1:], dtype=int)
    for t in range(T):
        z = x[t]
        obs = np.isfinite(z)
        init = obs & ~np.isfinite(state)
        state = np.where(init, z, state)
        P = np.where(init, r, P)
        started = np.isfinite(state)
        # predict (2),(3)
        P_pred = np.where(started, P + q, np.nan)
        # update (4)-(6) only where observed (and already started before this step)
        upd = obs & started & ~init
        K = P_pred / (P_pred + r)
        state = np.where(upd, state + K * (z - state), state)
        P = np.where(upd, (1 - K) * P_pred, np.where(init, P, P_pred))
        gap = np.where(obs, 0, gap + 1)
        keep = started & (obs | (gap <= (max_gap if max_gap is not None else T)))
        out[t] = np.where(keep, state, np.nan)
        filled[t] = keep & ~obs
    return out, filled


def savgol_time(x: np.ndarray, window: int = 5, order: int = 2) -> np.ndarray:
    """Savitzky-Golay along time (2.2.6), applied independently to each contiguous
    finite run of >= window samples; shorter runs are returned unchanged."""
    out = x.copy()
    T = x.shape[0]
    if T < window:
        return out
    full = np.all(np.isfinite(x), axis=0)
    if full.any():
        out[:, full] = savgol_filter(x[:, full], window, order, axis=0, mode="interp")
    ys, xs = np.where(~full & np.any(np.isfinite(x), axis=0))
    for i, j in zip(ys, xs):
        s = x[:, i, j]
        fin = np.isfinite(s)
        idx = np.flatnonzero(np.diff(np.concatenate(([0], fin.view(np.int8), [0]))))
        for a, b in zip(idx[::2], idx[1::2]):
            if b - a >= window:
                out[a:b, i, j] = savgol_filter(s[a:b], window, order, mode="interp")
    return out


def apply_fire_mask(arrs: dict[str, np.ndarray], fire_count: np.ndarray, channels=("no2", "co")) -> dict[str, np.ndarray]:
    """2.2.8: NaN the CO and NO2 channels in cell-months with an active fire detection."""
    burning = fire_count > 0
    out = dict(arrs)
    for c in channels:
        out[c] = np.where(burning, np.nan, arrs[c])
    return out


def preprocess_channel(raw: np.ndarray, *, mad_k=4.0, q=1e-4, r=1e-2, max_gap=6, window=5, order=2, normalise=True) -> dict:
    """Full per-channel chain: z-score -> MAD clip -> Kalman -> Savitzky-Golay."""
    stats = (np.nan, np.nan)
    x = raw.astype(float)
    if normalise:
        x, med, scale = robust_zscore(x)
        stats = (med, scale)
    clipped, outliers = mad_clip(x, mad_k)
    filt, filled = kalman_fill(clipped, q, r, max_gap)
    smooth = savgol_time(filt, window, order)
    return {"data": smooth, "outliers": outliers, "filled": filled, "median": stats[0], "scale": stats[1]}
