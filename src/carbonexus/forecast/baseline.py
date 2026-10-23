"""Baseline forecaster: OLS linear trend + annual harmonics (2 pairs).

This is a *baseline*, NOT the paper's BiLSTM (see bilstm.py for that). Prediction
intervals are analytic OLS prediction intervals (1 sigma); they assume i.i.d. Gaussian
residuals and are not validated beyond the 12-month backtest horizon.
"""
from __future__ import annotations

import numpy as np


def _design(t: np.ndarray, n_harm: int = 2) -> np.ndarray:
    cols = [np.ones_like(t, dtype=float), t.astype(float)]
    for k in range(1, n_harm + 1):
        w = 2 * np.pi * k * t / 12.0
        cols += [np.sin(w), np.cos(w)]
    return np.column_stack(cols)


class SeasonalTrend:
    def __init__(self, n_harm: int = 2):
        self.n_harm = n_harm

    def fit(self, y: np.ndarray, month0: int = 0):
        """y: monthly series (NaNs allowed). month0: calendar month index (0=Jan) of y[0]."""
        y = np.asarray(y, float)
        self.n = len(y)
        self.month0 = month0
        t = np.arange(self.n) + month0
        ok = np.isfinite(y)
        if ok.sum() < 2 * self.n_harm + 4:
            raise ValueError("series too short / too many gaps to fit the baseline")
        X = _design(t[ok], self.n_harm)
        self.beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        res = y[ok] - X @ self.beta
        dof = max(ok.sum() - X.shape[1], 1)
        self.sigma = float(np.sqrt((res ** 2).sum() / dof))
        self.XtXinv = np.linalg.pinv(X.T @ X)
        return self

    def predict(self, h: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, 1-sigma prediction std) for the next h months."""
        t = np.arange(self.n, self.n + h) + self.month0
        X = _design(t, self.n_harm)
        mean = X @ self.beta
        var = self.sigma ** 2 * (1 + np.einsum("ij,jk,ik->i", X, self.XtXinv, X))
        return mean, np.sqrt(var)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.size < 2:
        return {"n": int(y.size)}
    err = p - y
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"n": int(y.size), "rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean()),
            "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else None}


def backtest_baseline(series: dict[str, np.ndarray], holdout: int = 12) -> dict:
    """Train on all but the last `holdout` months for each region, score the holdout.
    Also scores a seasonal-naive reference (value 12 months earlier)."""
    ys, ps, ns = [], [], []
    for name, y in series.items():
        y = np.asarray(y, float)
        if len(y) < holdout + 24:
            continue
        m = SeasonalTrend().fit(y[:-holdout])
        mean, _ = m.predict(holdout)
        ys.append(y[-holdout:]); ps.append(mean); ns.append(y[-holdout - 12:-12])
    if not ys:
        return {"error": "series too short for backtest"}
    Y, P, N = np.concatenate(ys), np.concatenate(ps), np.concatenate(ns)
    return {"model": "SeasonalTrend (baseline)", "holdout_months": holdout, "regions": len(ys),
            "baseline": metrics(Y, P), "seasonal_naive": metrics(Y, N)}


def forecast_summary(y: np.ndarray, h: int, month0: int = 0) -> dict:
    m = SeasonalTrend().fit(y, month0)
    mean, sd = m.predict(h)
    return {"mean": mean.tolist(), "std": sd.tolist(), "model": "SeasonalTrend (baseline)"}
