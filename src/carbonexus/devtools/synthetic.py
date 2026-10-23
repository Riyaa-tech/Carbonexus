"""SYNTHETIC data generator for tests and UI smoke-testing ONLY.

Everything produced here is artificial: smooth Gaussian "sources" placed near the paper's
named regions, plus seasonal cycle, trend, noise, cloud-like gaps and one fake fire.
It exists so the pipeline/API/dashboard can be exercised without credentials. Any analysis
built from it is tagged provenance="SYNTHETIC" and the dashboard shows a banner.
It must never be presented as a measurement or used to report results.
"""
from __future__ import annotations

import numpy as np

from ..config import CHANNELS, Config

PROVENANCE = "SYNTHETIC (not real data)"


def _blob(lats, lons, lat0, lon0, s):
    lon2, lat2 = np.meshgrid(lons, lats)
    return np.exp(-(((lat2 - lat0) ** 2) + ((lon2 - lon0) ** 2)) / (2 * s ** 2))


def make_raw(cfg: Config, seed: int = 0):
    rng = np.random.default_rng(seed)
    lats, lons = cfg.lats, cfg.lons
    T, H, W = len(cfg.months), len(lats), len(lons)
    t = np.arange(T)
    season = np.cos(2 * np.pi * (t % 12) / 12)[:, None, None]       # winter peak
    trend = (1 - 0.1 * t / T)[:, None, None]
    la, bay, sjv = _blob(lats, lons, 34.0, -118.2, 0.5), _blob(lats, lons, 37.8, -122.2, 0.4), _blob(lats, lons, 36.2, -119.8, 0.9)
    base = lambda *f: sum(f)

    def gas(field, noise, level, scale):
        a = level + scale * field[None] * trend * (1 + 0.2 * season) + noise * rng.normal(size=(T, H, W))
        return a

    raw = {
        "no2": gas(base(la * 1.0, bay * 0.5, sjv * 0.2), 2e-6, 5e-6, 1e-4),
        "ch4": gas(base(la * 0.3, sjv * 1.0, bay * 0.2), 2.0, 1850.0, 60.0),
        "so2": gas(base(la * 0.3, bay * 0.4), 1e-5, 5e-5, 3e-4),
        "co": gas(base(la * 1.0, bay * 0.5), 1e-4, 0.03, 0.02),
        "u10": rng.normal(1, 1, (T, H, W)), "v10": rng.normal(0, 1, (T, H, W)),
        "t2m": 288 + 10 * -season + rng.normal(0, 1, (T, H, W)),
        "tp": np.abs(rng.normal(0.05, 0.02, (T, H, W))), "sp": 1e5 + rng.normal(0, 300, (T, H, W)),
        "built": np.clip(la + bay + 0.3 * sjv, 0, 1)[None].repeat(T, 0),
        "crop": np.clip(sjv, 0, 1)[None].repeat(T, 0),
        "nlights": (5 * (la + bay))[None].repeat(T, 0) * (1 + 0.01 * rng.normal(size=(T, H, W))),
    }
    gaps = rng.random((T, H, W)) < 0.15           # cloud-like gaps
    for g in ("no2", "ch4", "so2", "co"):
        raw[g] = np.where(gaps, np.nan, raw[g])
    raw = {c: raw[c].astype(float) for c in CHANNELS}
    fires = np.zeros((T, H, W), dtype=np.int32)
    fy, fx = int(np.argmin(abs(lats - 38.0))), int(np.argmin(abs(lons + 120.5)))   # Sierra foothills, inside CA
    fires[T // 2, fy, fx] = 3
    return raw, fires
