"""Data validation gate run before and after preprocessing. Produces a JSON-able report."""
from __future__ import annotations

import numpy as np

from .config import CHANNELS, GASES

# Plausible physical ranges of the raw GEE products (native units). Values outside are
# flagged, not silently dropped.
RANGES = {
    "no2": (-1e-3, 5e-3),        # mol/m2 tropospheric column
    "ch4": (1500.0, 2300.0),     # ppb XCH4
    "so2": (-5e-2, 5e-2),        # mol/m2
    "co": (0.0, 0.2),            # mol/m2
    "u10": (-40, 40), "v10": (-40, 40),   # m/s
    "t2m": (230, 330),           # K
    "tp": (0.0, 2.0),            # m (monthly sum)
    "sp": (4e4, 1.1e5),          # Pa
    "built": (0.0, 1.0), "crop": (0.0, 1.0),
    "nlights": (-5, 1e4),        # nW/cm2/sr
}


def validate_raw(cube: dict[str, np.ndarray], state_mask: np.ndarray, min_valid_fraction: float = 0.5) -> dict:
    """Per channel: fraction of in-state cells that are finite, per month; out-of-range counts.
    `ok` is False if any gas channel has a month below `min_valid_fraction`."""
    rep = {"channels": {}, "problems": []}
    n_cells = int(state_mask.sum())
    for c in CHANNELS:
        a = cube[c]
        T = a.shape[0]
        frac = np.array([np.isfinite(a[t][state_mask]).mean() if n_cells else 0.0 for t in range(T)])
        lo, hi = RANGES[c]
        with np.errstate(invalid="ignore"):
            oor = int(((a < lo) | (a > hi))[:, state_mask].sum())
        rep["channels"][c] = {"valid_fraction_mean": float(frac.mean()), "valid_fraction_min": float(frac.min()),
                              "months_below_threshold": int((frac < min_valid_fraction).sum()),
                              "out_of_range_values": oor}
        if c in GASES and (frac < min_valid_fraction).any():
            rep["problems"].append(f"{c}: {(frac < min_valid_fraction).sum()} months have <{min_valid_fraction:.0%} valid in-state cells")
        if oor:
            rep["problems"].append(f"{c}: {oor} values outside plausible range {RANGES[c]}")
    rep["ok"] = not any("valid in-state" in p for p in rep["problems"])
    return rep


def validate_processed(norm: dict[str, np.ndarray], cei: np.ndarray, state_mask: np.ndarray) -> dict:
    rep = {"problems": []}
    for g in GASES:
        v = norm[g][:, state_mask]
        if not np.isfinite(v).any():
            rep["problems"].append(f"{g}: no finite values after preprocessing")
    c = cei[:, state_mask]
    rep["cei_finite_fraction"] = float(np.isfinite(c).mean())
    rep["cei_min"] = float(np.nanmin(c)) if np.isfinite(c).any() else None
    rep["cei_max"] = float(np.nanmax(c)) if np.isfinite(c).any() else None
    if rep["cei_finite_fraction"] < 0.8:
        rep["problems"].append("fewer than 80% of in-state cell-months have a CEI")
    rep["ok"] = not rep["problems"]
    return rep
