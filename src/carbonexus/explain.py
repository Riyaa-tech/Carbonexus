"""Explainability.

CEI is an *additive* function of the normalised gas columns, so the contribution of each
gas to a cell's CEI is known exactly (it equals the Shapley value of that gas for the
linear model, relative to a zero baseline). We report
  * the raw contribution  c_g = w_g * GWP_g * z_g / sum(w)           (sums to CEI)
  * the excess over the statewide median contribution of the same gas in that month
    and the positive-excess share of each gas  -> "what pushed this cell above typical".
This is NOT DeepSHAP on a SegFormer (the paper's method); the classifier here is K-means
on CEI, whose input is exactly this decomposition. Meteorology and land-use layers are
reported as *context* z-scores, not as attributions: they do not enter CEI.
"""
from __future__ import annotations

import numpy as np

from .config import GASES


def explain_cells(contrib: dict[str, np.ndarray], t: int, mask_cells: np.ndarray, state_mask: np.ndarray) -> dict:
    """Aggregate gas contributions over `mask_cells` ([H,W] bool) for month index t."""
    baseline = {}
    for g in GASES:
        v = contrib[g][t][state_mask]
        v = v[np.isfinite(v)]
        baseline[g] = float(np.median(v)) if v.size else 0.0
    rows = {}
    for g in GASES:
        v = contrib[g][t][mask_cells]
        v = v[np.isfinite(v)]
        mean_c = float(v.mean()) if v.size else float("nan")
        rows[g] = {"contribution": mean_c, "statewide_median": baseline[g],
                   "excess": mean_c - baseline[g] if np.isfinite(mean_c) else float("nan"),
                   "valid_cells": int(v.size)}
    pos = {g: max(r["excess"], 0.0) if np.isfinite(r["excess"]) else 0.0 for g, r in rows.items()}
    tot = sum(pos.values())
    for g in GASES:
        rows[g]["excess_share"] = pos[g] / tot if tot > 0 else 0.0
    total = float(np.nansum([r["contribution"] for r in rows.values()]))
    return {"gases": rows, "cei": total}


def context_zscores(channels: dict[str, np.ndarray], t: int, mask_cells: np.ndarray, names) -> dict:
    """Mean of each context layer over the cells, expressed as a z-score of the cells'
    own multi-year distribution for the same calendar month (anomaly vs climatology)."""
    out = {}
    T = next(iter(channels.values())).shape[0]
    same_month = np.arange(t % 12, T, 12)
    for n in names:
        arr = channels.get(n)
        if arr is None:
            continue
        series = np.array([np.nanmean(arr[i][mask_cells]) if np.isfinite(arr[i][mask_cells]).any() else np.nan for i in same_month])
        cur = np.nanmean(arr[t][mask_cells]) if np.isfinite(arr[t][mask_cells]).any() else np.nan
        sd = np.nanstd(series)
        out[n] = {"value": float(cur) if np.isfinite(cur) else None,
                  "anomaly_z": float((cur - np.nanmean(series)) / sd) if sd and np.isfinite(sd) and np.isfinite(cur) else None}
    return out
