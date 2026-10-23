"""Carbon Emission Index (paper Eq. 8).

    CEI(t,h,w) = sum_g [ w_g * GWP_g * gas_norm(g,t,h,w) ] / sum_g w_g

When a gas is missing in a cell-month (QA gap, or NO2/CO removed by the wildfire mask)
the sums run over the available gases only, so the denominator is the weight sum of the
gases actually present. This is my handling of a case the paper does not discuss.
"""
from __future__ import annotations

import numpy as np

from .config import GASES


def gas_contributions(norm: dict[str, np.ndarray], weights: dict[str, float], gwp: dict[str, float]) -> dict[str, np.ndarray]:
    """Per-gas additive contribution to CEI, [T,H,W] each; they sum to CEI exactly."""
    stack_w = np.stack([np.where(np.isfinite(norm[g]), weights[g], 0.0) for g in GASES])
    denom = stack_w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {g: np.where(denom > 0, weights[g] * gwp[g] * norm[g] / np.where(denom > 0, denom, np.nan), np.nan)
                for g in GASES}


def compute_cei(norm: dict[str, np.ndarray], weights: dict[str, float], gwp: dict[str, float]) -> np.ndarray:
    contrib = gas_contributions(norm, weights, gwp)
    stack = np.stack([contrib[g] for g in GASES])
    allnan = np.all(np.isnan(stack), axis=0)
    cei = np.nansum(stack, axis=0)
    cei[allnan] = np.nan
    return cei
