"""End-to-end analysis: raw monthly cube -> preprocessing -> CEI -> hotspot classes ->
regional statistics -> forecasts. Pure function of its inputs; persistence in `save`/`load`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import cei as ceimod
from . import hotspots as hs
from . import preprocess as pp
from . import validate
from .config import CHANNELS, GASES, LANDUSE, MET, Config
from .forecast import baseline as fb
from .grid import REGIONS, polygon_mask, region_masks


def _regional_mean(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        sel = arr[:, mask]
        return np.where(np.isfinite(sel).any(axis=1), np.nanmean(np.where(np.isfinite(sel), sel, np.nan), axis=1), np.nan)


def preprocess_cube(raw: dict[str, np.ndarray], fires: np.ndarray, cfg: Config) -> dict:
    p = cfg["preprocess"]
    norm, stats, qa = {}, {}, {}
    for c in GASES + MET + ("nlights",):
        r = pp.preprocess_channel(raw[c], mad_k=p["mad_k"], q=p["kalman_q"], r=p["kalman_r"],
                                  max_gap=p["kalman_max_gap"], window=p["savgol_window"], order=p["savgol_order"])
        norm[c] = r["data"]
        stats[c] = {"median": r["median"], "scale": r["scale"]}
        qa[c] = {"outliers_replaced": int(r["outliers"].sum()), "cells_gap_filled": int(r["filled"].sum())}
    for c in ("built", "crop"):  # static fractions in [0,1]: used as-is, no temporal filtering
        norm[c] = raw[c]
    norm = pp.apply_fire_mask(norm, fires)
    return {"norm": norm, "stats": stats, "qa": qa}


def _region_features(cfg, series: dict[str, np.ndarray], cei_series: np.ndarray):
    """[T,13] feature matrix per region in the paper's order (4 gases, 5 met, 3 land use, CEI),
    standardised with statistics of the training months (all but the last 24)."""
    cols = [series[c] for c in GASES + MET + LANDUSE] + [cei_series]
    F = np.column_stack(cols).astype(float)
    ref = F[:-24] if len(F) > 48 else F
    mu = np.nanmean(ref, axis=0)
    sd = np.nanstd(ref, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (F - mu) / sd, mu, sd


def run_bilstm_forecast(cfg: Config, region_series: dict[str, dict], cei_region: dict[str, np.ndarray],
                        epochs=200, horizon_rounds=1, seed=0) -> dict:
    from .forecast import bilstm
    bilstm.require_torch()
    feats, norm_params = {}, {}
    for rid in cei_region:
        F, mu, sd = _region_features(cfg, region_series[rid], cei_region[rid])
        if np.isfinite(F).all():
            feats[rid] = F.astype(np.float32); norm_params[rid] = (mu, sd)
    if len(feats) < 2:
        return {"error": "fewer than 2 regions with complete series"}
    tcol = 12
    model, rep = bilstm.train_bilstm(feats, tcol, tin=cfg["forecast"]["input_months"], tout=cfg["forecast"]["horizon"],
                                     epochs=epochs, seed=seed)
    test = bilstm.test_metrics(model, feats, tcol, tin=cfg["forecast"]["input_months"], tout=cfg["forecast"]["horizon"])
    # test metrics are in standardised units (CEI std of train months); convert RMSE/MAE to CEI units
    sd_cei = float(np.mean([norm_params[r][1][tcol] for r in feats]))
    for k in ("rmse", "mae", "mean_mc_std"):
        if k in test:
            test[k] *= sd_cei
    out = {"model": "BiLSTM (paper architecture; MC-dropout)", "training": rep, "test_metrics": test, "forecasts": {}}
    months = cfg.months
    nxt = (months[-1].month) % 12
    for rid, F in feats.items():
        mu, sd = norm_params[rid]
        cal = (months[0].month - 1 + np.arange(len(F))) % 12   # calendar month of each row
        train = np.arange(len(F)) < len(F) - 24
        clim_cal = np.stack([F[train & (cal == m)].mean(axis=0) if (train & (cal == m)).any() else np.zeros(F.shape[1])
                             for m in range(12)])
        mean, std = bilstm.roll_forward(model, F[-cfg["forecast"]["input_months"]:], tcol, clim_cal, nxt, horizon_rounds)
        out["forecasts"][rid] = {"mean": (mean * sd[tcol] + mu[tcol]).tolist(), "std": (std * sd[tcol]).tolist()}
    return out


def run_analysis(raw: dict[str, np.ndarray], fires: np.ndarray, cfg: Config, *, provenance: str,
                 use_bilstm: bool = False, bilstm_epochs: int = 200, inventory: dict | None = None) -> dict:
    lats, lons = cfg.lats, cfg.lons
    state = polygon_mask(lats, lons)
    rmasks = region_masks(lats, lons, state)
    report = {"provenance": provenance, "validation_raw": validate.validate_raw(raw, state, cfg["preprocess"]["min_valid_fraction"])}

    pre = preprocess_cube(raw, fires, cfg)
    norm = pre["norm"]
    w, g = cfg.cei_weights(), cfg.gwp()
    contrib = ceimod.gas_contributions(norm, w, g)
    cei = ceimod.compute_cei(norm, w, g)
    cei = np.where(state[None], cei, np.nan)
    report["validation_processed"] = validate.validate_processed(norm, cei, state)
    report["preprocess_qa"] = pre["qa"]
    report["normalisation"] = pre["stats"]

    h = cfg["hotspots"]
    labels, centers, thresholds = hs.label_hotspots(cei, state, h["n_classes"], h["label_mode"], h["random_state"])

    # ---- regional series and stats ------------------------------------------------
    months = cfg.months
    region_series = {rid: {c: _regional_mean(norm[c] if c in norm else raw[c], m) for c in CHANNELS}
                     for rid, m in rmasks.items()}
    cei_region = {rid: _regional_mean(cei, m) for rid, m in rmasks.items()}
    regions = {}
    for rid, m in rmasks.items():
        s = cei_region[rid]
        cls = [hs.classify_value(s[t], thresholds[t]) for t in range(len(months))]
        cell_cls = labels[:, m]
        ok = np.isfinite(s)
        slope = float(np.polyfit(np.arange(len(s))[ok], s[ok], 1)[0] * 12) if ok.sum() > 3 else None
        regions[rid] = {
            "name": REGIONS[rid][0], "box": REGIONS[rid][1], "cells": int(m.sum()),
            "cei": [None if not np.isfinite(v) else float(v) for v in s],
            "class": cls,
            "class_share_latest": {hs.CLASS_NAMES[k]: float((cell_cls[-1] == k).sum() / max((cell_cls[-1] >= 0).sum(), 1)) for k in range(4)},
            "months_very_high": int(sum(c == 3 for c in cls)),
            "trend_cei_per_year": slope,
            "mean_cei": float(np.nanmean(s)) if ok.any() else None,
            "p90_cei_latest": float(np.nanpercentile(cei[-1][m], 90)) if np.isfinite(cei[-1][m]).any() else None,
        }

    # ---- forecast (baseline always; BiLSTM optional) ------------------------------
    horizon = max(cfg["forecast"]["horizon"], 60)
    fc = {"baseline": {}, "bilstm": None}
    ok_series = {rid: s for rid, s in cei_region.items() if np.isfinite(s).sum() >= 36}
    thr_future = np.nanmean(thresholds[-12:], axis=0)
    for rid, s in ok_series.items():
        f = fb.forecast_summary(s, horizon, months[0].month - 1)
        f["class"] = [hs.classify_value(v, thr_future) for v in f["mean"]]
        f["start"] = f"{(months[-1] + pd.offsets.MonthBegin(1)):%Y-%m}"
        fc["baseline"][rid] = f
    report["forecast_backtest"] = fb.backtest_baseline(ok_series, cfg["forecast"]["holdout_months"])
    if use_bilstm:
        try:
            fc["bilstm"] = run_bilstm_forecast(cfg, region_series, cei_region, epochs=bilstm_epochs)
            if "error" not in fc["bilstm"]:
                report["forecast_backtest"]["bilstm"] = fc["bilstm"]["test_metrics"]
        except RuntimeError as e:
            fc["bilstm"] = {"error": str(e)}
    if inventory:
        mean_cei = np.nanmean(cei, axis=0)
        mean_lab, *_ = hs.label_hotspots(mean_cei[None], state, h["n_classes"], "mean", h["random_state"])
        report["inventory_validation"] = {k: _validate_inv(mean_cei, mean_lab[0], v, state) for k, v in inventory.items()}

    ds = xr.Dataset(
        coords={"time": months, "lat": lats, "lon": lons},
        data_vars={
            "cei": (("time", "lat", "lon"), cei.astype("float32")),
            "label": (("time", "lat", "lon"), labels),
            "fires": (("time", "lat", "lon"), fires.astype("int16")),
            "state_mask": (("lat", "lon"), state),
            # only the meteorological context layers are kept; gases are represented by contrib_*
            # and the raw monthly composites stay in data/raw
            **{f"z_{c}": (("time", "lat", "lon"), norm[c].astype("float32")) for c in MET},
            **{f"contrib_{c}": (("time", "lat", "lon"), contrib[c].astype("float32")) for c in GASES},
            "centers": (("time", "cls"), centers),
            "thresholds": (("time", "thr"), thresholds),
        },
        attrs={"provenance": provenance, "label_mode": h["label_mode"],
               "weights": json.dumps(w), "gwp": json.dumps(g)},
    )
    summary = {"provenance": provenance, "months": [f"{m:%Y-%m}" for m in months],
               "class_names": list(hs.CLASS_NAMES), "regions": regions, "forecast": fc, "report": report,
               "config": {k: cfg.raw[k] for k in ("cei", "hotspots", "preprocess", "forecast")}}
    return {"dataset": ds, "summary": summary}


def _validate_inv(mean_cei, labels, inv_grid, mask):
    from .data.inventories import validate_against_inventory
    return validate_against_inventory(mean_cei, labels, inv_grid, mask)


def save(result: dict, cfg: Config, name: str = "analysis") -> Path:
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    nc = cfg.processed_dir / f"{name}.nc"
    enc = {v: {"zlib": True, "complevel": 4, "shuffle": True} for v in result["dataset"].data_vars if result["dataset"][v].ndim >= 2}
    result["dataset"].to_netcdf(nc, encoding=enc)
    (cfg.processed_dir / f"{name}.json").write_text(json.dumps(result["summary"], default=_json_default))
    return nc


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load(cfg: Config, name: str = "analysis") -> tuple[xr.Dataset, dict]:
    nc, js = cfg.processed_dir / f"{name}.nc", cfg.processed_dir / f"{name}.json"
    if not nc.exists() or not js.exists():
        raise FileNotFoundError(f"no analysis in {cfg.processed_dir}; run `carbonexus run` (needs fetched data)")
    return xr.load_dataset(nc), json.loads(js.read_text())
