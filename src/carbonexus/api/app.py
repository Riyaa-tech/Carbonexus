"""FastAPI backend + static dashboard. Reads the saved analysis; computes nothing heavy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import explain, hotspots as hs, pipeline
from ..config import GASES, MET, Config, load_config
from ..grid import REGIONS, box_mask

WEB = Path(__file__).resolve().parent.parent / "web"


def _nan_to_none(a):
    return [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in a]


def create_app(cfg: Config | None = None, name: str = "analysis") -> FastAPI:
    cfg = cfg or load_config()
    app = FastAPI(title="CarbonNexus MVP", version="0.1.0")
    state: dict = {}

    def data():
        if "ds" not in state:
            try:
                state["ds"], state["summary"] = pipeline.load(cfg, name)
            except FileNotFoundError as e:
                raise HTTPException(503, str(e))
            ds = state["ds"]
            state["months"] = [f"{pd_t:%Y-%m}" for pd_t in ds["time"].to_index()]
            state["lats"], state["lons"] = ds["lat"].values, ds["lon"].values
            state["mask"] = ds["state_mask"].values.astype(bool)
            state["contrib"] = {g: ds[f"contrib_{g}"].values for g in GASES}
            state["ctx"] = {m: ds[f"z_{m}"].values for m in MET if f"z_{m}" in ds}
        return state

    def tidx(s, month):
        try:
            return s["months"].index(month)
        except ValueError:
            raise HTTPException(404, f"month {month} not in {s['months'][0]}..{s['months'][-1]}")

    @app.get("/api/status")
    def status():
        try:
            s = data()
        except HTTPException as e:
            return {"ready": False, "detail": e.detail}
        sm = s["summary"]
        return {"ready": True, "provenance": sm["provenance"], "months": s["months"], "class_names": sm["class_names"],
                "bounds": list(cfg.bounds), "shape": [len(s["lats"]), len(s["lons"])],
                "label_mode": sm["config"]["hotspots"]["label_mode"],
                "forecast_models": ["baseline"] + (["bilstm"] if (sm["forecast"].get("bilstm") or {}).get("forecasts") else []),
                "validation_ok": sm["report"]["validation_raw"]["ok"] and sm["report"]["validation_processed"]["ok"]}

    @app.get("/api/grid/{month}")
    def grid(month: str):
        s = data(); t = tidx(s, month)
        ds = s["ds"]
        lab = ds["label"].values[t].astype(int)
        return {"month": month, "bounds": list(cfg.bounds), "labels": lab.tolist(),
                "cei": _nan_to_none(ds["cei"].values[t]),
                "centers": [None if not np.isfinite(v) else float(v) for v in ds["centers"].values[t]],
                "thresholds": [None if not np.isfinite(v) else float(v) for v in ds["thresholds"].values[t]]}

    @app.get("/api/regions")
    def regions():
        sm = data()["summary"]
        out = []
        for rid, r in sm["regions"].items():
            out.append({"id": rid, "name": r["name"], "box": r["box"], "cells": r["cells"],
                        "latest_cei": r["cei"][-1], "latest_class": r["class"][-1],
                        "mean_cei": r["mean_cei"], "months_very_high": r["months_very_high"],
                        "trend_cei_per_year": r["trend_cei_per_year"], "class_share_latest": r["class_share_latest"]})
        return out

    @app.get("/api/regions/{rid}/timeseries")
    def timeseries(rid: str):
        s = data()
        r = s["summary"]["regions"].get(rid)
        if not r:
            raise HTTPException(404, "unknown region")
        return {"months": s["months"], "cei": r["cei"], "class": r["class"], "name": r["name"]}

    @app.get("/api/hotspots/{month}")
    def hotspots(month: str):
        s = data(); t = tidx(s, month)
        ds = s["ds"]
        lab = ds["label"].values[t]
        cei = ds["cei"].values[t]
        res = cfg["study"]["res"]
        found = hs.extract_hotspots(lab, cei, s["lats"], s["lons"], res, cfg["hotspots"]["min_component_cells"])
        gj = hs.hotspots_geojson(found, s["lats"], s["lons"], res)
        for h, f in zip(found, gj["features"]):
            m = np.zeros(lab.shape, bool); m[h["_idx"]] = True
            ex = explain.explain_cells(s["contrib"], t, m, s["mask"])
            top = max(GASES, key=lambda g: ex["gases"][g]["excess_share"])
            f["properties"]["top_driver"] = top
            f["properties"]["explanation"] = ex["gases"]
            f["properties"]["context"] = explain.context_zscores(s["ctx"], t, m, MET)
            f["properties"]["region"] = _region_of(h["centroid"])
        return gj

    @app.get("/api/explain/{month}")
    def explain_view(month: str, region: str | None = None, lat: float | None = None, lon: float | None = None):
        s = data(); t = tidx(s, month)
        if region:
            if region not in REGIONS:
                raise HTTPException(404, "unknown region")
            m = box_mask(s["lats"], s["lons"], REGIONS[region][1]) & s["mask"]
            label = REGIONS[region][0]
        elif lat is not None and lon is not None:
            y = int(np.argmin(np.abs(s["lats"] - lat))); x = int(np.argmin(np.abs(s["lons"] - lon)))
            m = np.zeros(s["mask"].shape, bool); m[y, x] = True
            label = f"cell ({s['lats'][y]:.3f}, {s['lons'][x]:.3f})"
        else:
            raise HTTPException(400, "give region= or lat=&lon=")
        if not (m & s["mask"]).any():
            raise HTTPException(404, "no in-state cells selected")
        ex = explain.explain_cells(s["contrib"], t, m, s["mask"])
        ex["context"] = explain.context_zscores(s["ctx"], t, m, MET)
        ex["target"] = label
        ex["method"] = ("exact additive decomposition of CEI (Shapley values of the linear CEI); "
                        "NOT DeepSHAP on SegFormer. Meteorology is context, not an attribution.")
        return ex

    @app.get("/api/forecast/{rid}")
    def forecast(rid: str, model: str = Query("baseline", pattern="^(baseline|bilstm)$"), horizon: int = Query(12, ge=1, le=60)):
        s = data()
        fc = s["summary"]["forecast"]
        src = fc["baseline"] if model == "baseline" else (fc.get("bilstm") or {}).get("forecasts")
        if not src or rid not in src:
            raise HTTPException(404, f"no {model} forecast for {rid}")
        f = src[rid]
        n = min(horizon, len(f["mean"]))
        import pandas as pd
        start = pd.Period(s["months"][-1], "M") + 1
        return {"model": model, "label": ("BASELINE: seasonal-trend OLS (not the paper's BiLSTM)" if model == "baseline"
                                           else "BiLSTM per paper architecture; MC-dropout std of the last round only"),
                "months": [str(start + i) for i in range(n)], "mean": f["mean"][:n], "std": f["std"][:n],
                "class": f.get("class", [None] * n)[:n]}

    @app.get("/api/validation")
    def validation():
        sm = data()["summary"]
        return {**sm["report"], "config": sm["config"]}

    def _region_of(centroid):
        for rid, (name, (w, so, e, n)) in REGIONS.items():
            if w <= centroid[0] <= e and so <= centroid[1] <= n:
                return name
        return None

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app
