import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from carbonexus import pipeline
from carbonexus.api.app import create_app
from carbonexus.config import GASES


def test_pipeline_outputs_are_consistent(analysis):
    cfg, raw, fires, res = analysis
    ds, sm = res["dataset"], res["summary"]
    assert sm["provenance"].startswith("SYNTHETIC")
    assert ds["cei"].shape == (72, 48, 60)
    lab, cei, mask = ds["label"].values, ds["cei"].values, ds["state_mask"].values
    assert (lab[:, ~mask] == -1).all()
    assert set(np.unique(lab[:, mask])) <= {-1, 0, 1, 2, 3}
    # contributions sum to CEI where CEI exists
    tot = np.nansum([ds[f"contrib_{g}"].values for g in GASES], axis=0)
    ok = np.isfinite(cei)
    assert np.allclose(tot[ok], cei[ok], atol=1e-3, rtol=1e-4)
    # classes monotone in CEI within a month
    t = 10
    for k in range(3):
        a, b = cei[t][lab[t] == k], cei[t][lab[t] == k + 1]
        assert a.max() <= b.min() + 1e-6
    # fire mask removed NO2/CO but not CH4 in the burning cell
    t0, y, x = np.argwhere(fires > 0)[0]
    assert mask[y, x]
    assert np.isnan(ds["contrib_no2"].values[t0, y, x]) and np.isnan(ds["contrib_co"].values[t0, y, x])
    assert np.isfinite(ds["contrib_ch4"].values[t0, y, x]) and np.isfinite(cei[t0, y, x])


def test_pipeline_regions_forecast_and_backtest(analysis):
    *_, res = analysis
    sm = res["summary"]
    la = sm["regions"]["la_basin"]
    assert len(la["cei"]) == 72 and len(la["class"]) == 72 and la["cells"] > 5
    f = sm["forecast"]["baseline"]["la_basin"]
    assert len(f["mean"]) == 60 and f["start"] == "2025-01" and all(s >= 0 for s in f["std"])
    bt = sm["report"]["forecast_backtest"]
    assert bt["baseline"]["n"] > 0 and bt["seasonal_naive"]["n"] > 0 and "bilstm" not in bt
    assert sm["report"]["validation_raw"]["ok"]


def test_save_load_roundtrip(analysis):
    cfg, *_ , res = analysis
    ds, sm = pipeline.load(cfg)
    assert np.allclose(ds["cei"].values, res["dataset"]["cei"].values, equal_nan=True)
    assert sm["regions"].keys() == res["summary"]["regions"].keys()
    json.dumps(sm)


@pytest.fixture(scope="module")
def client(analysis):
    cfg = analysis[0]
    return TestClient(create_app(cfg))


def test_api_status_and_grid(client):
    st = client.get("/api/status").json()
    assert st["ready"] and st["provenance"].startswith("SYNTHETIC") and len(st["months"]) == 72
    g = client.get("/api/grid/2024-12").json()
    assert len(g["labels"]) == 48 and len(g["labels"][0]) == 60 and len(g["thresholds"]) == 3
    assert client.get("/api/grid/1999-01").status_code == 404


def test_api_regions_timeseries_forecast(client):
    regs = client.get("/api/regions").json()
    assert {r["id"] for r in regs} >= {"la_basin", "san_joaquin_valley"}
    ts = client.get("/api/regions/la_basin/timeseries").json()
    assert len(ts["cei"]) == 72
    fc = client.get("/api/forecast/la_basin?horizon=6").json()
    assert len(fc["mean"]) == 6 and fc["months"][0] == "2025-01" and "BASELINE" in fc["label"]
    assert client.get("/api/forecast/la_basin?model=bilstm").status_code == 404
    assert client.get("/api/regions/nope/timeseries").status_code == 404


def test_api_explain_and_hotspots(client):
    ex = client.get("/api/explain/2024-12?region=san_joaquin_valley").json()
    assert sum(ex["gases"][g]["excess_share"] for g in GASES) == pytest.approx(1.0)
    assert "NOT DeepSHAP" in ex["method"]
    assert client.get("/api/explain/2024-12?lat=34.05&lon=-118.25").status_code == 200
    assert client.get("/api/explain/2024-12").status_code == 400
    gj = client.get("/api/hotspots/2024-12").json()
    assert gj["type"] == "FeatureCollection"
    for f in gj["features"]:
        assert f["properties"]["max_class"] >= 2 and f["properties"]["top_driver"] in GASES


def test_api_reports_not_ready_without_analysis(tmp_path):
    from carbonexus.config import load_config
    c = TestClient(create_app(load_config(data_dir=tmp_path)))
    assert c.get("/api/status").json()["ready"] is False
    assert c.get("/api/regions").status_code == 503
    assert "CarbonNexus" in c.get("/").text


def test_validation_endpoint_exposes_weights(client):
    v = client.get("/api/validation").json()
    assert v["config"]["cei"]["gwp_ch4"] == 28.0
