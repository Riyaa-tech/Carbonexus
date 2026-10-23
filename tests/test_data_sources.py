import numpy as np
import pandas as pd
import pytest

from carbonexus.config import CHANNELS, load_config
from carbonexus.data import firms, gee, inventories as inv

CFG = load_config()

CSV = """latitude,longitude,bright_ti4,acq_date,acq_time,confidence,type
34.05,-118.25,330,2020-08-03,1200,n,0
34.06,-118.26,330,2020-08-04,1200,h,0
34.05,-118.25,330,2020-08-05,1200,l,0
34.05,-118.25,330,2020-08-05,1200,n,2
34.05,-118.25,330,2020-08-05,1200,n,3
"""


def test_firms_parse_filter_and_rasterise():
    df = firms.filter_wildfires(firms.parse_csv(CSV))
    assert len(df) == 2                           # low-confidence and static/offshore types dropped
    counts = firms.monthly_counts(df, CFG)
    t = (2020 - 2019) * 12 + 7
    assert counts.shape == (72, 190, 240) and counts.sum() == 2 and counts[t].sum() == 2


def test_firms_rejects_error_payload():
    with pytest.raises(RuntimeError):
        firms.parse_csv("Invalid MAP_KEY.")
    assert firms.parse_csv("").empty


def test_gee_parse_pixels_maps_nodata_to_nan():
    dt = [(c, "<f4") for c in CHANNELS]
    arr = np.zeros((2, 3), dtype=dt)
    for c in CHANNELS:
        arr[c][:] = 1.0
    arr["no2"][0, 0] = gee.NODATA
    out = gee.parse_pixels(arr)
    assert np.isnan(out["no2"][0, 0]) and out["no2"][1, 1] == 1.0 and set(out) == set(CHANNELS)


def test_gee_grid_request_matches_study_grid():
    g = gee.grid_request(CFG)
    assert g["dimensions"] == {"width": 240, "height": 190}
    assert g["affineTransform"]["scaleX"] == 0.05 and g["affineTransform"]["scaleY"] == -0.05
    assert g["affineTransform"]["translateX"] == -124.5 and g["affineTransform"]["translateY"] == 42.0


class FakeImg:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def __getattr__(self, attr):
        def f(*a, **k):
            self.log.append((self.name, attr, a))
            return self
        return f


class FakeEE:
    def __init__(self):
        self.log = []

    def ImageCollection(self, cid):
        self.log.append(("collection", cid)); return FakeImg(self.log, cid)

    class Image:
        @staticmethod
        def cat(bands):
            return FakeImg([], "cat")


def test_gee_monthly_image_requests_expected_collections_and_month_window():
    ee = FakeEE()
    static = {"built": FakeImg([], "b"), "crop": FakeImg([], "c")}
    gee.monthly_image(ee, CFG, pd.Timestamp("2020-02-01"), static)
    ids = [e[1] for e in ee.log if e[0] == "collection"]
    for cid, *_ in gee.S5P.values():
        assert cid in ids
    assert gee.ERA5_COLL in ids and gee.NIGHTLIGHTS in ids
    windows = {e[2] for e in ee.log if len(e) == 3 and e[1] == "filterDate"}
    assert ("2020-02-01", "2020-03-01") in windows


def test_load_cube_reports_missing_months(tmp_path):
    cfg = load_config(data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="72 months"):
        gee.load_cube(cfg)


def test_inventory_regrid_and_validation():
    lats, lons = CFG.lats[:20], CFG.lons[:20]
    src_lats, src_lons = lats[::2], lons[::2]
    vals = np.ones((len(src_lats), len(src_lons)))
    g = inv.regrid_sum(vals, src_lats, src_lons, lats, lons, 0.05)
    assert g.sum() == vals.size
    fac = pd.DataFrame({"latitude": [lats[3]], "longitude": [lons[4]], "emissions": [7.0]})
    fg = inv.facilities_to_grid(fac, lats, lons, 0.05)
    assert fg[3, 4] == 7.0 and fg.sum() == 7.0
    rng = np.random.default_rng(0)
    cei = rng.random((20, 20)); lab = (cei > 0.75).astype(int) * 3
    rep = inv.validate_against_inventory(cei, lab, cei ** 2 + 0.01, np.ones((20, 20), bool))
    assert rep["spearman_rho"] > 0.99 and rep["inventory_share_in_very_high"] > rep["very_high_area_share"]


def test_facility_csv_requires_columns(tmp_path):
    p = tmp_path / "f.csv"; p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        inv.load_facilities_csv(p)


def test_edgar_netcdf_roundtrip(tmp_path):
    import xarray as xr
    da = xr.DataArray(np.arange(6.).reshape(2, 3), coords={"lat": [35.0, 36.0], "lon": [-120.0, -119.9, -119.8]}, dims=("lat", "lon"), name="emi_co2")
    p = tmp_path / "e.nc"; da.to_dataset().to_netcdf(p)
    v, la, lo = inv.load_edgar_nc(p)
    assert la[0] == 36.0 and v.shape == (2, 3) and v[0, 0] == 3.0
