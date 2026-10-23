import numpy as np

from carbonexus import validate
from carbonexus.config import CHANNELS, load_config
from carbonexus.grid import REGIONS, box_mask, polygon_mask, region_masks


def test_california_mask_places_known_points_correctly():
    pts = {"LA": (34.05, -118.25, True), "SF": (37.77, -122.42, True), "Fresno": (36.74, -119.79, True),
           "Las Vegas": (36.17, -115.14, False), "Pacific": (35.0, -125.0, False), "Phoenix": (33.45, -112.07, False),
           "Portland": (45.5, -122.7, False), "Reno": (39.53, -119.81, False)}
    for name, (lat, lon, inside) in pts.items():
        m = polygon_mask(np.array([lat]), np.array([lon]))
        assert bool(m[0, 0]) is inside, name


def test_grid_dimensions_match_paper():
    cfg = load_config()
    assert (len(cfg.lats), len(cfg.lons), len(cfg.months)) == (190, 240, 72)
    assert cfg.bounds == (-124.5, 32.5, -112.5, 42.0)


def test_regions_are_inside_study_grid_and_have_cells():
    cfg = load_config()
    state = polygon_mask(cfg.lats, cfg.lons)
    masks = region_masks(cfg.lats, cfg.lons, state)
    assert set(masks) == set(REGIONS)
    assert all(m.sum() > 0 for m in masks.values())
    assert box_mask(cfg.lats, cfg.lons, REGIONS["la_basin"][1]).sum() > 100


def test_validate_raw_flags_empty_gas_and_out_of_range():
    cfg = load_config()
    T, H, W = 3, 4, 4
    base = {"no2": 1e-4, "ch4": 1850., "so2": 1e-4, "co": 0.03, "u10": 1., "v10": 1., "t2m": 290., "tp": .01, "sp": 1e5,
            "built": .1, "crop": .1, "nlights": 1.}
    cube = {c: np.full((T, H, W), base[c]) for c in CHANNELS}
    mask = np.ones((H, W), bool)
    assert validate.validate_raw(cube, mask)["ok"]
    cube["no2"][1] = np.nan
    cube["ch4"][0, 0, 0] = 5000.
    rep = validate.validate_raw(cube, mask, 0.5)
    assert not rep["ok"] and any("no2" in p for p in rep["problems"]) and any("ch4" in p and "range" in p for p in rep["problems"])
