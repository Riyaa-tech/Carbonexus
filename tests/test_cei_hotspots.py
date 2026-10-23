import numpy as np
import pytest

from carbonexus import cei, explain, hotspots as hs
from carbonexus.config import GASES, load_config

CFG = load_config()
W, G = CFG.cei_weights(), CFG.gwp()


def full(v):
    return np.full((1, 1, 1), float(v))


def test_paper_weights_and_gwp_defaults():
    assert W == {"no2": 0.30, "co": 0.20, "ch4": 0.35, "so2": 0.15}
    assert sum(W.values()) == pytest.approx(1.0)
    assert G["ch4"] == 28.0


def test_cei_matches_eq8_by_hand():
    z = {"no2": full(1), "ch4": full(2), "so2": full(-1), "co": full(0.5)}
    expect = (0.30 * 1 * 1 + 0.35 * 28 * 2 + 0.15 * 1 * -1 + 0.20 * 1 * 0.5) / 1.0
    assert cei.compute_cei(z, W, G)[0, 0, 0] == pytest.approx(expect)


def test_cei_renormalises_over_available_gases():
    z = {"no2": full(np.nan), "ch4": full(1), "so2": full(1), "co": full(np.nan)}
    expect = (0.35 * 28 + 0.15) / (0.35 + 0.15)
    assert cei.compute_cei(z, W, G)[0, 0, 0] == pytest.approx(expect)
    assert np.isnan(cei.compute_cei({g: full(np.nan) for g in GASES}, W, G)).all()


def test_contributions_sum_to_cei():
    rng = np.random.default_rng(0)
    z = {g: rng.normal(size=(3, 4, 5)) for g in GASES}
    z["no2"][0, 0, 0] = np.nan
    c = cei.gas_contributions(z, W, G)
    total = np.nansum([c[g] for g in GASES], axis=0)   # a missing gas contributes nothing
    expect = cei.compute_cei(z, W, G)
    assert np.allclose(total, expect)
    assert np.isfinite(expect[0, 0, 0])   # one missing gas must not blank the cell


def test_label_hotspots_orders_classes_by_cei():
    rng = np.random.default_rng(0)
    cei_map = np.concatenate([rng.normal(m, 0.1, 100) for m in (0, 1, 2, 3)]).reshape(1, 20, 20)
    mask = np.ones((20, 20), bool)
    labels, centers, thr = hs.label_hotspots(cei_map, mask)
    assert np.all(np.diff(centers[0]) > 0) and thr.shape == (1, 3)
    for k in range(4):
        assert cei_map[0][labels[0] == k].mean() == pytest.approx(k, abs=0.15)
    # thresholds are consistent with the labels
    assert np.array_equal(np.searchsorted(thr[0], cei_map[0]), labels[0])


def test_label_hotspots_masks_nan_and_outside():
    c = np.random.default_rng(1).normal(size=(2, 10, 10)); c[0, 0, 0] = np.nan
    mask = np.ones((10, 10), bool); mask[5:] = False
    labels, *_ = hs.label_hotspots(c, mask)
    assert labels[0, 0, 0] == -1 and (labels[:, 5:] == -1).all() and (labels[:, :5][np.isfinite(c[:, :5])] >= 0).all()


def test_label_mode_mean_uses_one_clustering():
    c = np.random.default_rng(2).normal(size=(3, 10, 10))
    _, centers, _ = hs.label_hotspots(c, np.ones((10, 10), bool), mode="mean")
    assert np.allclose(centers[0], centers[2])


def test_too_few_cells_raises_for_mean_mode():
    with pytest.raises(ValueError):
        hs.label_hotspots(np.ones((1, 2, 1)), np.ones((2, 1), bool), mode="mean")


def test_extract_hotspots_finds_components_and_filters_small():
    lab = np.zeros((10, 10), int); lab[1:4, 1:4] = 3; lab[8, 8] = 3; lab[6, 1:3] = 2
    cei_map = lab.astype(float)
    lats, lons = 40 - 0.05 * np.arange(10), -120 + 0.05 * np.arange(10)
    found = hs.extract_hotspots(lab, cei_map, lats, lons, 0.05, min_cells=3)
    assert len(found) == 1 and found[0]["cells"] == 9 and found[0]["max_class"] == 3
    gj = hs.hotspots_geojson(found, lats, lons, 0.05)
    assert gj["features"][0]["geometry"]["type"] == "MultiPolygon" and len(gj["features"][0]["geometry"]["coordinates"]) == 9


def test_classify_value():
    thr = np.array([1.0, 2.0, 3.0])
    assert [hs.classify_value(v, thr) for v in (0, 1.5, 2.5, 9)] == [0, 1, 2, 3]
    assert hs.classify_value(np.nan, thr) == -1


def test_explain_shares_sum_to_one_and_point_to_dominant_gas():
    rng = np.random.default_rng(0)
    z = {g: rng.normal(0, 0.1, (1, 10, 10)) for g in GASES}
    z["co"][0, :3, :3] += 5            # co anomaly in the target cells
    contrib = cei.gas_contributions(z, W, G)
    target = np.zeros((10, 10), bool); target[:3, :3] = True
    ex = explain.explain_cells(contrib, 0, target, np.ones((10, 10), bool))
    shares = {g: ex["gases"][g]["excess_share"] for g in GASES}
    assert sum(shares.values()) == pytest.approx(1.0)
    assert max(shares, key=shares.get) == "co"
    assert ex["cei"] == pytest.approx(np.nanmean(cei.compute_cei(z, W, G)[0][target]), rel=1e-6)
