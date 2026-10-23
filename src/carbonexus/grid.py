"""Study grid helpers: California mask and named analysis regions.

Both the California outline and the region boxes are *hand-drawn, coarse
approximations* (documented in docs/limitations.md). They are used for masking
and aggregation only; they are not an official boundary dataset.
"""
from __future__ import annotations

import numpy as np

# Coarse California outline (lon, lat), ~0.1-0.3 deg accuracy. San Francisco Bay
# is not carved out.
CALIFORNIA_OUTLINE = [
    (-124.2, 42.0), (-120.0, 42.0), (-120.0, 39.0), (-114.63, 35.0), (-114.14, 34.30),
    (-114.43, 34.0), (-114.72, 32.72), (-117.12, 32.53), (-117.25, 32.9), (-117.4, 33.2),
    (-117.7, 33.46), (-118.2, 33.75), (-118.4, 33.75), (-118.5, 34.0), (-119.1, 34.1),
    (-119.7, 34.4), (-120.47, 34.45), (-120.65, 34.58), (-120.65, 34.9), (-120.87, 35.37),
    (-121.28, 35.67), (-121.9, 36.3), (-121.9, 36.6), (-122.0, 36.95), (-122.5, 37.5),
    (-122.51, 37.78), (-123.0, 38.0), (-123.05, 38.3), (-123.7, 38.95), (-124.4, 40.44),
    (-124.2, 40.8), (-124.2, 41.75),
]

# Named analysis regions: (west, south, east, north). Approximate analyst-defined boxes
# modelled on the regions named in the paper (Section 3.1/3.2).
REGIONS = {
    "la_basin": ("LA Basin", (-118.9, 33.6, -117.0, 34.4)),
    "inland_empire": ("Inland Empire", (-117.7, 33.8, -116.8, 34.3)),
    "san_diego": ("San Diego", (-117.4, 32.55, -116.9, 33.2)),
    "bay_area": ("Bay Area", (-122.7, 37.2, -121.7, 38.2)),
    "sacramento_metro": ("Sacramento Metro", (-121.8, 38.3, -121.0, 38.9)),
    "san_joaquin_valley": ("San Joaquin Valley", (-121.2, 35.0, -118.6, 37.8)),
    "northern_mountains": ("Northern Mountains", (-123.5, 40.5, -121.0, 42.0)),
    "eastern_desert": ("Eastern Desert", (-117.0, 34.5, -114.6, 36.5)),
}


def polygon_mask(lats: np.ndarray, lons: np.ndarray, poly=CALIFORNIA_OUTLINE) -> np.ndarray:
    """Boolean [H, W] mask of cell centres inside polygon (even-odd ray casting)."""
    lon2, lat2 = np.meshgrid(lons, lats)
    x, y = lon2.ravel(), lat2.ravel()
    inside = np.zeros(x.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if y1 == y2:
            continue
        cond = (y1 > y) != (y2 > y)
        xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (x < xint)
    return inside.reshape(lat2.shape)


def box_mask(lats: np.ndarray, lons: np.ndarray, box) -> np.ndarray:
    w, s, e, n = box
    lon2, lat2 = np.meshgrid(lons, lats)
    return (lon2 >= w) & (lon2 <= e) & (lat2 >= s) & (lat2 <= n)


def region_masks(lats, lons, state_mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    out = {}
    for rid, (_, box) in REGIONS.items():
        m = box_mask(lats, lons, box)
        if state_mask is not None:
            m &= state_mask
        out[rid] = m
    return out
