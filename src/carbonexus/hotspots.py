"""Hotspot classification (paper Section 2.8) and hotspot extraction.

Classes come from adaptive K-means (k=4, k-means++, n_init=10, random_state=42) on the
CEI of the in-state cells, with clusters re-ordered so class 0 has the lowest centre.
Class meaning is therefore *relative to the distribution of the month (or of the time-mean
map)*, not an absolute emission threshold. See docs/limitations.md.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

CLASS_NAMES = ("Low", "Moderate", "High", "Very High")


def _kmeans_1d(values: np.ndarray, k: int, seed: int):
    if values.size < k:
        raise ValueError(f"Not enough valid values for {k} clusters ({values.size})")
    km = KMeans(n_clusters=k, n_init=10, init="k-means++", random_state=seed).fit(values.reshape(-1, 1))
    centers = km.cluster_centers_.ravel()
    order = np.argsort(centers)
    remap = np.empty(k, dtype=np.int8)
    remap[order] = np.arange(k)
    return km, remap, centers[order]


def label_hotspots(cei: np.ndarray, mask: np.ndarray, n_classes: int = 4, mode: str = "per_month", random_state: int = 42):
    """cei [T,H,W], mask [H,W] (in-state). Returns (labels int8 [T,H,W] with -1 = no data,
    centers [T,k], thresholds [T,k-1]) where thresholds are midpoints between sorted centres.
    With mode='mean' one clustering of the time-mean CEI is applied to every month."""
    T = cei.shape[0]
    labels = np.full(cei.shape, -1, dtype=np.int8)
    centers = np.full((T, n_classes), np.nan)
    if mode == "mean":
        with np.errstate(all="ignore"):
            mean_cei = np.nanmean(cei, axis=0)
        valid = mask & np.isfinite(mean_cei)
        km, remap, c = _kmeans_1d(mean_cei[valid], n_classes, random_state)
        centers[:] = c
        thr = (c[:-1] + c[1:]) / 2
        for t in range(T):
            ok = mask & np.isfinite(cei[t])
            labels[t][ok] = np.searchsorted(thr, cei[t][ok]).astype(np.int8)
    elif mode == "per_month":
        for t in range(T):
            valid = mask & np.isfinite(cei[t])
            if valid.sum() < n_classes:
                continue
            km, remap, c = _kmeans_1d(cei[t][valid], n_classes, random_state)
            centers[t] = c
            labels[t][valid] = remap[km.labels_]
    else:
        raise ValueError(f"unknown label_mode {mode!r}")
    thresholds = (centers[:, :-1] + centers[:, 1:]) / 2
    return labels, centers, thresholds


def classify_value(v: float, thresholds: np.ndarray) -> int:
    """Class (0..k-1) of a CEI value (e.g. a regional mean) given a month's thresholds."""
    if not np.isfinite(v) or not np.all(np.isfinite(thresholds)):
        return -1
    return int(np.searchsorted(thresholds, v))


def extract_hotspots(label_map: np.ndarray, cei_map: np.ndarray, lats, lons, res: float,
                     min_cells: int = 3, min_class: int = 2) -> list[dict]:
    """Connected components (8-neighbour) of cells with class >= min_class.
    Each hotspot carries its class (modal class of cells, plus the peak class)."""
    binary = label_map >= min_class
    lab, n = ndimage.label(binary, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        cells = lab == i
        cnt = int(cells.sum())
        if cnt < min_cells:
            continue
        ys, xs = np.where(cells)
        vals = cei_map[cells]
        classes = label_map[cells]
        peak = int(np.argmax(vals))
        out.append({
            "id": i, "cells": cnt,
            "area_km2": float(sum(111.32 * res * 111.32 * res * np.cos(np.deg2rad(lats[y])) for y in ys)),
            "centroid": [float(lons[xs].mean()), float(lats[ys].mean())],
            "peak_cell": [float(lons[xs[peak]]), float(lats[ys[peak]])],
            "bbox": [float(lons[xs].min() - res / 2), float(lats[ys].min() - res / 2),
                     float(lons[xs].max() + res / 2), float(lats[ys].max() + res / 2)],
            "mean_cei": float(np.nanmean(vals)), "max_cei": float(np.nanmax(vals)),
            "max_class": int(classes.max()),
            "frac_very_high": float((classes == 3).mean()),
            "_idx": (ys, xs),
        })
    out.sort(key=lambda h: (-h["max_class"], -h["mean_cei"]))
    return out


def hotspots_geojson(hotspots: list[dict], lats, lons, res: float) -> dict:
    """GeoJSON FeatureCollection; each hotspot is a MultiPolygon of its grid cells."""
    feats = []
    for h in hotspots:
        ys, xs = h["_idx"]
        polys = []
        for y, x in zip(ys, xs):
            w, e = lons[x] - res / 2, lons[x] + res / 2
            s, n = lats[y] - res / 2, lats[y] + res / 2
            polys.append([[[w, s], [e, s], [e, n], [w, n], [w, s]]])
        props = {k: v for k, v in h.items() if not k.startswith("_")}
        props["class_name"] = CLASS_NAMES[h["max_class"]]
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "MultiPolygon", "coordinates": polys}})
    return {"type": "FeatureCollection", "features": feats}
