# CarbonNexus MVP

A runnable implementation of the pipeline in *"AI-Driven Climate Intelligence: Spatio-Temporal Framework for Carbon Emission Hotspot Detection and Prediction"* (Sonawane & Murugavelh, ISPRS manuscript) for **California, 2019-2024, 0.05° grid (190×240), 72 monthly composites**.

> **Status, read this first.** The code was built and tested in a sandbox with no route to Google Earth Engine, Copernicus, NASA FIRMS, EDGAR or CARB. **No real satellite data has been downloaded or analysed by this repository's author, and no result in the repo comes from real data.** The data fetchers are written against the documented APIs but are *untested against the live services*; everything downstream of them is tested on clearly-labelled synthetic fixtures. The first thing to do on your machine is `carbonexus fetch` and inspect the validation report. See [docs/limitations.md](docs/limitations.md).

## What is implemented

| Paper component | Here | Status |
|---|---|---|
| TROPOMI NO₂/CH₄/SO₂/CO, ERA5 (5 vars), land use (3) | `data/gee.py` via Google Earth Engine (ERA5-**Land**, WorldCover, VIIRS night lights) | implemented, **not run live** |
| FIRMS VIIRS fire detections | `data/firms.py` (FIRMS area API) | implemented, **not run live** |
| EDGAR v8.0 / CARB cross-check | `data/inventories.py` (manual downloads) | implemented, tested on tiny fixtures |
| Preprocessing: MAD clip (Eq.1), Kalman gap-fill (Eq.2-6), Savitzky-Golay, robust z (Eq.7), fire masking | `preprocess.py` | implemented + unit-tested |
| QA filter, reprojection, monthly median | done server-side in Earth Engine | implemented, not run live |
| Data validation | `validate.py` (coverage / range gates, JSON report, shown in dashboard) | implemented + tested |
| CEI (Eq. 8) | `cei.py` | implemented + tested against hand calculation |
| Hotspot classes (adaptive K-means, k=4) | `hotspots.py` | implemented + tested |
| SegFormer-B4 segmentation | – | **not implemented** (see below) |
| BiLSTM forecaster (Fig. 3, MC dropout) | `forecast/bilstm.py` (optional, needs torch) | implemented; trains on tiny data, compared honestly with baseline |
| Baseline forecaster | `forecast/baseline.py` (trend + harmonics OLS) | default, always clearly labelled "BASELINE" |
| DeepSHAP attribution | `explain.py`: *exact additive decomposition of CEI* | substitute, labelled as such |
| FastAPI backend + GIS dashboard | `api/app.py`, `web/` (Leaflet + Chart.js, vendored, works offline except map tiles) | implemented |
| PostGIS storage | NetCDF (xarray) + JSON in `data/processed/` | substituted |

**Not implemented, deliberately:** SegFormer-B4 and DeepSHAP on it. In the paper the K-means labels *are* the training target and SegFormer learns to reproduce them from the same inputs (the paper concedes this circularity in §4). Hotspot classes in this MVP are therefore the K-means labels directly. Training a 64 M-parameter transformer to imitate them, without GPU, real data, or any way to validate, would produce numbers I could not honestly stand behind. None of the paper's accuracy figures (mIoU 0.847, RMSE 0.031, R² 0.921, ...) are reproduced or claimed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Python >= 3.11
pip install -e '.[gee,dev]'                             # add ,torch for the BiLSTM
pytest                                                  # 47 tests, ~6 s, no network needed
```

### Credentials (needed for real data)

1. **Google Earth Engine** (TROPOMI, ERA5-Land, land use, night lights; free for non-commercial use): register a Cloud project for Earth Engine, then
   `earthengine authenticate` and `export EE_PROJECT=<your-project-id>`
   (or `EE_SERVICE_ACCOUNT` + `EE_PRIVATE_KEY_FILE`).
2. **NASA FIRMS**: free MAP_KEY from <https://firms.modaps.eosdis.nasa.gov/api/map_key/> → `export FIRMS_MAP_KEY=...`
3. *(optional)* **EDGAR v8.0** NetCDF (<https://edgar.jrc.ec.europa.eu/>) and **CARB** facility CSV (<https://ww2.arb.ca.gov/mrr-data>, reshaped to columns `latitude,longitude,emissions`).

Details in [docs/data_sources.md](docs/data_sources.md).

## Run

```bash
carbonexus fetch                 # 72 monthly composites -> data/raw/gee/*.npz (resumable)
carbonexus fetch-firms           # fire detections       -> data/raw/firms/
carbonexus run                   # preprocess, CEI, classes, forecasts -> data/processed/analysis.{nc,json}
#   options: --bilstm [--epochs N]   --edgar edgar.nc   --carb facilities.csv
carbonexus serve                 # http://127.0.0.1:8000  (dashboard + /docs for the API)
```

Check the output of `carbonexus run` for `VALIDATION:` lines and open the dashboard's **Data** tab: it shows per-channel coverage, out-of-range counts, the CEI weights actually used, and any inventory cross-check.

### Try the software without credentials (synthetic)

```bash
carbonexus --data-dir /tmp/cn-demo demo-synthetic
carbonexus --data-dir /tmp/cn-demo serve
```
This produces **artificial** fields (blobs near LA / Bay Area / San Joaquin plus noise). The dashboard shows a red banner and the analysis is tagged `provenance = SYNTHETIC`. It exists only to exercise the code; do not read anything into it.

## Dashboard

Map (class or continuous CEI, month slider, region boxes, detected hotspot outlines; click anywhere for a per-cell explanation) · region table · **Trend** · **Forecast** (model choice, horizon, ±1σ band, backtest table) · **Hotspots** (connected High/Very High clusters with area and top driver) · **Why?** (per-gas contributions + meteorological context) · **Data** (validation, weights, inventory check).

API (`/docs` for OpenAPI): `/api/status`, `/api/grid/{YYYY-MM}`, `/api/regions`, `/api/regions/{id}/timeseries`, `/api/hotspots/{YYYY-MM}` (GeoJSON), `/api/explain/{YYYY-MM}?region=|lat=&lon=`, `/api/forecast/{id}?model=baseline|bilstm&horizon=`, `/api/validation`.

## Layout

```
config/carbonexus.toml    every parameter, each tagged [paper] or [assumed]
src/carbonexus/           data/ preprocess cei hotspots explain forecast/ pipeline api/ web/ cli
src/carbonexus/devtools/  synthetic generator (tests + demo only)
tests/                    47 tests
docs/                     methodology.md  data_sources.md  limitations.md
```

More: [methodology & deviations](docs/methodology.md) · [data sources](docs/data_sources.md) · [limitations](docs/limitations.md)
