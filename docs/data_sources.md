# Data sources and API configuration

| Quantity | Source | Access used here | Credentials |
|---|---|---|---|
| NO₂ tropospheric column | Sentinel-5P/TROPOMI L3 OFFL, `COPERNICUS/S5P/OFFL/L3_NO2`, band `tropospheric_NO2_column_number_density` | Google Earth Engine | GEE account + Cloud project |
| CH₄ (XCH₄, bias-corrected) | `COPERNICUS/S5P/OFFL/L3_CH4`, `CH4_column_volume_mixing_ratio_dry_air_bias_corrected` | GEE | same |
| SO₂ column | `COPERNICUS/S5P/OFFL/L3_SO2`, `SO2_column_number_density` | GEE | same |
| CO column | `COPERNICUS/S5P/OFFL/L3_CO`, `CO_column_number_density` | GEE | same |
| Meteorology (u10, v10, t2m, tp, sp) | ERA5-Land monthly aggregates, `ECMWF/ERA5_LAND/MONTHLY_AGGR` (~11 km) | GEE | same |
| Built-up / cropland fraction | ESA WorldCover v200 (10 m, 2021), classes 50 / 40, averaged to 0.05° | GEE | same |
| Night-light radiance | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`, `avg_rad` | GEE | same |
| Fire detections | NASA FIRMS VIIRS S-NPP standard processing (`VIIRS_SNPP_SP`) | FIRMS area API, 5-day chunks | free `MAP_KEY` |
| Gridded inventory | EDGAR v8.0, 0.1° NetCDF | manual download | none |
| Facility inventory | CARB mandatory GHG reporting | manual download | none |

All Earth Engine layer ids and band names were written from the public catalog from memory of the documentation and **have not been verified against the live catalog**; if Earth Engine reports a missing band or collection, adjust the constants at the top of `src/carbonexus/data/gee.py` (they are the only place these names appear).

### Why Earth Engine rather than raw Copernicus/CDS downloads
The paper itself exports composites from Earth Engine (§2.9a); it is the only route that gives monthly gridded TROPOMI + ERA5 for 72 months without handling terabytes of L2 swaths. Direct ESA Copernicus Data Space (L2 NetCDF) and the Climate Data Store (ERA5) are viable alternatives but need separate accounts and a regridding step that is not implemented here. TROPOMI L3 products on GEE are pre-gridded by Earth Engine's own ingestion (harpconvert), not by the authors.

### Configuration reference
| Variable | Meaning |
|---|---|
| `EE_PROJECT` | Earth Engine Cloud project id (required for `carbonexus fetch`) |
| `EE_SERVICE_ACCOUNT`, `EE_PRIVATE_KEY_FILE` | optional service-account auth (otherwise uses `earthengine authenticate`) |
| `FIRMS_MAP_KEY` | NASA FIRMS key (required for `carbonexus fetch-firms`) |
| `CARBONEXUS_DATA` or `--data-dir` | data directory (default `./data`) |

### Storage layout
```
data/raw/gee/YYYY-MM.npz            one file per month, 12 channels, NaN = no data
data/raw/firms/viirs_snpp_wildfire.csv.gz
data/processed/analysis.nc          CEI, class labels, per-gas contributions, fire counts, met z-scores, class centres/thresholds
data/processed/analysis.json        regional series/stats, forecasts, validation report, backtest, config snapshot
```

### Expected request volume
72 `computePixels` calls (each ≈ 2.4 MB) and ≈ 440 FIRMS calls. Both fetchers cache and resume.

### Inventory file formats
* EDGAR: any NetCDF with a `lat`/`lon` data variable (summed over any extra dimension). Units are not converted; only rank correlation and shares are reported.
* CARB facilities: CSV with `latitude,longitude,emissions` (reshape the CARB download accordingly).
