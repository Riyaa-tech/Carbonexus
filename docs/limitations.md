# Limitations (honest list)

## What was and was not verified
* **No real data was fetched.** The development sandbox's egress proxy blocked Earth Engine data access, Copernicus, NASA FIRMS, CDS, EDGAR and CARB hosts. The fetchers (`data/gee.py`, `data/firms.py`) were verified only through unit tests with a fake `ee` object (collection ids, month window, grid definition), a synthetic structured array for response parsing, and sample FIRMS CSV text. They may need small fixes against the live services (band names, `computePixels` return details, FIRMS response format).
* The dashboard was exercised in headless Chromium against **synthetic** data only (map overlay, charts, tabs); base-map tiles were not reachable there.
* **No accuracy metric from the paper is reproduced.** The only metrics the code produces are its own forecast backtests, computed at run time from whatever data is loaded.

## Methodological
* **Circular labels.** Classes come from K-means on the CEI that is computed from the same inputs (the paper acknowledges this). Agreement with CARB/EDGAR is a plausibility check, not ground truth.
* **Relative classes.** "Very High" is the top K-means cluster of the month, not a regulatory or absolute level. A uniformly clean month still has a Very High class.
* **CEI is a z-score composite, not an emissions estimate.** Column densities ≠ emissions: wind transport, chemistry, surface albedo and retrieval bias move signals away from sources (TROPOMI NO₂ is biased low versus ground truth).
* **GWP ambiguity** (see methodology §3): with literal paper weights, methane dominates CEI; the paper's CEI scale ([0,1]) is not reproducible from the text.
* **Heavy tails.** A global robust z-score gives point sources very large values (tens to hundreds of σ). K-means on such a distribution places most cells in Low/Moderate and a few in High/Very High. The paper does not discuss this; no transform (log, clipping) is applied.
* **Resolution.** TROPOMI footprints are ~3.5×5.5 km (CH₄ 7×7 km); a 0.05° cell cannot separate neighbouring sources. ERA5-Land is ~11 km, bilinearly resampled, i.e. smoother than the cell size suggests.
* **Monthly medians** discard weekday/weekend and day/night structure; persistent winter cloud leaves long gaps, which Kalman filling only partly bridges (capped at 6 months).
* **Fire mask** removes NO₂/CO for the whole cell-month with any detection, which also lowers the CEI there (fewer gases averaged); it is not re-imputed.
* **Hand-drawn geography.** The California outline and region boxes are coarse approximations (San Francisco Bay is not excluded; "San Joaquin Valley" is a rectangle that includes foothills). Regional statistics are indicative only.
* **Forecasting.** 72 months per region is tiny. The baseline is a simple OLS trend+season model whose analytic intervals assume Gaussian i.i.d. residuals and are unvalidated beyond 12 months; five-year baseline numbers are extrapolation. The BiLSTM is trained on ≈13 windows per region; MC-dropout is not a calibrated predictive distribution, and its autoregressive std ignores error propagation. Treat all forecasts as scenarios (as the paper itself says).
* **Explainability** is an exact decomposition of an additive index, not source apportionment and not DeepSHAP on a deep model.

## Engineering
* Storage is NetCDF/JSON, not PostGIS. The whole analysis (~100 MB for 72×190×240) is loaded into memory by the API; fine for one state, not for many users.
* Single-user local server, no authentication.
* Python-level loops in Savitzky-Golay handle pixels with leftover NaNs; fine at this size (seconds), slow for much larger grids.
* Not installed/tested: the `[gee]` extra (earthengine-api).
