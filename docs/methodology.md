# Methodology: what follows the paper, what does not

Parameters live in `config/carbonexus.toml`; each is tagged `[paper]` or `[assumed]`.

## 1. Data and grid
* Study area/period: California, 2019-01 … 2024-12, 72 months, 0.05° grid, **190 lat × 240 lon** `[paper]`. The paper gives the grid size but not the bounding box; I use west −124.5, north 42.0 (→ south 32.5, east −112.5) `[assumed]`, which covers California plus parts of neighbouring states. All statistics are restricted to cells inside a hand-drawn **coarse California outline** (`grid.py`), so that K-means classes describe California only `[assumed]`.
* Channels: 4 gases (NO₂, CH₄, SO₂, CO), 5 ERA5 (u10, v10, t2m, total precipitation, surface pressure), 3 land-use. The paper does not say which three land-use indicators; I use **ESA WorldCover built-up fraction, WorldCover cropland fraction, and VIIRS monthly night-light radiance** (the paper mentions night lights as an industrial-activity proxy) `[assumed]`. The paper's 13th channel, the CEI, is computed, not fetched.
* ERA5: the paper says ~11 km monthly means, i.e. **ERA5-Land** (`ECMWF/ERA5_LAND/MONTHLY_AGGR`), which is land-only (NaN offshore).

## 2. Preprocessing (paper §2.2)
| Step | Implementation |
|---|---|
| QA filter (NO₂ > 0.75, CH₄ > 0.5) | applied by the Earth Engine L3 ingestion of the OFFL products (the `qa_value` band is not exposed); other gases use the 0.5 default. Not independently re-checkable here. |
| Reprojection, bilinear | `computePixels` onto the study grid, `resample('bilinear')` |
| Monthly **median** composite | `ImageCollection.median()` over the month |
| MAD outlier clip, Eq. 1 | per-pixel time series, replace by pixel median (`mad_clip`) |
| Kalman gap-fill, Eq. 2-6 | scalar random-walk filter per pixel, Q=1e-4, R=1e-2 `[paper]`. **Additions:** a pixel starts at its first observation (leading NaNs stay NaN) and gaps longer than 6 months are set back to NaN to avoid the "manufactured trend" the paper warns about `[assumed]` |
| Savitzky-Golay along time | window 5, order 2 `[assumed]` (not given) |
| Robust z-score, Eq. 7 | global per channel (median/1.4826·MAD over all valid pixels and months), so CEI is comparable across months |
| Wildfire masking | NO₂ and CO NaN'd in cell-months with a VIIRS detection. Only `type==0` (vegetation fire) and non-low-confidence detections count, so industrial static sources (type 2) are not masked `[assumed]` |

**Order deviation:** the paper lists normalisation last, but Q=1e-4 / R=1e-2 only make sense for unit-scale data (raw NO₂ is ~1e-4 mol/m²). I normalise first, then clip → Kalman → smooth. The fire mask is applied last (as in the paper) and is *not* followed by re-filling.

## 3. Carbon Emission Index, Eq. 8
`CEI = Σ_g w_g·GWP_g·z_g / Σ_g w_g`, w = NO₂ 0.30, CO 0.20, CH₄ 0.35, SO₂ 0.15 `[paper]`, GWP-100(CH₄) = 28 `[paper]`.

Things the paper leaves open, and what I did:
* **GWP for NO₂, CO, SO₂ is not given** (the text calls NO₂ a "proxy"). I set them to 1 `[assumed]`.
* **Consequence:** CH₄ gets weight 0.35·28 = 9.8 versus ≤ 0.30 for any other gas, so the CEI is mathematically dominated by methane variability wherever the CH₄ field varies at all. That is hard to reconcile with the paper's reported attribution (LA Basin ≈68 % on-road combustion via NO₂/CO). Whether the authors rescaled something, I cannot tell from the text. I implemented the formula literally and made GWP configurable (`[cei]` in the TOML); the dashboard "Why?" tab states the CH₄ factor. If you prefer a proxy-unit variant, set `gwp_ch4 = 1.0` and re-run.
* **Scale:** the paper's CEI values (0.43-0.81) lie in [0, 1]; the formula above produces unbounded robust-z units. No rescaling is specified, so none is applied, and numbers in this app are **not comparable** to Tables 2-3 of the paper.
* Missing gases (QA gaps, fire-masked NO₂/CO): sums run over the gases that are present, so the denominator is the weight sum of those gases `[assumed]`.

## 4. Hotspot classes (paper §2.8)
K-means, k=4, k-means++, `n_init=10`, `random_state=42`, clusters ordered low→high `[paper]`. The paper's text says per-month clustering, while its code listing (Fig. 5) clusters the **time-mean** CEI. Both are supported: `label_mode = "per_month"` (default, text) or `"mean"` (code). Classes are **relative** to the distribution of the month, not an absolute emission level; "Very High" always exists, however clean the month. Regional classes place the regional mean CEI on that month's class boundaries (midpoints of the cluster centres). Connected components (8-neighbour) of High/Very High cells with ≥ 3 cells are listed as hotspots `[assumed]`.

## 5. Forecasting
* **Baseline (default):** OLS fit of intercept + linear trend + two annual harmonics on the regional mean CEI; analytic 1σ prediction intervals (assume Gaussian i.i.d. residuals; not validated beyond the 12-month backtest). Backtest: fit on all but the last 12 months, score the 12 held-out months pooled over regions; a seasonal-naive reference (value 12 months earlier) is reported alongside. Class of a forecast uses the mean class boundaries of the last 12 observed months.
* **BiLSTM (`carbonexus run --bilstm`, needs torch):** the architecture of Fig. 3 (2 × bidirectional LSTM, hidden 128, LayerNorm, dropout 0.15, GELU head), T_in = 24 → T_out = 12, 13 features, MSE, AdamW (lr 5e-4, wd 1e-5), MC-dropout (50 samples) `[paper]`. Deviations: trained on **region-aggregated** series pooled over the 8 regions (paper: "city-level"), so training has 13 windows per region; early stopping on the validation windows `[assumed]`; during the autoregressive 5-year roll-out future ERA5/land-use features are replaced by training-period monthly climatology `[assumed]`; the reported std is MC-dropout of the last round only and understates long-horizon uncertainty. Its test metrics are computed on the same final 12 months as the baseline and shown side by side. With ≈ 100 training windows it will often *not* beat the baseline; the dashboard shows whichever result the data gives.
* The paper's "temporal consistency" metric is undefined in the text and is not computed. MAPE is not reported because CEI crosses zero.
* **Not done:** the fusion module of Fig. 1, SegFormer-B4 (§2.5), patch extraction, logit-averaged sliding-window inference.

## 6. Explainability
CEI is linear in the normalised columns, so the contribution of gas *g* to a cell's CEI, `w_g·GWP_g·z_g/Σw`, is exactly its Shapley value for that model (baseline 0). The app reports contributions, their excess over the statewide median of the same gas in that month, and each gas's share of the positive excess. This answers "which gas pushed this cell/region/hotspot up" faithfully, **but** it is a property of the index formula, not evidence about emission *sources*. The paper's source attributions (on-road 68 %, dairy 65 %) come from SHAP over a trained network and are not reproduced. Meteorology and land use are displayed as context (anomaly z-scores), not as attributions, because they do not enter CEI.

## 7. Validation against inventories
`carbonexus run --edgar … --carb …` computes Spearman ρ between time-mean CEI and the inventory and the share of inventory emissions falling in High/Very High cells vs the area share of Very High. Nothing is retrained from the inventories (the paper uses them only for cross-checking the labels).
