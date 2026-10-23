"use strict";
const COLORS = ["#5aa469", "#f2c14e", "#e8743b", "#b3202a"];
const GAS = {no2: "NO₂", ch4: "CH₄", so2: "SO₂", co: "CO"};
const $ = s => document.querySelector(s);
const S = {months: [], region: null, month: 0, overlay: null, hsLayer: null, regLayer: {}, charts: {}, grid: null, classes: []};
const get = u => fetch(u).then(r => { if (!r.ok) throw new Error(u + " " + r.status); return r.json(); });
const chip = c => c >= 0 ? `<i class="chip k${c}">${S.classes[c]}</i>` : `<i class="chip kn">n/a</i>`;
const fmt = (v, d = 2) => v == null || isNaN(v) ? "–" : (+v).toFixed(d);

async function init() {
  const st = await get("/api/status");
  if (!st.ready) { $("#setup").style.display = "block"; $("#setupmsg").textContent = st.detail; return; }
  $("#app").style.display = "grid";
  S.months = st.months; S.classes = st.class_names; S.bounds = st.bounds;
  $("#meta").textContent = `${st.months[0]} → ${st.months.at(-1)} · ${st.shape[1]}×${st.shape[0]} cells @0.05° · labels: ${st.label_mode}` + (st.validation_ok ? "" : " · ⚠ validation problems (see Data tab)");
  if (/SYNTHETIC/i.test(st.provenance)) { const b = $("#banner"); b.style.display = "block"; b.textContent = "⚠ SYNTHETIC DEMO DATA – artificial values for software testing only. Not a measurement."; }
  const m = $("#month"); m.max = S.months.length - 1; m.value = S.months.length - 1; S.month = +m.value;
  const [w, s, e, n] = S.bounds;
  S.map = L.map("map").fitBounds([[s, w], [n, e]]);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {attribution: "© OpenStreetMap", opacity: .6, maxZoom: 12}).addTo(S.map);
  S.map.on("click", ev => explain({lat: ev.latlng.lat, lon: ev.latlng.lng}));
  m.oninput = () => { S.month = +m.value; refresh(); };
  $("#layer").onchange = drawGrid;
  document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => tab(b.dataset.t));
  $("#fcmodel").onchange = $("#fch").onchange = drawForecast;
  if (st.forecast_models.includes("bilstm")) $("#fcmodel").add(new Option("BiLSTM (paper architecture, MC dropout)", "bilstm"));
  S.regions = await get("/api/regions");
  S.regions.forEach(r => {
    const [rw, rs, re, rn] = r.box;
    S.regLayer[r.id] = L.rectangle([[rs, rw], [rn, re]], {color: "#333", weight: 1, fill: false, dashArray: "4"}).addTo(S.map)
      .on("click", ev => { L.DomEvent.stopPropagation(ev); selectRegion(r.id); });
  });
  await ensureSeries();
  await refresh();
  selectRegion(S.regions[0].id);
}

function tab(t) {
  document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("on", b.dataset.t === t));
  ["trend", "fc", "hs", "ex", "val"].forEach(k => $("#tab-" + k).style.display = k === t ? "block" : "none");
  if (t === "val") loadValidation();
}

async function refresh() {
  const mo = S.months[S.month];
  $("#monthlbl").textContent = mo;
  S.grid = await get("/api/grid/" + mo);
  drawGrid();
  const th = S.grid.thresholds;
  $("#thr").textContent = th.every(v => v != null) ? "Class boundaries (CEI): " + th.map(v => v.toFixed(2)).join(" | ") : "";
  S.regions = await get("/api/regions");
  renderRegTable();
  loadHotspots();
  if (S.region) drawTrend();
}

const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
const hsl = (h, s, l) => { s /= 100; l /= 100; const k = n => (n + h / 30) % 12, a = s * Math.min(l, 1 - l), f = n => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1))); return [f(0), f(8), f(4)].map(v => Math.round(255 * v)); };

function drawGrid() {
  const g = S.grid, H = g.labels.length, W = g.labels[0].length, layer = $("#layer").value;
  const cv = document.createElement("canvas"); cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(W, H);
  const flat = g.cei.flat().filter(v => v != null).sort((a, b) => a - b);
  const lo = flat[Math.floor(flat.length * .02)] ?? 0, hi = flat[Math.floor(flat.length * .98)] ?? 1;
  const pal = COLORS.map(hex);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = 4 * (y * W + x), c = g.labels[y][x], v = g.cei[y][x];
    let rgb = null;
    if (layer === "class" && c >= 0) rgb = pal[c];
    if (layer === "cei" && v != null) { const k = Math.min(1, Math.max(0, (v - lo) / (hi - lo || 1))); rgb = hsl(120 - 120 * k, 60, 60 - 20 * k); }
    if (rgb) img.data.set([rgb[0], rgb[1], rgb[2], 200], i);
  }
  ctx.putImageData(img, 0, 0);
  const [w, s, e, n] = S.bounds;
  if (S.overlay) S.map.removeLayer(S.overlay);
  S.overlay = L.imageOverlay(cv.toDataURL(), [[s, w], [n, e]], {opacity: .85}).addTo(S.map);
  S.overlay.getElement().style.imageRendering = "pixelated";
}

function renderRegTable() {
  const tb = $("#regtable tbody"); tb.innerHTML = "";
  S.regions.forEach(r => {
    const tr = document.createElement("tr"); tr.dataset.id = r.id; tr.className = r.id === S.region ? "sel" : "";
    tr.innerHTML = `<td>${r.name}</td><td>${fmt(S.cur(r))}</td><td>${chip(S.curClass(r))}</td>`;
    tr.onclick = () => selectRegion(r.id); tb.appendChild(tr);
  });
}
S.cur = r => (S.series?.[r.id]?.cei?.[S.month]) ?? (S.month === S.months.length - 1 ? r.latest_cei : null);
S.curClass = r => (S.series?.[r.id]?.class?.[S.month]) ?? (S.month === S.months.length - 1 ? r.latest_class : -1);

async function ensureSeries() {
  if (S.series) return;
  S.series = {};
  await Promise.all(S.regions.map(async r => S.series[r.id] = await get(`/api/regions/${r.id}/timeseries`)));
}

async function selectRegion(id) {
  await ensureSeries();
  S.region = id; renderRegTable(); drawTrend(); drawForecast(); explain({region: id});
  const r = S.regions.find(r => r.id === id), [w, s, e, n] = r.box;
  S.map.fitBounds([[s, w], [n, e]], {maxZoom: 9});
}

function chartOn(id, cfg) {
  S.charts[id]?.destroy();
  S.charts[id] = new Chart($("#" + id), cfg);
}

async function drawTrend() {
  await ensureSeries();
  const ser = S.series[S.region], r = S.regions.find(r => r.id === S.region);
  $("#trendttl").textContent = `Historical CEI – ${ser.name}`;
  chartOn("trend", {type: "line", data: {labels: ser.months, datasets: [
    {label: "Region-mean CEI", data: ser.cei, borderColor: "#1f6f54", pointRadius: 0, borderWidth: 2, spanGaps: false},
    {label: "Selected month", data: ser.cei.map((v, i) => i === S.month ? v : null), pointRadius: 6, pointBackgroundColor: "#b3202a", showLine: false}]},
    options: {animation: false, scales: {x: {ticks: {maxTicksLimit: 7}}}, plugins: {legend: {display: false}}}});
  $("#regstats").innerHTML = `Cells: ${r.cells} · mean CEI ${fmt(r.mean_cei)} · trend ${fmt(r.trend_cei_per_year)} CEI/yr · months classed Very High: ${r.months_very_high}/${ser.months.length}<br>Latest class mix: ` +
    Object.entries(r.class_share_latest).map(([k, v]) => `${k} ${(100 * v).toFixed(0)}%`).join(" · ");
}

async function drawForecast() {
  if (!S.region) return;
  const model = $("#fcmodel").value, h = +$("#fch").value;
  try {
    const f = await get(`/api/forecast/${S.region}?model=${model}&horizon=${h}`);
    const ser = S.series[S.region];
    const hist = ser.cei.slice(-24), hl = ser.months.slice(-24);
    const labels = hl.concat(f.months);
    const pad = new Array(hist.length).fill(null);
    chartOn("fc", {type: "line", data: {labels, datasets: [
      {label: "Observed", data: hist.concat(new Array(f.mean.length).fill(null)), borderColor: "#1f6f54", pointRadius: 0},
      {label: "Forecast mean", data: pad.concat(f.mean), borderColor: "#e8743b", pointRadius: 0, borderDash: [5, 3]},
      {label: "+1σ", data: pad.concat(f.mean.map((m, i) => m + f.std[i])), borderColor: "rgba(232,116,59,.3)", pointRadius: 0, fill: "+1"},
      {label: "−1σ", data: pad.concat(f.mean.map((m, i) => m - f.std[i])), borderColor: "rgba(232,116,59,.3)", pointRadius: 0}]},
      options: {animation: false, scales: {x: {ticks: {maxTicksLimit: 7}}}, plugins: {legend: {labels: {boxWidth: 10}}}}});
    const last = f.class.at(-1);
    $("#fcnote").innerHTML = `<b>${f.label}</b><br>Projected class at +${f.mean.length} mo: ${last != null ? chip(last) : "n/a"} (using mean class boundaries of the last 12 observed months). Scenario, not a prediction: uncertainty bands cover model/residual error only.`;
  } catch (e) { $("#fcnote").textContent = "No forecast for this region: " + e.message; chartOn("fc", {type: "line", data: {labels: [], datasets: []}}); }
  const v = await get("/api/validation");
  const bt = v.forecast_backtest || {};
  const row = (n, m) => m && m.rmse != null ? `<tr><td>${n}</td><td>${fmt(m.rmse, 3)}</td><td>${fmt(m.mae, 3)}</td><td>${fmt(m.r2, 3)}</td></tr>` : "";
  $("#fcbt").innerHTML = `<b>Backtest on last ${bt.holdout_months ?? "?"} months (computed here, all regions pooled)</b><table><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th></tr>${row("Baseline", bt.baseline)}${row("Seasonal naive", bt.seasonal_naive)}${row("BiLSTM", bt.bilstm)}</table>`;
}

async function loadHotspots() {
  const gj = await get("/api/hotspots/" + S.months[S.month]);
  if (S.hsLayer) S.map.removeLayer(S.hsLayer);
  S.hsLayer = L.geoJSON(gj, {style: f => ({color: "#000", weight: .5, opacity: .35, fillOpacity: 0}), onEachFeature: (f, l) => l.on("click", ev => { L.DomEvent.stopPropagation(ev); showHotspot(f, l); })}).addTo(S.map);
  const el = $("#hslist"); el.innerHTML = gj.features.length ? "" : "<p class='note'>No High/Very High clusters ≥ minimum size this month.</p>";
  gj.features.slice(0, 25).forEach(f => {
    const p = f.properties, d = document.createElement("div"); d.className = "hs";
    d.innerHTML = `${chip(p.max_class)} <b>${p.region ?? "Outside named regions"}</b><br><span class="note">${p.cells} cells · ${Math.round(p.area_km2)} km² · mean CEI ${fmt(p.mean_cei)} · top driver ${GAS[p.top_driver]}</span>`;
    d.onclick = () => { const [w, s, e, n] = p.bbox; S.map.fitBounds([[s, w], [n, e]], {maxZoom: 9}); renderExplain(p.explanation, p.context, `Hotspot near ${p.centroid.map(v => v.toFixed(2)).join(", ")}`); tab("ex"); };
    el.appendChild(d);
  });
}
function showHotspot(f) { const p = f.properties; renderExplain(p.explanation, p.context, `Hotspot near ${p.centroid.map(v => v.toFixed(2)).join(", ")}`); tab("ex"); }

async function explain(q) {
  const mo = S.months[S.month], qs = new URLSearchParams(q).toString();
  try { const e = await get(`/api/explain/${mo}?${qs}`); renderExplain(e.gases, e.context, e.target, e.method); }
  catch (err) { $("#exbody").textContent = "No explanation: " + err.message; }
}
function renderExplain(gases, ctx, title, method) {
  $("#exttl").textContent = "Contributing factors – " + title;
  const rows = Object.keys(GAS).map(g => { const r = gases[g]; return `<div class="bar"><span class="n">${GAS[g]}</span><i style="width:${Math.round(180 * r.excess_share)}px"></i><span>${(100 * r.excess_share).toFixed(0)}% of excess</span></div>`; }).join("");
  const tbl = Object.keys(GAS).map(g => { const r = gases[g]; return `<tr><td>${GAS[g]}</td><td>${fmt(r.contribution)}</td><td>${fmt(r.statewide_median)}</td><td>${fmt(r.excess)}</td></tr>`; }).join("");
  const cx = Object.entries(ctx || {}).map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v.value, 3)}</td><td>${fmt(v.anomaly_z)}</td></tr>`).join("");
  $("#exbody").innerHTML = `<p class="note">Share of the positive CEI excess over the statewide median, by gas.</p>${rows}
    <table><tr><th>Gas</th><th>Contribution</th><th>State median</th><th>Excess</th></tr>${tbl}</table>
    <h2 style="margin-top:12px">Meteorological context (not part of CEI)</h2><table><tr><th>Var</th><th>Value (z)</th><th>Anomaly z vs same calendar month</th></tr>${cx}</table>
    <p class="note">${method || "Exact additive decomposition of CEI; not DeepSHAP."} The weights/GWP behind these numbers: see Data tab. CH₄ carries a ×28 GWP factor in Eq. 8, which can dominate the index.</p>`;
}

async function loadValidation() {
  const v = await get("/api/validation");
  const ch = v.validation_raw.channels;
  const rows = Object.entries(ch).map(([k, c]) => `<tr><td>${k}</td><td>${(100 * c.valid_fraction_mean).toFixed(0)}%</td><td>${(100 * c.valid_fraction_min).toFixed(0)}%</td><td>${c.out_of_range_values}</td></tr>`).join("");
  const probs = [...v.validation_raw.problems, ...v.validation_processed.problems];
  const inv = v.inventory_validation ? "<pre>" + JSON.stringify(v.inventory_validation, null, 1) + "</pre>" : "<p class='note'>No EDGAR/CARB inventory validation was run (needs manual downloads; see docs).</p>";
  const c = v.config.cei, wt = ["no2", "ch4", "so2", "co"].map(g => `<tr><td>${GAS[g]}</td><td>${c["w_" + g]}</td><td>${c["gwp_" + g]}</td></tr>`).join("");
  $("#valbody").innerHTML = `<h2>CEI weights (Eq. 8)</h2><table><tr><th>Gas</th><th>w<sub>g</sub></th><th>GWP<sub>g</sub></th></tr>${wt}</table><p class="note">GWP-100 of CH₄ = 28 is from the paper; GWP of NO₂/CO/SO₂ is not given there and is assumed 1.</p><p>Provenance: <b>${v.provenance}</b></p><table><tr><th>Channel</th><th>Valid (mean)</th><th>Valid (min month)</th><th>Out-of-range</th></tr>${rows}</table>
   <p>${probs.length ? "<b>Problems:</b><ul><li>" + probs.join("</li><li>") + "</li></ul>" : "No validation problems flagged."}</p><h2>Inventory cross-check</h2>${inv}`;
}
init().catch(e => { $("#setup").style.display = "block"; $("#setupmsg").textContent = e.message; });
