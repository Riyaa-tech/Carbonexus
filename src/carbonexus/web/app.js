"use strict";
const GAS = {no2: "NO₂", ch4: "CH₄", so2: "SO₂", co: "CO"};
const $ = s => document.querySelector(s);
const S = {months: [], region: null, month: 0, overlay: null, hsLayer: null, regLayer: {}, charts: {}, grid: null,
           classes: [], series: null, req: 0, exq: null, selHs: null, hsCount: null, status: null};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const get = u => fetch(u).then(async r => { if (!r.ok) { let d = ""; try { d = (await r.json()).detail || ""; } catch (_) {} throw new Error(d || `${u} → ${r.status}`); } return r.json(); });
const fmt = (v, d = 2) => v == null || isNaN(v) ? "–" : (+v).toFixed(d);
const chip = c => c >= 0 ? `<i class="chip k${c}">${esc(S.classes[c])}</i>` : `<i class="chip kn">n/a</i>`;
const errBox = (el, e) => { el.innerHTML = `<div class="error">${esc(e.message || e)}</div>`; };
const loading = (el, t = "Loading…") => { el.innerHTML = `<div class="skeleton">${esc(t)}</div>`; };
const classColors = () => [0, 1, 2, 3].map(i => css("--c" + i));
const TABS = ["trend", "fc", "hs", "ex", "val"];

async function init() {
  const st = await get("/api/status");
  if (!st.ready) { $("#setup").style.display = "block"; $("#setupmsg").textContent = st.detail; $("#meta").textContent = "No analysis loaded"; return; }
  S.status = st; S.months = st.months; S.classes = st.class_names; S.bounds = st.bounds;
  $("#app").hidden = false;
  $("#meta").textContent = `${st.months[0]} → ${st.months.at(-1)} · ${st.months.length} monthly composites · ${st.shape[1]}×${st.shape[0]} cells at 0.05° · class labels: ${st.label_mode.replace("_", " ")}`;
  $("#chips").innerHTML = `<span class="pill">Period <b>${esc(st.months[0])} – ${esc(st.months.at(-1))}</b></span>` +
    `<span class="pill">Source <b>${esc(st.provenance)}</b></span>` +
    `<span class="pill ${st.validation_ok ? "ok" : "bad"}">Validation <b>${st.validation_ok ? "passed" : "problems – see Data"}</b></span>`;

  const m = $("#month"); m.max = S.months.length - 1; m.value = S.months.length - 1; S.month = +m.value;
  const [w, s, e, n] = S.bounds;
  S.map = L.map("map").fitBounds([[s, w], [n, e]]);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {attribution: "© OpenStreetMap", opacity: .55, maxZoom: 12}).addTo(S.map);
  S.map.on("mousemove", ev => readout(ev.latlng));
  S.map.on("mouseout", () => { $("#readout").innerHTML = ""; });
  S.map.on("click", ev => explainQuery({lat: ev.latlng.lat, lon: ev.latlng.lng}));
  if (window.ResizeObserver) new ResizeObserver(() => S.map.invalidateSize()).observe($("#map"));
  window.addEventListener("resize", () => S.map.invalidateSize());

  m.oninput = () => { S.month = +m.value; refresh(); };
  $("#prev").onclick = () => step(-1); $("#next").onclick = () => step(1);
  $("#layer").onchange = drawGrid;
  document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => tab(b.dataset.t));
  $("#fcmodel").onchange = $("#fch").onchange = drawForecast;
  if (st.forecast_models.includes("bilstm")) $("#fcmodel").add(new Option("BiLSTM (paper architecture, MC dropout)", "bilstm"));

  S.regions = await get("/api/regions");
  S.regions.forEach(r => {
    const [rw, rs, re, rn] = r.box;
    S.regLayer[r.id] = L.rectangle([[rs, rw], [rn, re]], {color: css("--ink"), weight: 1, opacity: .45, fill: false, dashArray: "4 4"}).addTo(S.map)
      .bindTooltip(() => `${esc(r.name)} · CEI ${fmt(S.cur(r))}`, {sticky: true})
      .on("click", ev => { L.DomEvent.stopPropagation(ev); selectRegion(r.id); });
  });
  await ensureSeries();
  S.region = S.regions[0].id;
  await refresh();
  S.map.invalidateSize();
  const db = dataBounds(); if (db) S.map.fitBounds(db, {padding: [4, 4]});   // frame the classified cells, not the whole bbox
  await selectRegion(S.region, false);
}

function step(d) { const m = $("#month"), v = Math.min(+m.max, Math.max(0, +m.value + d)); if (v !== +m.value) { m.value = v; S.month = v; refresh(); } }

function tab(t) {
  document.querySelectorAll(".tabs button").forEach(b => { const on = b.dataset.t === t; b.classList.toggle("on", on); b.setAttribute("aria-selected", on); });
  TABS.forEach(k => $("#tab-" + k).hidden = k !== t);
  if (t === "val") loadValidation();
  for (const c of Object.values(S.charts)) c.resize();
}

async function refresh() {
  const mo = S.months[S.month], req = ++S.req;
  $("#monthlbl").textContent = mo;
  try {
    const grid = await get("/api/grid/" + mo);
    if (req !== S.req) return;               // a newer month was requested meanwhile
    S.grid = grid; drawGrid();
    const th = grid.thresholds;
    $("#thr").textContent = th.every(v => v != null) ? `Class boundaries this month (CEI): Low < ${fmt(th[0])} ≤ Moderate < ${fmt(th[1])} ≤ High < ${fmt(th[2])} ≤ Very High` : "";
  } catch (e) { $("#thr").textContent = "Map data unavailable: " + e.message; return; }
  S.regions = await get("/api/regions").catch(() => S.regions);
  S.hsCount = null;
  renderRegTable(); renderKpis();
  loadHotspots(req);
  if (S.region) drawTrend();
  if (S.exq) explainQuery(S.exq, true);
}

/* ---------- map ---------- */
// Ordered sequential ramp (ColorBrewer YlOrRd). The class colours (--c0..--c3 in styles.css) are
// four stops of this same ramp, so legend, chips, map and charts share one scale.
const RAMP = ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"];
const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
const RAMP_RGB = RAMP.map(hex);
const rampRgb = k => { k = Math.min(1, Math.max(0, k)); const x = k * (RAMP_RGB.length - 1), i = Math.min(Math.floor(x), RAMP_RGB.length - 2), f = x - i; return RAMP_RGB[i].map((v, j) => Math.round(v + (RAMP_RGB[i + 1][j] - v) * f)); };
const CLASS_ALPHA = [125, 180, 225, 250];   // quieter background classes so hotspots stand out

function drawGrid() {
  const g = S.grid, H = g.labels.length, W = g.labels[0].length, layer = $("#layer").value;
  const cv = document.createElement("canvas"); cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(W, H);
  const flat = g.cei.flat().filter(v => v != null).sort((a, b) => a - b);
  const lo = flat[Math.floor(flat.length * .02)] ?? 0, hi = flat[Math.floor(flat.length * .98)] ?? 1;
  S.ramp = [lo, hi];
  const pal = classColors().map(hex);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = 4 * (y * W + x), c = g.labels[y][x], v = g.cei[y][x];
    if (layer === "class" && c >= 0) img.data.set([...pal[c], CLASS_ALPHA[c]], i);
    if (layer === "cei" && v != null) { const k = Math.min(1, Math.max(0, (v - lo) / (hi - lo || 1))); img.data.set([...rampRgb(k), Math.round(70 + 185 * k)], i); }
  }
  ctx.putImageData(img, 0, 0);
  const [w, s, e, n] = S.bounds;
  if (S.overlay) S.map.removeLayer(S.overlay);
  S.overlay = L.imageOverlay(cv.toDataURL(), [[s, w], [n, e]], {opacity: 1}).addTo(S.map);
  S.overlay.getElement().style.imageRendering = "pixelated";   // one data cell = one block; no interpolation
  drawLegend(layer);
}

function drawLegend(layer) {
  const th = S.grid.thresholds, cols = classColors(), ok = th.every(v => v != null);
  const rng = ["< " + fmt(th[0]), `${fmt(th[0])} – ${fmt(th[1])}`, `${fmt(th[1])} – ${fmt(th[2])}`, "≥ " + fmt(th[2])];
  $("#legend").innerHTML = layer === "class"
    ? `<div class="ttl">CEI class</div>` + S.classes.map((c, i) =>
        `<div class="row"><span class="sw" style="background:${cols[i]}"></span>${esc(c)}<span class="rng">${ok ? rng[i] : ""}</span></div>`).join("")
    : `<div class="ttl">CEI (2nd–98th pct)</div><div class="ramp" style="background:linear-gradient(90deg,${RAMP.join(",")})"></div><div class="ends"><span>${fmt(S.ramp[0])} low</span><span>high ${fmt(S.ramp[1])}</span></div>`;
}

function dataBounds() {
  const g = S.grid, H = g.labels.length, W = g.labels[0].length;
  let x0 = W, x1 = -1, y0 = H, y1 = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (g.labels[y][x] >= 0) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  if (x1 < 0) return null;
  const [w, s, e, n] = S.bounds, dx = (e - w) / W, dy = (n - s) / H;
  return [[n - (y1 + 1) * dy, w + x0 * dx], [n - y0 * dy, w + (x1 + 1) * dx]];
}

function readout(ll) {
  const el = $("#readout"), g = S.grid; if (!g) return;
  const [w, s, e, n] = S.bounds, H = g.labels.length, W = g.labels[0].length;
  const x = Math.floor((ll.lng - w) / (e - w) * W), y = Math.floor((n - ll.lat) / (n - s) * H);
  if (x < 0 || y < 0 || x >= W || y >= H) { el.innerHTML = ""; return; }
  const c = g.labels[y][x], v = g.cei[y][x], pos = `${ll.lat.toFixed(2)}, ${ll.lng.toFixed(2)}`;
  el.innerHTML = c < 0 || v == null ? `${pos} · no data` : `${pos} · CEI <b>${fmt(v)}</b> ${chip(c)}`;
}

/* ---------- KPIs (all derived from API responses) ---------- */
function renderKpis() {
  const lab = S.grid.labels.flat().filter(v => v >= 0), n = lab.length;
  const hi = lab.filter(v => v >= 2).length, vh = lab.filter(v => v === 3).length;
  let top = null;
  S.regions.forEach(r => { const v = S.series?.[r.id]?.cei?.[S.month]; if (v != null && (top === null || v > top.v)) top = {r, v}; });
  const hsCount = S.hsCount;
  const card = (l, v, sub, sm) => `<div class="card kpi"><div class="label">${l}</div><div class="value ${sm ? "sm" : ""}">${v}</div><div class="sub">${sub}</div></div>`;
  $("#kpis").innerHTML =
    card("Analysis month", esc(S.months[S.month]), `${S.month + 1} of ${S.months.length}`) +
    card("Cells High or Very High", n ? (100 * hi / n).toFixed(1) + "%" : "–", `${hi.toLocaleString()} of ${n.toLocaleString()} classified cells`) +
    card("Very High cells", vh.toLocaleString(), n ? (100 * vh / n).toFixed(1) + "% of classified cells" : "") +
    card("Hotspot clusters", hsCount == null ? "…" : hsCount, "connected High / Very High groups") +
    card("Highest regional CEI", top ? fmt(top.v) : "–", top ? `${esc(top.r.name)} · ${chip(S.series[top.r.id].class[S.month])}` : "", false) +
    card("Data validation", S.status.validation_ok ? "Passed" : "Problems", `<a href="#" id="kpi-val" style="color:inherit">${S.status.validation_ok ? "No gate failures" : "Open Data tab"}</a>`, true);
  $("#kpi-val").onclick = ev => { ev.preventDefault(); tab("val"); };
}

/* ---------- regions ---------- */
S.cur = r => S.series?.[r.id]?.cei?.[S.month] ?? null;
S.curClass = r => S.series?.[r.id]?.class?.[S.month] ?? -1;

function renderRegTable() {
  const tb = $("#regtable tbody"); tb.innerHTML = "";
  S.regions.forEach(r => {
    const tr = document.createElement("tr"); tr.dataset.id = r.id; tr.tabIndex = 0; tr.className = r.id === S.region ? "sel" : "";
    tr.innerHTML = `<td>${esc(r.name)}</td><td class="num">${fmt(S.cur(r))}</td><td>${chip(S.curClass(r))}</td><td class="num">${fmt(r.trend_cei_per_year)}</td><td class="num">${esc(r.months_very_high)} / ${S.months.length}</td><td class="num">${esc(r.cells)}</td>`;
    tr.onclick = () => selectRegion(r.id);
    tr.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectRegion(r.id); } };
    tb.appendChild(tr);
  });
}

async function ensureSeries() {
  if (S.series) return;
  S.series = {};
  await Promise.all(S.regions.map(async r => { try { S.series[r.id] = await get(`/api/regions/${r.id}/timeseries`); } catch (_) { /* region without series */ } }));
}

async function selectRegion(id, fit = true) {
  await ensureSeries();
  S.region = id; S.exq = {region: id}; renderRegTable(); renderKpis(); drawTrend(); drawForecast(); explainQuery(S.exq);
  if (fit) { const r = S.regions.find(r => r.id === id), [w, s, e, n] = r.box; S.map.fitBounds([[s, w], [n, e]], {maxZoom: 9}); }
}

/* ---------- charts ---------- */
function chartOn(id, cfg) {
  S.charts[id]?.destroy();
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.color = css("--mute");
  cfg.options = Object.assign({responsive: true, maintainAspectRatio: false, animation: false, interaction: {mode: "index", intersect: false}}, cfg.options);
  S.charts[id] = new Chart($("#" + id), cfg);
}
const axes = () => ({x: {grid: {display: false}, ticks: {maxTicksLimit: 7, maxRotation: 0}, title: {display: true, text: "Month", font: {size: 11}}}, y: {grid: {color: css("--line")}, ticks: {maxTicksLimit: 6}, border: {display: false}, title: {display: true, text: "CEI (index units)", font: {size: 11}}}});
const tip = () => ({backgroundColor: css("--ink"), titleColor: css("--panel"), bodyColor: css("--panel"), padding: 8, cornerRadius: 6, callbacks: {label: c => c.parsed.y == null ? null : `${c.dataset.label}: ${(+c.parsed.y).toFixed(2)}`}});

function drawTrend() {
  const ser = S.series?.[S.region], r = S.regions.find(r => r.id === S.region);
  if (!ser || !r) { $("#regstats").innerHTML = `<div class="error">No time series for this region.</div>`; return; }
  $("#trendttl").textContent = `Historical CEI – ${ser.name}`;
  const acc = css("--accent");
  chartOn("trend", {type: "line", data: {labels: ser.months, datasets: [
    {label: "Region-mean CEI", data: ser.cei, borderColor: acc, backgroundColor: acc + "22", fill: true, pointRadius: 0, borderWidth: 2, tension: .25},
    {label: "Selected month", data: ser.cei.map((v, i) => i === S.month ? v : null), pointRadius: 6, pointBackgroundColor: css("--c3"), pointBorderColor: css("--panel"), pointBorderWidth: 2, showLine: false}]},
    options: {scales: axes(), plugins: {legend: {display: false}, tooltip: tip()}}});
  const cols = classColors(), mix = Object.entries(r.class_share_latest);
  $("#regstats").innerHTML = `<div class="stats">
      <div class="stat"><div class="k">Cells</div><div class="v">${esc(r.cells)}</div></div>
      <div class="stat"><div class="k">Mean CEI</div><div class="v">${fmt(r.mean_cei)}</div></div>
      <div class="stat"><div class="k">Trend per year</div><div class="v">${fmt(r.trend_cei_per_year)}</div></div>
      <div class="stat"><div class="k">Months classed Very High</div><div class="v">${esc(r.months_very_high)} / ${ser.months.length}</div></div></div>
    <h2 class="mini">Class mix of cells, latest month</h2>
    <div class="mix" role="img" aria-label="class mix">${mix.map(([k, v], i) => `<i style="width:${100 * v}%;background:${cols[i]}" title="${esc(k)} ${(100 * v).toFixed(0)}%"></i>`).join("")}</div>
    <div class="mixlbl">${mix.map(([k, v], i) => `<span style="--dot:${cols[i]}">${esc(k)} ${(100 * v).toFixed(0)}%</span>`).join("")}</div>`;
}

async function drawForecast() {
  if (!S.region) return;
  const model = $("#fcmodel").value, h = +$("#fch").value;
  try {
    const f = await get(`/api/forecast/${S.region}?model=${model}&horizon=${h}`);
    const ser = S.series[S.region], hist = ser.cei.slice(-24), hl = ser.months.slice(-24);
    const pad = new Array(hist.length).fill(null), fcol = css("--fc");
    chartOn("fc", {type: "line", data: {labels: hl.concat(f.months), datasets: [
      {label: "Observed", data: hist.concat(new Array(f.mean.length).fill(null)), borderColor: css("--accent"), borderWidth: 2, pointRadius: 0},
      {label: "Forecast mean", data: pad.concat(f.mean), borderColor: fcol, borderWidth: 2, borderDash: [6, 4], pointRadius: 0},
      {label: "±1σ band", data: pad.concat(f.mean.map((m, i) => m + f.std[i])), borderColor: "transparent", backgroundColor: fcol + "33", pointRadius: 0, fill: "+1"},
      {label: "−1σ", data: pad.concat(f.mean.map((m, i) => m - f.std[i])), borderColor: "transparent", pointRadius: 0}]},
      options: {scales: axes(), plugins: {tooltip: tip(), legend: {labels: {boxWidth: 10, filter: i => i.text !== "−1σ"}}}}});
    const last = f.class.at(-1);
    $("#fcnote").innerHTML = `<div class="callout warn"><b>${esc(f.label)}</b><br>Projected class at +${f.mean.length} months: ${last != null ? chip(last) : "n/a"} (mean class boundaries of the last 12 observed months). This is a scenario, not a prediction: the band covers model/residual error only and is not validated beyond 12 months.</div>`;
  } catch (e) { errBox($("#fcnote"), new Error("No forecast for this region: " + e.message)); S.charts.fc?.destroy(); }
  try {
    const v = await get("/api/validation"), bt = v.forecast_backtest || {};
    const row = (n, m) => m && m.rmse != null ? `<tr><td>${n}</td><td class="num">${fmt(m.rmse, 3)}</td><td class="num">${fmt(m.mae, 3)}</td><td class="num">${fmt(m.r2, 3)}</td></tr>` : "";
    $("#fcbt").innerHTML = `<h2>Backtest, last ${esc(bt.holdout_months ?? "?")} months (all regions pooled, computed from this data)</h2><div class="scroll-x"><table><tr><th>Model</th><th class="num">RMSE</th><th class="num">MAE</th><th class="num">R²</th></tr>${row("Baseline", bt.baseline)}${row("Seasonal naive", bt.seasonal_naive)}${row("BiLSTM", bt.bilstm)}</table></div>`;
  } catch (e) { errBox($("#fcbt"), e); }
}

/* ---------- hotspots ---------- */
async function loadHotspots(req) {
  const el = $("#hslist"); loading(el, "Loading hotspots…");
  try {
    const gj = await get("/api/hotspots/" + S.months[S.month]);
    if (req !== undefined && req !== S.req) return;
    S.hsCount = gj.features.length; renderKpis();
    if (S.hsLayer) S.map.removeLayer(S.hsLayer);
    S.hsLayer = L.geoJSON(gj, {style: () => ({color: css("--ink"), weight: .4, opacity: .22, fillOpacity: 0}),
      onEachFeature: (f, l) => l.on("click", ev => { L.DomEvent.stopPropagation(ev); showHotspot(f); })}).addTo(S.map);
    el.innerHTML = gj.features.length ? "" : "<div class='callout'>No High / Very High clusters of the minimum size this month.</div>";
    gj.features.slice(0, 25).forEach((f, idx) => {
      const p = f.properties, d = document.createElement("button"); d.type = "button"; d.className = "hs"; d.dataset.idx = idx;
      d.innerHTML = `${chip(p.max_class)} <b>${esc(p.region ?? "Outside named regions")}</b><div class="meta"><span>${esc(p.cells)} cells</span><span>${Math.round(p.area_km2).toLocaleString()} km²</span><span>mean CEI ${fmt(p.mean_cei)}</span><span>top driver ${GAS[p.top_driver] ?? esc(p.top_driver)}</span></div>`;
      d.onclick = () => { const [w, s, e, n] = p.bbox; S.map.fitBounds([[s, w], [n, e]], {maxZoom: 9}); showHotspot(f, d); };
      el.appendChild(d);
    });
  } catch (e) { S.hsCount = null; errBox(el, e); }
}
function showHotspot(f, card) {
  const p = f.properties; S.exq = null;
  document.querySelectorAll(".hs").forEach(x => x.classList.toggle("sel", x === card));
  renderExplain(p.explanation, p.context, `Hotspot near ${p.centroid.map(v => v.toFixed(2)).join(", ")}`); tab("ex");
}

/* ---------- explanations ---------- */
async function explainQuery(q, quiet) {
  const mo = S.months[S.month], qs = new URLSearchParams(q).toString();
  S.exq = q;
  if (!quiet) loading($("#exbody"), "Loading explanation…");
  try { const e = await get(`/api/explain/${mo}?${qs}`); if (S.exq === q) renderExplain(e.gases, e.context, e.target, e.method); }
  catch (err) { errBox($("#exbody"), new Error("No explanation: " + err.message)); }
}
function renderExplain(gases, ctx, title, method) {
  $("#exttl").textContent = "Contributing factors – " + title;
  const keys = Object.keys(GAS);
  const rows = keys.map(g => { const r = gases[g]; return `<div class="bar"><span>${GAS[g]}</span><div class="track"><i style="width:${Math.round(100 * r.excess_share)}%"></i></div><span class="pct">${(100 * r.excess_share).toFixed(0)}%</span></div>`; }).join("");
  const tbl = keys.map(g => { const r = gases[g]; return `<tr><td>${GAS[g]}</td><td class="num">${fmt(r.contribution)}</td><td class="num">${fmt(r.statewide_median)}</td><td class="num">${fmt(r.excess)}</td></tr>`; }).join("");
  const cx = Object.entries(ctx || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${fmt(v.value, 3)}</td><td class="num">${fmt(v.anomaly_z)}</td></tr>`).join("");
  $("#exbody").innerHTML = `<p class="note">Share of the positive CEI excess over the statewide median, by gas.</p>${rows}
    <div class="scroll-x"><table><tr><th>Gas</th><th class="num">Contribution</th><th class="num">State median</th><th class="num">Excess</th></tr>${tbl}</table></div>
    <h2 style="margin-top:14px">Meteorological context (not part of CEI)</h2>
    <div class="scroll-x"><table><tr><th>Variable</th><th class="num">Value (z)</th><th class="num">Anomaly z vs same calendar month</th></tr>${cx}</table></div>
    <div class="callout">${esc(method || "Exact additive decomposition of CEI; not DeepSHAP.")} Weights and GWP are listed on the Data tab. CH₄ carries a ×28 GWP factor in Eq. 8, which can dominate the index. Contributions describe the index formula, not emission sources.</div>`;
}

/* ---------- validation ---------- */
async function loadValidation() {
  const el = $("#valbody"); loading(el, "Loading validation…");
  try {
    const v = await get("/api/validation"), ch = v.validation_raw.channels, c = v.config.cei;
    const wt = Object.keys(GAS).map(g => `<tr><td>${GAS[g]}</td><td class="num">${esc(c["w_" + g])}</td><td class="num">${esc(c["gwp_" + g])}</td></tr>`).join("");
    const rows = Object.entries(ch).map(([k, x]) => `<tr><td>${esc(k)}</td><td><span class="vbar"><i style="width:${100 * x.valid_fraction_mean}%"></i></span>${(100 * x.valid_fraction_mean).toFixed(0)}%</td><td class="num">${(100 * x.valid_fraction_min).toFixed(0)}%</td><td class="num ${x.out_of_range_values ? "bad-cell" : ""}">${esc(x.out_of_range_values)}</td></tr>`).join("");
    const probs = [...v.validation_raw.problems, ...v.validation_processed.problems];
    const inv = v.inventory_validation
      ? Object.entries(v.inventory_validation).map(([src, d]) => `<h2 style="margin-top:10px">${esc(src.toUpperCase())}</h2><table>${Object.entries(d).map(([k, x]) => `<tr><td>${esc(k.replace(/_/g, " "))}</td><td class="num">${typeof x === "number" ? fmt(x, 3) : esc(x)}</td></tr>`).join("")}</table>`).join("")
      : "<div class='callout'>No EDGAR / CARB inventory cross-check was run (needs manual downloads; see docs).</div>";
    el.innerHTML = `<div class="callout">Provenance: <b>${esc(v.provenance)}</b></div>
      ${probs.length ? `<div class="error"><b>Problems</b><ul style="margin:4px 0 0 18px;padding:0">${probs.map(p => `<li>${esc(p)}</li>`).join("")}</ul></div>` : `<div class="callout">No validation problems flagged.</div>`}
      <h2 style="margin-top:12px">CEI weights (Eq. 8)</h2>
      <table><tr><th>Gas</th><th class="num">w<sub>g</sub></th><th class="num">GWP<sub>g</sub></th></tr>${wt}</table>
      <p class="note">GWP-100 of CH₄ = 28 is from the paper; GWP of NO₂/CO/SO₂ is not given there and is assumed 1.</p>
      <h2 style="margin-top:12px">Channel coverage (in-state cells)</h2>
      <div class="scroll-x"><table><tr><th>Channel</th><th>Valid (mean)</th><th class="num">Worst month</th><th class="num">Out of range</th></tr>${rows}</table></div>
      <h2 style="margin-top:12px">Inventory cross-check</h2>${inv}`;
  } catch (e) { errBox(el, e); }
}

init().catch(e => { $("#setup").style.display = "block"; $("#setupmsg").textContent = e.message; $("#meta").textContent = "Unable to load"; });
