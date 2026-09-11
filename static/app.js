/* Delta-V prototype. Binary field transport and original physical orientation retained. */
"use strict";
const $ = (id) => document.getElementById(id);
const CH_CANON = ["scalar", "pressure", "velocity_x", "velocity_y"];
const DIVERGING = new Set(["velocity_x", "velocity_y"]);
const PRED_HW = 256;
const S = {
  systems: [],
  samples: [],
  active: null,
  sample: null,
  frames: null,
  shape: null,
  pred: null,
  err: null,
  scales: null,
  errScales: null,
  quality: null,
  ch: 0,
  tIdx: 0,
  dialPos: {},
  epoch: null,
  runTime: null,
  loading: false,
  running: false,
  loadVersion: 0,
  paramVersion: 0,
};
const HISTORY_KEY = "deltav-runs-v1";
let page = "home",
  playback = null,
  tourStep = -1,
  pendingSystem = null;
const escapeHtml = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const fmtSci = (v) =>
  !Number.isFinite(v)
    ? "—"
    : v === 0
      ? "0"
      : Math.abs(v) < 0.001 || Math.abs(v) >= 10000
        ? v.toExponential(2).replace("e+", "e")
        : Number(v.toPrecision(4)).toString();
const pretty = (value) => String(value).replaceAll("_", " ");
const pct = (v) => (Number.isFinite(v) ? (v * 100).toFixed(1) + "%" : "—");
const dialLabel = (sys, d) => sys.dial_meta?.[d]?.description || pretty(d);
function notice(message) {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}
function storageGet(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}
let history = storageGet(HISTORY_KEY, []);
if (!Array.isArray(history)) history = [];
history = history
  .filter(
    (r) =>
      r &&
      typeof r.id === "string" &&
      typeof r.system === "string" &&
      typeof r.date === "string" &&
      ["field", "parametric"].includes(r.kind),
  )
  .slice(0, 20);
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const next = theme === "dark" ? "light" : "dark";
  $("theme").textContent = theme === "dark" ? "☼" : "☾";
  $("theme").setAttribute("aria-label", "Switch to " + next + " mode");
  $("settings-theme").textContent = "Switch to " + next + " mode";
  try {
    localStorage.setItem("deltav-theme", JSON.stringify(theme));
  } catch {}
}
function toggleTheme() {
  setTheme(
    document.documentElement.dataset.theme === "dark" ? "light" : "dark",
  );
}
function route() {
  const hash = location.hash.slice(1);
  page = [
    "simulator",
    "history",
    "datasets",
    "models",
    "settings",
    "about",
  ].includes(hash)
    ? hash
    : "home";
  document
    .querySelectorAll(".page")
    .forEach((el) => (el.hidden = el.id !== "page-" + page));
  $("workspace").hidden = page === "home";
  document.querySelectorAll("[data-nav]").forEach((el) => {
    el.classList.toggle("active", el.dataset.nav === page);
    if (el.dataset.nav === page) el.setAttribute("aria-current", "page");
    else el.removeAttribute("aria-current");
  });
  document.title =
    "Delta-V — " +
    (page === "home" ? "Explore learned fluid dynamics" : pretty(page));
  if (page !== "simulator") stopPlayback();
  if (page === "simulator" && S.frames) requestAnimationFrame(renderStage);
  window.scrollTo(0, 0);
}
async function request(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    signal: AbortSignal.timeout(120000),
  });
  if (!response.ok) {
    const e = new Error(
      response.status === 503
        ? "The model is still warming up. Please try again shortly."
        : response.status === 422
          ? "That parameter combination is not supported by the table. Try values closer to a bundled sample."
          : response.status === 404
            ? "This sample or system is unavailable. Choose another one."
            : "The service could not complete this request. Please try again.",
    );
    throw e;
  }
  return response;
}
function parseNpy(buf) {
  const u8 = new Uint8Array(buf);
  if (String.fromCharCode(...u8.slice(1, 6)) !== "NUMPY")
    throw new Error("not an .npy file");
  const major = u8[6];
  const hlen =
    major === 1
      ? u8[8] | (u8[9] << 8)
      : u8[8] | (u8[9] << 8) | (u8[10] << 16) | (u8[11] << 24);
  const hstart = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(u8.slice(hstart, hstart + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)[1] === "True";
  const shape = /'shape':\s*\(([^)]*)\)/
    .exec(header)[1]
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  if (fortran) throw new Error("fortran-ordered npy not supported");
  let data;
  if (descr === "<f4") data = new Float32Array(buf, hstart + hlen);
  else if (descr === "|u1") data = new Uint8Array(buf, hstart + hlen);
  else throw new Error(`unsupported npy dtype ${descr}`);
  return { shape, data };
}

function paint(cv, nx, ny, f, table) {
  cv.width = nx;
  cv.height = ny;
  const ctx = cv.getContext("2d"),
    img = ctx.createImageData(nx, ny),
    d = img.data;
  for (let j = 0; j < ny; j++) {
    const row = ny - 1 - j;
    for (let i = 0; i < nx; i++) {
      const rgb = table[f(i, j)],
        k = (row * nx + i) * 4;
      d[k] = rgb >> 16;
      d[k + 1] = (rgb >> 8) & 255;
      d[k + 2] = rgb & 255;
      d[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function paintColorbar(cb, table) {
  cb.width = 6;
  cb.height = 160;
  const ctx = cb.getContext("2d");
  for (let y = 0; y < cb.height; y++) {
    const rgb = table[Math.round((1 - y / (cb.height - 1)) * 255)];
    ctx.fillStyle = `rgb(${rgb >> 16},${(rgb >> 8) & 255},${rgb & 255})`;
    ctx.fillRect(0, y, cb.width, 1);
  }
}

function nat(t, i, j, c) {
  const [, H, W, C] = S.shape;
  return S.frames[((t * H + i) * W + j) * C + c];
}
function lut(c) {
  return DIVERGING.has(CH_CANON[c]) ? COLORMAPS.coolwarm : COLORMAPS.viridis;
}
function panelSize(width, maxHeight) {
  const { x, y } = S.active.domain,
    ratio = (x[1] - x[0]) / (y[1] - y[0]);
  const w = Math.max(40, Math.min(width - 55, maxHeight * ratio));
  return [Math.round(w), Math.round(w / ratio)];
}
function makePanel(title, sub, nx, ny, f, lo, hi, table, size) {
  const el = document.createElement("div");
  el.className = "panel";
  el.innerHTML =
    '<div class="ptitle">' +
    escapeHtml(title) +
    ' <span class="s mono">' +
    escapeHtml(sub) +
    '</span></div><div class="pbody"><canvas class="frame" role="img"></canvas><div class="cbar"><span class="mono"></span><canvas aria-hidden="true"></canvas><span class="mono"></span></div></div>';
  const cv = el.querySelector(".frame");
  paint(cv, nx, ny, f, table);
  cv.setAttribute(
    "aria-label",
    title +
      "; " +
      pretty(S.active.channels[S.ch]) +
      "; color range " +
      fmtSci(lo) +
      " to " +
      fmtSci(hi),
  );
  cv.style.width = size[0] + "px";
  cv.style.height = size[1] + "px";
  const cb = el.querySelector(".cbar");
  cb.style.height = size[1] + "px";
  paintColorbar(cb.querySelector("canvas"), table);
  cb.firstElementChild.textContent = fmtSci(hi);
  cb.lastElementChild.textContent = fmtSci(lo);
  return el;
}
function dialValue(sys, d) {
  const [lo, hi] = sys.dial_ranges[d],
    p = S.dialPos[sys.system][d];
  return sys.dial_meta?.[d]?.scale === "log"
    ? 10 ** (Math.log10(lo) + p * (Math.log10(hi) - Math.log10(lo)))
    : lo + p * (hi - lo);
}
function renderDials() {
  const sys = S.active;
  $("dials").innerHTML = "";
  sys.dials.forEach((d, index) => {
    const [lo, hi] = sys.dial_ranges[d],
      p = S.dialPos[sys.system][d],
      el = document.createElement("div");
    el.className = "dial";
    el.innerHTML =
      '<div class="row"><label for="dial-' +
      index +
      '">' +
      escapeHtml(dialLabel(sys, d)) +
      ' <span class="scale mono">' +
      (sys.dial_meta?.[d]?.scale === "log" ? "log scale" : "linear") +
      '</span></label><span class="val mono">' +
      fmtSci(dialValue(sys, d)) +
      '</span></div><input id="dial-' +
      index +
      '" type="range" min="0" max="1000" step="1" value="' +
      Math.round(p * 1000) +
      '"><div class="bounds mono"><span>' +
      fmtSci(lo) +
      "</span><span>" +
      fmtSci(hi) +
      "</span></div>";
    el.querySelector("input").oninput = (e) => {
      S.dialPos[sys.system][d] = Number(e.target.value) / 1000;
      el.querySelector(".val").textContent = fmtSci(dialValue(sys, d));
      S.paramVersion++;
      clearMetrics();
    };
    $("dials").append(el);
  });
}
function clearMetrics() {
  if (!S.active) return;
  $("metrics").innerHTML = S.active.metrics
    .map(
      (m) =>
        '<div class="metric"><div class="k">' +
        escapeHtml(S.active.metric_meta?.[m]?.label || pretty(m)) +
        '</div><div class="f mono">' +
        escapeHtml(S.active.metric_meta?.[m]?.formula || "") +
        '</div><div class="v mono" data-m="' +
        escapeHtml(m) +
        '">—</div></div>',
    )
    .join("");
}
function selectionInfo() {
  if (!S.active) return;
  $("selection-info").innerHTML =
    "<strong>" +
    escapeHtml(S.active.display_name) +
    "</strong><p>" +
    escapeHtml(S.active.grid.join(" × ")) +
    " native grid</p><p>" +
    escapeHtml(pretty(S.active.channels[S.ch])) +
    "</p><p>" +
    escapeHtml(S.sample?.id || "No sample selected") +
    "</p><p>4 input frames → 1 next state</p>";
}
function syncControls() {
  const ready = !!S.active;
  $("systems").disabled = !ready;
  $("channels").disabled = !ready;
  $("samples").disabled = !ready;
  $("time-select").disabled = !S.frames || S.loading;
  $("run").disabled = !S.frames || S.loading || S.running;
  $("run").textContent = S.running ? "Predicting…" : "Run prediction ↗";
  $("predict").disabled = !ready;
  ["zoom", "fullscreen", "download"].forEach(
    (id) => ($(id).disabled = !S.frames || S.loading),
  );
  if (!ready) return;
  $("systems").value = S.active.system;
  $("channels").innerHTML = S.active.channels
    .map(
      (c, i) =>
        '<option value="' + i + '">' + escapeHtml(pretty(c)) + "</option>",
    )
    .join("");
  $("channels").value = S.ch;
  $("samples").innerHTML = S.samples
    .filter((s) => s.system === S.active.system)
    .map(
      (s) =>
        '<option value="' +
        escapeHtml(s.id) +
        '">' +
        escapeHtml(s.id) +
        "</option>",
    )
    .join("");
  if (S.sample) $("samples").value = S.sample.id;
  selectionInfo();
}
async function selectSystem(sys, sampleId) {
  if (!sys) return;
  stopPlayback();
  S.loadVersion++;
  S.paramVersion++;
  S.active = sys;
  S.sample = null;
  S.frames = null;
  S.pred = null;
  S.err = null;
  S.quality = null;
  S.running = false;
  S.loading = false;
  S.tIdx = 0;
  renderDials();
  clearMetrics();
  syncControls();
  renderStage();
  renderTimeline();
  $("units-note").textContent = sys.units;
  const sample =
    S.samples.find((s) => s.system === sys.system && s.id === sampleId) ||
    S.samples.find((s) => s.system === sys.system);
  if (sample) await loadSample(sample);
}
async function loadSample(sample) {
  stopPlayback();
  const version = ++S.loadVersion;
  S.sample = sample;
  S.frames = null;
  S.pred = null;
  S.err = null;
  S.quality = null;
  S.runTime = null;
  S.loading = true;
  S.running = false;
  S.tIdx = 0;
  notice("");
  syncControls();
  renderStage();
  renderTimeline();
  try {
    const r = await request("/samples/" + encodeURIComponent(sample.id));
    const parsed = parseNpy(await r.arrayBuffer());
    if (version !== S.loadVersion) return;
    if (
      parsed.shape.length !== 4 ||
      parsed.shape[0] !== 5 ||
      parsed.shape[3] !== 4
    )
      throw new Error(
        "The sample format could not be displayed. Choose another sample.",
      );
    S.frames = parsed.data;
    S.shape = parsed.shape;
    S.scales = JSON.parse(r.headers.get("X-Scales"));
  } catch (e) {
    if (version === S.loadVersion)
      notice(
        e.name === "TimeoutError"
          ? "The sample took too long to load. Select it again to retry."
          : e.message === "Failed to fetch"
            ? "Connection lost. Check your connection and select the sample again."
            : e.message,
      );
  } finally {
    if (version === S.loadVersion) {
      S.loading = false;
      syncControls();
      renderStage();
      renderTimeline();
    }
  }
}
function renderStage() {
  const has = !!S.frames;
  $("stage-empty").hidden = has;
  $("panels").hidden = !has;
  $("comparison").hidden = !S.pred;
  $("comparison-empty").hidden = !!S.pred;
  $("quality").hidden = !S.quality;
  if (!has) {
    $("stage-empty").textContent = S.loading
      ? "Loading real simulation frames…"
      : "Choose a sample to inspect a real field.";
    $("panels").replaceChildren();
    $("comparison").replaceChildren();
    $("foot").textContent = "";
    return;
  }
  const c = S.ch,
    [, nx, ny] = S.shape,
    { lo, hi, log } = S.scales[c],
    table = lut(c),
    sys = S.active;
  const size = panelSize(Math.max(160, $("stage").clientWidth - 26), 310);
  $("panels").replaceChildren(
    makePanel(
      S.tIdx === 4 ? "Reference next state" : "Recorded input frame",
      "t" + S.tIdx + " · " + fmtSci(S.sample.times[S.tIdx]),
      nx,
      ny,
      (i, j) => nat(S.tIdx, i, j, c),
      lo,
      hi,
      table,
      size,
    ),
  );
  $("view-description").textContent =
    (S.tIdx === 4
      ? "Reference next frame, held back from the four-frame input."
      : "Real simulation frame " + S.tIdx + " of the four-frame model input.") +
    " Physical time " +
    fmtSci(S.sample.times[S.tIdx]) +
    ".";
  $("scale-note").textContent = DIVERGING.has(CH_CANON[c])
    ? "Coolwarm · symmetric about zero"
    : log
      ? "Viridis · log₁₀ color scale"
      : "Viridis · linear color scale";
  $("field-meta").textContent = nx + " × " + ny + " native";
  $("foot").textContent =
    "x ∈ [" +
    sys.domain.x.join(", ") +
    "] · y ∈ [" +
    sys.domain.y.join(", ") +
    "] · y up · code units";
  if (S.pred) {
    const mobile = window.innerWidth <= 600;
    const cw = $("comparison-card").clientWidth;
    const size = panelSize(
        mobile ? cw - 40 : (cw - 65) / 3,
        mobile ? 245 : 240,
      ),
      off = c * PRED_HW * PRED_HW,
      m = S.errScales[c];
    $("comparison").replaceChildren(
      makePanel(
        "Predicted",
        "t4 · 256 × 256",
        256,
        256,
        (i, j) => S.pred[off + i * 256 + j],
        lo,
        hi,
        table,
        size,
      ),
      makePanel(
        "Ground truth",
        "t4 · native grid",
        nx,
        ny,
        (i, j) => nat(4, i, j, c),
        lo,
        hi,
        table,
        size,
      ),
      makePanel(
        "Signed difference",
        "prediction − truth",
        256,
        256,
        (i, j) => S.err[off + i * 256 + j],
        -m,
        m,
        COLORMAPS.coolwarm,
        size,
      ),
    );
  }
  renderQuality();
  selectionInfo();
}
function qualityHtml(quality, channels, selected = -1) {
  return channels
    .map((name, c) => {
      const q = quality?.[CH_CANON[c]] || {};
      return (
        '<div class="q ' +
        (selected === c ? "selected" : "") +
        '"><div class="k">' +
        escapeHtml(pretty(name)) +
        '</div><div class="v mono">' +
        pct(q.rel_l2) +
        ' <span class="s">rel. L2</span></div><div class="s mono">RMSE ' +
        fmtSci(q.rmse) +
        "</div></div>"
      );
    })
    .join("");
}
function renderQuality() {
  $("quality").hidden = !S.quality;
  if (S.quality)
    $("quality").innerHTML = qualityHtml(S.quality, S.active.channels, S.ch);
}
function setFrame(t) {
  if (!S.frames) return;
  S.tIdx = Math.max(0, Math.min(4, Number(t)));
  renderTimeline();
  renderStage();
}
function stopPlayback() {
  if (playback) clearInterval(playback);
  playback = null;
  $("play").textContent = "▶";
  $("play").setAttribute("aria-label", "Play recorded frames");
}
function startPlayback() {
  if (!S.frames) return;
  stopPlayback();
  $("play").textContent = "Ⅱ";
  $("play").setAttribute("aria-label", "Pause recorded frames");
  playback = setInterval(
    () => setFrame((S.tIdx + 1) % 5),
    1000 / Number($("fps").value),
  );
}
function renderTimeline() {
  $("timeline").hidden = !S.frames;
  if (!S.frames) {
    $("time-select").innerHTML = "<option>—</option>";
    return;
  }
  const times = S.sample.times;
  $("time-select").innerHTML = times
    .map(
      (t, i) =>
        '<option value="' + i + '">t' + i + " · " + fmtSci(t) + "</option>",
    )
    .join("");
  $("time-select").value = S.tIdx;
  $("scrubber").value = S.tIdx;
  $("tl-track").innerHTML = "";
  times.forEach((time, i) => {
    const b = document.createElement("button");
    b.className = "tl" + (i === S.tIdx ? " on" : "");
    b.textContent = "t" + i + " · " + fmtSci(time);
    b.setAttribute("aria-label", "Frame " + i + ", time " + fmtSci(time));
    b.setAttribute("aria-pressed", String(i === S.tIdx));
    b.onclick = () => {
      stopPlayback();
      setFrame(i);
    };
    $("tl-track").append(b);
  });
  $("tl-info").textContent =
    "t0–t3 → predict t4. Source start index " +
    S.sample.t_index +
    ". " +
    (S.runTime !== null
      ? "Last inference: " +
        S.runTime.toFixed(2) +
        " s (model computation only)."
      : "Times are simulation code units.");
}
async function predict() {
  const sys = S.active;
  if (!sys) return;
  const version = ++S.paramVersion;
  const dials = Object.fromEntries(
    sys.dials.map((d) => [d, dialValue(sys, d)]),
  );
  $("predict").disabled = true;
  $("predict").textContent = "Estimating…";
  notice("");
  try {
    const r = await request("/predict/parametric", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ system: sys.system, dials }),
    });
    const j = await r.json();
    if (version !== S.paramVersion || S.active !== sys) return;
    $("metrics")
      .querySelectorAll("[data-m]")
      .forEach((el) => (el.textContent = fmtSci(j.metrics[el.dataset.m])));
    saveRun({
      kind: "parametric",
      system: sys.system,
      name: sys.display_name,
      dials,
      metrics: j.metrics,
    });
  } catch (e) {
    if (version === S.paramVersion)
      notice(
        e.message === "Failed to fetch"
          ? "Connection lost. Please try the estimate again."
          : e.message,
      );
  } finally {
    $("predict").disabled = !S.active;
    $("predict").textContent = "Estimate outcomes →";
  }
}
async function runField() {
  if (!S.frames || S.running || S.loading) return;
  stopPlayback();
  const version = S.loadVersion,
    sample = S.sample,
    sys = S.active;
  S.running = true;
  notice("");
  syncControls();
  try {
    const r = await request(
      "/samples/" + encodeURIComponent(sample.id) + "/predict",
      { cache: "no-store" },
    );
    const parsed = parseNpy(await r.arrayBuffer());
    if (version !== S.loadVersion) return;
    if (parsed.shape.join(",") !== "2,4,256,256")
      throw new Error(
        "The prediction format could not be displayed. Please try again.",
      );
    const n = 4 * 256 * 256;
    S.pred = parsed.data.subarray(0, n);
    S.err = parsed.data.subarray(n, 2 * n);
    S.errScales = JSON.parse(r.headers.get("X-Error-Scales"));
    S.quality = JSON.parse(r.headers.get("X-Quality"));
    S.runTime = Number(r.headers.get("X-Inference-Seconds"));
    S.tIdx = 4;
    renderTimeline();
    renderStage();
    saveRun({
      kind: "field",
      system: sys.system,
      name: sys.display_name,
      sampleId: sample.id,
      ch: S.ch,
      channels: sys.channels,
      quality: S.quality,
      dials: sample.dials,
      seconds: S.runTime,
      epoch: r.headers.get("X-Checkpoint-Epoch"),
      preview: comparisonPreview(),
    });
  } catch (e) {
    if (version === S.loadVersion)
      notice(
        e.message === "Failed to fetch"
          ? "Connection lost during prediction. Your sample is still selected; try again."
          : e.message,
      );
  } finally {
    if (version === S.loadVersion) {
      S.running = false;
      syncControls();
    }
  }
}
function comparisonPreview() {
  const cv = document.createElement("canvas");
  cv.width = 720;
  cv.height = 260;
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#0B0A12";
  ctx.fillRect(0, 0, 720, 260);
  ctx.fillStyle = "#B5AEC7";
  ctx.font = "12px sans-serif";
  const titles = ["Predicted", "Ground truth", "Signed difference"];
  const ratio =
    (S.active.domain.x[1] - S.active.domain.x[0]) /
    (S.active.domain.y[1] - S.active.domain.y[0]);
  document.querySelectorAll("#comparison canvas.frame").forEach((frame, i) => {
    const w = Math.min(205, 215 * ratio),
      h = w / ratio;
    ctx.fillText(titles[i], i * 240 + 12, 21);
    ctx.drawImage(frame, i * 240 + (240 - w) / 2, 34, w, h);
  });
  return cv.toDataURL("image/png");
}
function saveRun(run) {
  run = { ...run, id: crypto.randomUUID(), date: new Date().toISOString() };
  history = [run, ...history].slice(0, 20);
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch {
    notice(
      "The result is ready, but browser storage is full or unavailable. History will last only while this page is open.",
    );
  }
  renderHistory();
}
function runCard(run) {
  const el = document.createElement("div");
  el.className = "run-card";
  el.innerHTML =
    "<strong>" +
    escapeHtml(run.name || run.system) +
    "</strong><p>" +
    escapeHtml(
      run.kind === "field"
        ? pretty(run.channels?.[run.ch] || "field") + " · " + run.sampleId
        : "Parameter estimate",
    ) +
    "</p><p>" +
    escapeHtml(new Date(run.date).toLocaleString()) +
    "</p>";
  if (run.preview?.startsWith("data:image/png;base64,")) {
    const img = document.createElement("img");
    img.src = run.preview;
    img.alt = "Saved comparison: predicted, ground truth, signed difference";
    el.append(img);
  }
  const button = document.createElement("button");
  button.textContent = "View saved result →";
  button.onclick = () => {
    location.hash = "history";
    showHistory(run);
  };
  el.append(button);
  return el;
}
function renderHistory() {
  for (const [id, runs] of [
    ["recent-runs", history.slice(0, 3)],
    ["history-list", history],
  ]) {
    const el = $(id);
    el.replaceChildren();
    if (!runs.length) {
      el.innerHTML =
        '<div class="empty">Your next exploration starts here.<br>Run a prediction or parameter estimate to save a result.</div>';
    } else runs.forEach((r) => el.append(runCard(r)));
  }
  $("clear-history").disabled = !history.length;
}
function showHistory(run) {
  const el = $("history-detail");
  el.hidden = false;
  el.innerHTML =
    '<p class="eyebrow">SAVED IN THIS BROWSER</p><h2>' +
    escapeHtml(run.name || run.system) +
    '</h2><p class="small muted">' +
    escapeHtml(new Date(run.date).toLocaleString()) +
    "</p>";
  if (run.preview?.startsWith("data:image/png;base64,")) {
    const img = document.createElement("img");
    img.src = run.preview;
    img.alt = "Saved predicted, ground truth, and signed difference panels";
    el.append(img);
  }
  if (run.kind === "field") {
    const q = document.createElement("div");
    q.className = "quality";
    q.innerHTML = qualityHtml(run.quality, run.channels || CH_CANON, run.ch);
    el.append(q);
  } else {
    const p = document.createElement("p");
    p.textContent = Object.entries(run.metrics || {})
      .map(([k, v]) => pretty(k) + ": " + fmtSci(v))
      .join(" · ");
    el.append(p);
  }
  const p = document.createElement("p");
  p.className = "small muted";
  p.textContent = Object.entries(run.dials || {})
    .map(([k, v]) => pretty(k) + " = " + fmtSci(v))
    .join(" · ");
  el.append(p);
  const button = document.createElement("button");
  button.className = "btn secondary small";
  button.textContent =
    run.kind === "field"
      ? "Open sample to run again →"
      : "Open these parameters →";
  button.onclick = async () => {
    const sys = S.systems.find((s) => s.system === run.system);
    if (!sys) {
      notice("The service is still connecting. Try again when it is ready.");
      return;
    }
    location.hash = "simulator";
    await selectSystem(sys, run.sampleId);
    if (run.kind === "field") {
      S.ch = run.ch || 0;
      syncControls();
      renderStage();
    } else {
      for (const d of sys.dials) {
        const [lo, hi] = sys.dial_ranges[d],
          v = run.dials[d];
        if (Number.isFinite(v))
          S.dialPos[sys.system][d] = Math.max(
            0,
            Math.min(
              1,
              sys.dial_meta?.[d]?.scale === "log"
                ? (Math.log10(v) - Math.log10(lo)) /
                    (Math.log10(hi) - Math.log10(lo))
                : (v - lo) / (hi - lo),
            ),
          );
      }
      renderDials();
    }
  };
  el.append(button);
  requestAnimationFrame(() => el.scrollIntoView({ block: "start" }));
}
const DATASET_NOTES = {
  turbulent_radiative_layer_2D: {
    title: "Cooling meets turbulence.",
    description:
      "A mixing layer with radiative cooling. Explore how cooling time changes the recorded flow and interpolated mass flux.",
    source: "turbulent_radiative_layer_2D",
  },
  rayleigh_benard: {
    title: "Convection, from below.",
    description:
      "Buoyancy-driven convection in a wide domain. Rayleigh and Prandtl numbers describe the balance of thermal driving and transport.",
    source: "rayleigh_benard",
  },
  shear_flow: {
    title: "Layers in motion.",
    description:
      "Shear-driven mixing with a passive tracer. Reynolds and Schmidt numbers control momentum and tracer transport.",
    source: "shear_flow",
  },
};
function renderDatasets() {
  $("dataset-cards").replaceChildren();
  S.systems.forEach((sys, index) => {
    const note = DATASET_NOTES[sys.system],
      el = document.createElement("section");
    el.className = "card dataset-card";
    el.innerHTML =
      '<div class="dataset-top"><div><p class="eyebrow">0' +
      (index + 1) +
      " / " +
      escapeHtml(sys.display_name.toUpperCase()) +
      "</p><h2>" +
      escapeHtml(note.title) +
      "</h2><p>" +
      escapeHtml(note.description) +
      '</p></div><a class="text-link" href="https://polymathic-ai.org/the_well/datasets/' +
      note.source +
      '/" target="_blank" rel="noopener">Dataset documentation ↗</a></div><div class="dataset-meta"><span>Native grid <strong>' +
      sys.grid.join(" × ") +
      "</strong></span><span>x <strong>[" +
      sys.domain.x.join(", ") +
      "]</strong> · y <strong>[" +
      sys.domain.y.join(", ") +
      "]</strong></span><span>Fields <strong>" +
      escapeHtml(sys.channels.map(pretty).join(", ")) +
      '</strong></span></div><p style="margin-top:15px">' +
      sys.dials
        .map(
          (d) =>
            escapeHtml(dialLabel(sys, d)) +
            ": " +
            sys.dial_ranges[d].map(fmtSci).join(" – "),
        )
        .join(" · ") +
      '<br>All quantities in simulation code units. Parameter ranges apply to the numeric explorer.</p><div class="sample-grid"></div>';
    S.samples
      .filter((s) => s.system === sys.system)
      .forEach((sample) => {
        const entry = document.createElement("div");
        entry.className = "sample-entry";
        const deltas = sample.times.slice(1).map((t, i) => t - sample.times[i]);
        entry.innerHTML =
          "<h3>" +
          escapeHtml(sample.id) +
          "</h3><p>" +
          Object.entries(sample.dials)
            .map(([k, v]) => escapeHtml(dialLabel(sys, k)) + " = " + fmtSci(v))
            .join(" · ") +
          "</p><p>Source start index " +
          sample.t_index +
          ' · 5 consecutive frames</p><p class="times">' +
          sample.times
            .map((t, i) => "t" + i + " = " + t.toFixed(6))
            .join("<br>") +
          '</p><p class="times">Δt: ' +
          deltas.map((d) => d.toFixed(6)).join(", ") +
          "</p>";
        const b = document.createElement("button");
        b.className = "btn secondary small";
        b.textContent = "Explore this window ↗";
        b.onclick = () => {
          location.hash = "simulator";
          selectSystem(sys, sample.id);
        };
        entry.append(b);
        el.querySelector(".sample-grid").append(entry);
      });
    $("dataset-cards").append(el);
  });
}
function inspectField(zoom) {
  if (!S.frames) return;
  const c = S.ch,
    [, nx, ny] = S.shape,
    { lo, hi } = S.scales[c];
  const size = panelSize(
    Math.min(window.innerWidth - 100, 760),
    zoom ? 1100 : window.innerHeight * 0.62,
  );
  $("dialog-field").classList.toggle("zoomed", zoom);
  $("dialog-field").replaceChildren(
    makePanel(
      "Ground truth",
      "t" + S.tIdx,
      nx,
      ny,
      (i, j) => nat(S.tIdx, i, j, c),
      lo,
      hi,
      lut(c),
      zoom ? [size[0] * 2, size[1] * 2] : size,
    ),
  );
  $("dialog-title").textContent =
    pretty(S.active.channels[c]) + " · " + S.sample.id;
  $("field-dialog").showModal();
}
function downloadField() {
  if (!S.frames) return;
  const source = $("panels").querySelector("canvas.frame"),
    out = document.createElement("canvas"),
    ratio =
      (S.active.domain.x[1] - S.active.domain.x[0]) /
      (S.active.domain.y[1] - S.active.domain.y[0]);
  const h = 600,
    w = Math.round(h * ratio);
  out.width = Math.max(w + 120, 480);
  out.height = h + 105;
  const ctx = out.getContext("2d");
  ctx.fillStyle = "#0B0A12";
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.fillStyle = "#F4F1FF";
  ctx.font = "14px sans-serif";
  ctx.fillText(
    "Delta-V · " +
      pretty(S.active.channels[S.ch]) +
      " · ground truth t" +
      S.tIdx,
    18,
    26,
  );
  ctx.drawImage(source, 18, 45, w, h);
  const table = lut(S.ch);
  for (let y = 0; y < h; y++) {
    const rgb = table[Math.round((1 - y / (h - 1)) * 255)];
    ctx.fillStyle = "#" + rgb.toString(16).padStart(6, "0");
    ctx.fillRect(w + 32, 45 + y, 8, 1);
  }
  ctx.fillStyle = "#B5AEC7";
  ctx.font = "12px monospace";
  ctx.fillText(fmtSci(S.scales[S.ch].hi), w + 46, 56);
  ctx.fillText(fmtSci(S.scales[S.ch].lo), w + 46, h + 45);
  ctx.fillText(
    S.sample.id + " · time " + fmtSci(S.sample.times[S.tIdx]),
    18,
    h + 73,
  );
  ctx.fillText(
    "Simulation code units · " +
      (S.scales[S.ch].log ? "log10" : "linear") +
      " color scale",
    18,
    h + 92,
  );
  const a = document.createElement("a");
  a.href = out.toDataURL();
  a.download =
    "deltav-" +
    S.sample.id +
    "-" +
    S.active.channels[S.ch] +
    "-t" +
    S.tIdx +
    ".png";
  a.click();
}
const TOUR = [
  [
    "selection-bar",
    "Choose your system and window",
    "The physical system selects the available fields and two recorded sample windows. Frame selection changes the reference view; the prediction always targets t4.",
  ],
  [
    "parametric",
    "Explore parameter estimates",
    "Move the logarithmic sliders, then estimate flux and turbulent velocity. These dials do not change the recorded field sample.",
  ],
  [
    "current-card",
    "Read the actual field",
    "This is a real recorded snapshot. Check the numeric color scale; velocity uses a scale symmetric about zero. Zoom, expand, or download it.",
  ],
  [
    "comparison-card",
    "Predict and inspect the difference",
    "Run prediction to see the learned next frame, reference, and signed error. Read the measured relative L2 and RMSE for every field. There is no pass/fail target.",
  ],
  [
    "timeline",
    "Move through recorded time",
    "Step through five real frames or play them at your chosen rate. This is recorded playback, not an autoregressive model rollout.",
  ],
  [
    "recent-runs",
    "Revisit and give feedback",
    "Run history saves metrics and previews in this browser. Datasets explains every window; About explains the research and limitations. Share observations using Give feedback.",
  ],
];
function showTour(step) {
  document
    .querySelectorAll(".tour-highlight")
    .forEach((el) => el.classList.remove("tour-highlight"));
  tourStep = step;
  $("tour").hidden = step < 0;
  if (step < 0) return;
  const [id, title, copy] = TOUR[step];
  $("tour-count").textContent =
    "WALKTHROUGH / " + (step + 1) + " OF " + TOUR.length;
  $("tour-title").textContent = title;
  $("tour-copy").textContent = copy;
  $("tour-back").disabled = step === 0;
  $("tour-next").textContent = step === TOUR.length - 1 ? "Finish" : "Next →";
  $(id).classList.add("tour-highlight");
  if (id === "parametric") $("parametric").open = true;
}
async function boot() {
  const started = Date.now();
  for (;;) {
    try {
      const response = await fetch("/health", {
        cache: "no-store",
        signal: AbortSignal.timeout(15000),
      });
      if (response.ok) {
        const health = await response.json();
        if (health.model_loaded) {
          S.epoch = health.checkpoint_epoch;
          $("wake").hidden = true;
          $("dot").className = "dot on";
          $("status-text").textContent = "Online · " + health.device;
          $("model-epoch").textContent = "Epoch " + health.checkpoint_epoch;
          break;
        }
        $("wake-phase").textContent = String(health.phase).startsWith("error")
          ? "Model unavailable"
          : "Loading model";
        $("wake-title").textContent = String(health.phase).startsWith("error")
          ? "The model could not start"
          : "Waking up the model";
        $("wake-fill").style.width =
          ({
            "downloading artifacts": 35,
            "loading model": 70,
            "loading tables": 90,
          }[health.phase] || 10) + "%";
      }
    } catch {
      $("wake-phase").textContent = "Connecting to service";
    }
    $("wake").hidden = false;
    $("dot").className = "dot warn";
    $("status-text").textContent = "Warming up";
    $("wake-elapsed").textContent =
      Math.round((Date.now() - started) / 1000) + "s";
    await new Promise((r) => setTimeout(r, 2500));
  }
  try {
    const [sysResponse, sampleResponse] = await Promise.all([
      request("/systems"),
      request("/samples"),
    ]);
    S.systems = await sysResponse.json();
    S.samples = await sampleResponse.json();
    S.systems.forEach(
      (sys) =>
        (S.dialPos[sys.system] = Object.fromEntries(
          sys.dials.map((d) => [d, 0.5]),
        )),
    );
    $("systems").innerHTML = S.systems
      .map(
        (sys) =>
          '<option value="' +
          escapeHtml(sys.system) +
          '">' +
          escapeHtml(sys.display_name) +
          "</option>",
      )
      .join("");
    renderDatasets();
    await selectSystem(
      S.systems.find((s) => s.system === (pendingSystem || "shear_flow")) ||
        S.systems[0],
    );
  } catch (e) {
    notice("The catalog could not load. Reload this page to reconnect.");
  }
}
setTheme(storageGet("deltav-theme", "dark") === "light" ? "light" : "dark");
route();
renderHistory();
syncControls();
window.addEventListener("hashchange", route);
document.querySelector(".skip").onclick = (event) => {
  event.preventDefault();
  $("main-content").setAttribute("tabindex", "-1");
  $("main-content").focus();
};
$("theme").onclick = toggleTheme;
$("settings-theme").onclick = toggleTheme;
$("systems").onchange = (e) =>
  selectSystem(S.systems.find((s) => s.system === e.target.value));
$("samples").onchange = (e) =>
  loadSample(S.samples.find((s) => s.id === e.target.value));
$("channels").onchange = (e) => {
  S.ch = Number(e.target.value);
  renderStage();
};
$("time-select").onchange = (e) => {
  stopPlayback();
  setFrame(e.target.value);
};
$("scrubber").oninput = (e) => {
  stopPlayback();
  setFrame(e.target.value);
};
$("prev-frame").onclick = () => {
  stopPlayback();
  setFrame(S.tIdx - 1);
};
$("next-frame").onclick = () => {
  stopPlayback();
  setFrame(S.tIdx + 1);
};
$("play").onclick = () => (playback ? stopPlayback() : startPlayback());
$("fps").onchange = () => {
  if (playback) startPlayback();
};
$("predict").onclick = predict;
$("run").onclick = runField;
$("zoom").onclick = () => inspectField(true);
$("fullscreen").onclick = () => inspectField(false);
$("download").onclick = downloadField;
$("dialog-close").onclick = () => $("field-dialog").close();
$("clear-history").onclick = () => {
  if (!confirm("Clear all saved runs from this browser?")) return;
  history = [];
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch {}
  $("history-detail").hidden = true;
  renderHistory();
};
$("tour-start").onclick = () => showTour(0);
$("about-tour").onclick = () => {
  location.hash = "simulator";
  showTour(0);
};
$("tour-next").onclick = () =>
  showTour(tourStep === TOUR.length - 1 ? -1 : tourStep + 1);
$("tour-back").onclick = () => showTour(Math.max(0, tourStep - 1));
$("tour-close").onclick = () => showTour(-1);
document.querySelectorAll("[data-anchor]").forEach(
  (a) =>
    (a.onclick = (e) => {
      e.preventDefault();
      $(a.dataset.anchor).scrollIntoView({ block: "start" });
    }),
);
document.querySelectorAll("[data-start]").forEach(
  (a) =>
    (a.onclick = () => {
      pendingSystem = a.dataset.start;
      const sys = S.systems.find((s) => s.system === pendingSystem);
      if (sys) selectSystem(sys);
    }),
);
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (S.frames && page === "simulator") renderStage();
  }, 100);
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopPlayback();
});
boot();
