/* Delta-V frontend. Talks to the FastAPI service on the same origin. */
const $ = (id) => document.getElementById(id);
const CH_CANON = ['scalar', 'pressure', 'velocity_x', 'velocity_y'];
const DIVERGING = new Set(['velocity_x', 'velocity_y']);
const PRED_HW = 256;

const S = {
  systems: [], samples: [], active: null,
  dialPos: {},               // system -> dial -> slider position 0..1
  sample: null,              // selected sample meta
  frames: null,              // Uint8Array (5,H,W,4) native, quantized per channel (see scales)
  shape: null,               // [5,H,W,4]
  pred: null,                // Uint8Array (4,256,256), same scales as frames
  err: null,                 // Uint8Array (4,256,256), symmetric ±errScales[c]
  errScales: null, quality: null,
  scales: null,              // per channel {lo, hi, log} — one scale across the whole sample
  ch: 0, tIdx: 4,
  runTime: null,
};

/* ---------- .npy parsing (little-endian float32, C order) ---------- */
function parseNpy(buf) {
  const u8 = new Uint8Array(buf);
  if (String.fromCharCode(...u8.slice(1, 6)) !== 'NUMPY') throw new Error('not an .npy file');
  const major = u8[6];
  const hlen = major === 1 ? (u8[8] | (u8[9] << 8)) : (u8[8] | (u8[9] << 8) | (u8[10] << 16) | (u8[11] << 24));
  const hstart = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(u8.slice(hstart, hstart + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)[1] === 'True';
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)[1].split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  if (fortran) throw new Error('fortran-ordered npy not supported');
  let data;
  if (descr === '<f4') data = new Float32Array(buf, hstart + hlen);
  else if (descr === '|u1') data = new Uint8Array(buf, hstart + hlen);
  else throw new Error(`unsupported npy dtype ${descr}`);
  return { shape, data };
}

/* ---------- formatting ---------- */
const fmtVal = (name, v) => {
  if (name === 't_cool') return v.toFixed(2);
  if (name === 'rayleigh_number') return v.toExponential(1).replace('e+', 'e');
  if (name === 'reynolds') return Number(v.toPrecision(3)).toLocaleString();
  return v.toPrecision(3).replace(/\.?0+$/, '');
};
const fmtSci = (v) => {
  if (!isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a < 1e-3 || a >= 1e4) return v.toExponential(2).replace('e+', 'e');
  return Number(v.toPrecision(4)).toString();
};
const dialLabel = (sys, d) => (sys.dial_meta[d] && sys.dial_meta[d].label) || d;

/* ---------- wake-up / health ---------- */
const PHASE_PCT = { connecting: 5, starting: 10, 'downloading artifacts': 35, 'loading model': 70, 'loading tables': 90, ready: 100 };
async function waitForModel() {
  const t0 = Date.now();
  let shown = false;
  for (;;) {
    let h = null;
    try { const r = await fetch('/health', { cache: 'no-store' }); if (r.ok) h = await r.json(); } catch (_) { /* sleeping */ }
    if (h && h.model_loaded) {
      $('wake').hidden = true;
      $('dot').className = 'dot on';
      $('status-text').innerHTML = `online<span class="sep">·</span>${h.device}<span class="sep">·</span>epoch ${h.checkpoint_epoch}`;
      return h;
    }
    const elapsed = Math.round((Date.now() - t0) / 1000);
    if (elapsed > 3 && !shown) { $('wake').hidden = false; shown = true; }
    const phase = h ? h.phase : 'connecting';
    $('wake-phase').textContent = phase;
    $('wake-elapsed').textContent = `${elapsed}s`;
    $('wake-fill').style.width = `${PHASE_PCT[phase] ?? 20}%`;
    $('dot').className = 'dot warn';
    $('status-text').textContent = phase;
    if (phase.startsWith('error')) { $('wake-title').textContent = 'The model failed to load'; }
    await new Promise((r) => setTimeout(r, 2000));
  }
}

/* ---------- systems & dials ---------- */
function renderSystems() {
  const el = $('systems'); el.innerHTML = '';
  S.systems.forEach((sys) => {
    const b = document.createElement('button');
    b.className = 'sys' + (sys === S.active ? ' on' : '');
    b.innerHTML = `<span>${sys.display_name}</span><span class="grid mono">${sys.grid[0]} × ${sys.grid[1]}</span>`;
    b.onclick = () => selectSystem(sys);
    el.appendChild(b);
  });
}

function dialValue(sys, d) {
  const [lo, hi] = sys.dial_ranges[d];
  const p = S.dialPos[sys.system][d];
  const log = (sys.dial_meta[d] || {}).scale === 'log';
  return log ? Math.pow(10, Math.log10(lo) + p * (Math.log10(hi) - Math.log10(lo))) : lo + p * (hi - lo);
}

function renderDials() {
  const sys = S.active, el = $('dials'); el.innerHTML = '';
  sys.dials.forEach((d) => {
    const [lo, hi] = sys.dial_ranges[d];
    const p = S.dialPos[sys.system][d];
    const log = (sys.dial_meta[d] || {}).scale === 'log';
    const w = document.createElement('div'); w.className = 'dial';
    w.innerHTML = `
      <div class="row"><span class="name">${dialLabel(sys, d)} <span class="mono scale">${d}${log ? ' · log' : ''}</span></span><span class="val mono">${fmtVal(d, dialValue(sys, d))}</span></div>
      <input type="range" min="0" max="1000" step="1" value="${Math.round(p * 1000)}" style="--fill:${p * 100}%">
      <div class="bounds mono"><span>${fmtVal(d, lo)}</span><span>${fmtVal(d, hi)}</span></div>`;
    w.querySelector('input').oninput = (e) => {
      S.dialPos[sys.system][d] = e.target.value / 1000;
      e.target.style.setProperty('--fill', `${e.target.value / 10}%`);
      w.querySelector('.val').textContent = fmtVal(d, dialValue(sys, d));
      clearMetrics();
    };
    el.appendChild(w);
  });
}

function selectSystem(sys) {
  S.active = sys;
  S.sample = null; S.frames = null; S.pred = null; S.err = null; S.quality = null; S.tIdx = 4; S.runTime = null;
  renderSystems(); renderDials(); clearMetrics(); renderSamples(); renderChannels(); renderStage(); renderTimeline(); renderQuality();
  $('field-meta').textContent = `FNO · ${sys.grid[0]} × ${sys.grid[1]} native → 256 × 256 · 4 channels`;
  $('units-note').textContent = sys.units;
}

/* ---------- parametric tier ---------- */
function clearMetrics() {
  const sys = S.active, el = $('metrics'); el.innerHTML = '';
  sys.metrics.forEach((m) => {
    const mm = sys.metric_meta[m] || {};
    el.insertAdjacentHTML('beforeend', `<div class="metric"><div class="k">${mm.label || m}</div><div class="f mono">${m}${mm.formula ? ' = ' + mm.formula : ''}</div><div class="v mono empty" data-m="${m}">—</div></div>`);
  });
}
async function predict() {
  const sys = S.active;
  const dials = {}; sys.dials.forEach((d) => { dials[d] = dialValue(sys, d); });
  $('predict').disabled = true;
  try {
    const r = await fetch('/predict/parametric', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ system: sys.system, dials }) });
    const j = await r.json();
    if (!r.ok) {
      $('metrics').querySelectorAll('.metric').forEach((m) => { m.querySelector('.v').textContent = '—'; if (!m.querySelector('.err')) m.insertAdjacentHTML('beforeend', `<div class="err">${j.detail || r.status}</div>`); });
      return;
    }
    $('metrics').querySelectorAll('.v').forEach((v) => { const m = v.dataset.m; v.textContent = fmtSci(j.metrics[m]); v.classList.remove('empty'); });
  } finally { $('predict').disabled = false; }
}

/* ---------- samples ---------- */
function sampleLabel(s) {
  return Object.entries(s.dials).map(([k, v]) => `${dialLabel(S.active, k)} ${fmtVal(k, v)}`).join(' · ');
}
function renderSamples() {
  const el = $('samples'); el.innerHTML = '';
  S.samples.filter((s) => s.system === S.active.system).forEach((s) => {
    const b = document.createElement('button');
    b.className = 'chip mono' + (S.sample && S.sample.id === s.id ? ' on' : '');
    b.textContent = sampleLabel(s);
    b.onclick = () => loadSample(s);
    el.appendChild(b);
  });
  $('run').disabled = !S.sample;
}
async function loadSample(s) {
  S.sample = s; S.pred = null; S.err = null; S.quality = null; S.runTime = null; S.tIdx = 0;
  renderSamples(); renderStage(true); renderQuality();
  const r = await fetch(`/samples/${s.id}`);
  const { shape, data } = parseNpy(await r.arrayBuffer());
  S.frames = data; S.shape = shape; S.scales = JSON.parse(r.headers.get('X-Scales'));
  renderChannels(); renderTimeline(); renderStage();
  $('run').disabled = false;
}

/* ---------- array helpers ---------- */
// native layout: frames[t, i(x), j(y), c], values already quantized 0..255
function nat(t, i, j, c) { const [, H, W, C] = S.shape; return S.frames[((t * H + i) * W + j) * C + c]; }

/* ---------- rendering ---------- */
function lut(c) { return DIVERGING.has(CH_CANON[c]) ? COLORMAPS.coolwarm : COLORMAPS.viridis; }

// Paint an image with cols = x (i), rows = y (j), y increasing upward, from a
// sampler f(i, j) returning an already-quantized 0..255 level.
function paint(cv, nx, ny, f, table) {
  cv.width = nx; cv.height = ny;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(nx, ny), d = img.data;
  for (let j = 0; j < ny; j++) {
    const row = ny - 1 - j;
    for (let i = 0; i < nx; i++) {
      const rgb = table[f(i, j)], k = (row * nx + i) * 4;
      d[k] = rgb >> 16; d[k + 1] = (rgb >> 8) & 255; d[k + 2] = rgb & 255; d[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function paintColorbar(cb, table) {
  cb.width = 6; cb.height = 160;
  const ctx = cb.getContext('2d');
  for (let y = 0; y < cb.height; y++) {
    const rgb = table[Math.round((1 - y / (cb.height - 1)) * 255)];
    ctx.fillStyle = `rgb(${rgb >> 16},${(rgb >> 8) & 255},${rgb & 255})`; ctx.fillRect(0, y, cb.width, 1);
  }
}

function frameSize(nPanels, stacked) {
  const sys = S.active, Lx = sys.domain.x[1] - sys.domain.x[0], Ly = sys.domain.y[1] - sys.domain.y[0];
  const stageW = $('stage').clientWidth - 32;
  let maxW, maxH;
  if (stacked) { maxW = stageW - 60; maxH = Math.min(220, 620 / nPanels); }
  else { maxW = (stageW - 22 * (nPanels - 1)) / nPanels - 60; maxH = 500; }
  let w = maxH * Lx / Ly, h = maxH;
  if (w > maxW) { w = maxW; h = maxW * Ly / Lx; }
  return [Math.round(w), Math.round(h)];
}

function makePanel(title, sub, nx, ny, f, lo, hi, table, size) {
  const el = document.createElement('div'); el.className = 'panel';
  el.innerHTML = `<div class="ptitle">${title} <span class="s mono">${sub}</span></div><div class="pbody"><canvas class="frame"></canvas><div class="cbar"><span class="mono"></span><canvas></canvas><span class="mono"></span></div></div>`;
  const cv = el.querySelector('canvas.frame'); paint(cv, nx, ny, f, table);
  cv.style.width = `${size[0]}px`; cv.style.height = `${size[1]}px`;
  const [top, , bot] = el.querySelectorAll('.cbar > *'); paintColorbar(el.querySelector('.cbar canvas'), table);
  top.textContent = fmtSci(hi); bot.textContent = fmtSci(lo);
  el.querySelector('.cbar').style.height = `${size[1]}px`;
  return el;
}

function renderStage(loading) {
  const has = !!S.frames, panels = $('panels');
  $('stage-empty').hidden = has; panels.hidden = !has;
  if (!has) { $('stage-empty').textContent = loading ? 'Loading initial condition…' : 'Select an initial condition'; $('foot').innerHTML = ''; $('scale-note').textContent = ''; return; }
  const c = S.ch, [, H, W] = S.shape, { lo, hi, log } = S.scales[c], table = lut(c);
  const chName = S.active.channels[c], sys = S.active;
  const stacked = (sys.domain.x[1] - sys.domain.x[0]) > (sys.domain.y[1] - sys.domain.y[0]);
  panels.innerHTML = ''; panels.classList.toggle('stack', stacked);
  const n = PRED_HW * PRED_HW, off = c * n;
  let foot;
  if (S.tIdx < 4) {
    panels.appendChild(makePanel(`history t${S.tIdx}`, `native ${H} × ${W}`, H, W, (i, j) => nat(S.tIdx, i, j, c), lo, hi, table, frameSize(1, false)));
    foot = `input frame t${S.tIdx} · ${chName}`;
  } else if (!S.pred) {
    panels.appendChild(makePanel('ground truth t4', `native ${H} × ${W}`, H, W, (i, j) => nat(4, i, j, c), lo, hi, table, frameSize(1, false)));
    foot = `ground truth t4 · ${chName} · run the field simulation to predict this frame`;
  } else {
    const size = frameSize(3, stacked), m = S.errScales[c];
    panels.appendChild(makePanel('ground truth t4', `native ${H} × ${W}`, H, W, (i, j) => nat(4, i, j, c), lo, hi, table, size));
    panels.appendChild(makePanel('prediction t4', `256 × 256 · ${S.runTime}`, PRED_HW, PRED_HW, (i, j) => S.pred[off + i * PRED_HW + j], lo, hi, table, size));
    panels.appendChild(makePanel('prediction − truth', '256 × 256', PRED_HW, PRED_HW, (i, j) => S.err[off + i * PRED_HW + j], -m, m, COLORMAPS.coolwarm, size));
    foot = `t4 · ${chName} · truth and prediction share one color scale; error is symmetric about 0`;
  }
  $('scale-note').textContent = DIVERGING.has(CH_CANON[c]) ? 'coolwarm · symmetric about 0' : (log ? 'viridis · log₁₀ scale' : 'viridis · linear');
  $('foot').innerHTML = `<span>${foot}</span><span>x ∈ [${sys.domain.x}] · y ∈ [${sys.domain.y}] · y up</span>`;
}

function renderChannels() {
  const el = $('channels'); el.innerHTML = '';
  S.active.channels.forEach((name, i) => {
    const b = document.createElement('button'); b.className = 'seg mono' + (i === S.ch ? ' on' : ''); b.textContent = name;
    b.onclick = () => { S.ch = i; renderChannels(); renderStage(); };
    el.appendChild(b);
  });
}

function renderTimeline() {
  const tl = $('timeline'); tl.hidden = !S.frames; if (!S.frames) return;
  const track = $('tl-track'); track.innerHTML = '';
  const times = S.sample.times;
  for (let t = 0; t < 5; t++) {
    const b = document.createElement('button');
    b.className = 'tl' + (t === S.tIdx ? ' on' : '') + (t === 4 ? (S.pred ? ' pred' : ' truth') : '');
    b.textContent = t < 4 ? `t${t} · ${times[t].toFixed(2)}` : `t4 · ${times[4].toFixed(2)} · ${S.pred ? 'predicted' : 'truth'}`;
    b.onclick = () => { S.tIdx = t; renderTimeline(); renderChannels(); renderStage(); };
    track.appendChild(b);
  }
  $('tl-info').textContent = `The model sees t0–t3 and predicts t4. Sample from the training slice at ${sampleLabel(S.sample)}, time index ${S.sample.t_index}.`;
}

/* ---------- field tier ---------- */
async function runField() {
  if (!S.sample) return;
  const btn = $('run'); btn.disabled = true; btn.textContent = 'Running the FNO…';
  try {
    const r = await fetch(`/samples/${S.sample.id}/predict`, { cache: 'no-store' });
    if (!r.ok) { const j = await r.json().catch(() => ({})); $('stage-empty').hidden = false; $('stage-empty').textContent = `Error: ${j.detail || r.status}`; return; }
    const { shape, data } = parseNpy(await r.arrayBuffer());
    if (shape.length !== 4 || shape[0] !== 2 || shape[2] !== PRED_HW) throw new Error(`unexpected shape ${shape}`);
    const n = 4 * PRED_HW * PRED_HW;
    S.pred = data.subarray(0, n); S.err = data.subarray(n, 2 * n);
    S.errScales = JSON.parse(r.headers.get('X-Error-Scales'));
    S.quality = JSON.parse(r.headers.get('X-Quality'));
    S.runTime = `${Number(r.headers.get('X-Inference-Seconds')).toFixed(1)} s on cpu`;
    S.tIdx = 4;
    renderTimeline(); renderChannels(); renderStage(); renderQuality();
  } catch (e) { $('stage-empty').hidden = false; $('stage-empty').textContent = `Error: ${e.message}`; }
  finally { btn.disabled = false; btn.textContent = 'Run field simulation again'; }
}

function renderQuality() {
  const q = $('quality'); q.hidden = !S.quality; if (!S.quality) return;
  q.innerHTML = '';
  S.active.channels.forEach((name, c) => {
    const { rel_l2, rmse } = S.quality[CH_CANON[c]];
    q.insertAdjacentHTML('beforeend', `<div class="q"><div class="k">${name}</div><div class="v mono">${rel_l2 == null ? '—' : (rel_l2 * 100).toFixed(1) + '%'} <span class="s">rel. L2</span></div><div class="s mono">rmse ${fmtSci(rmse)} · vs truth resampled to 256²</div></div>`);
  });
}

/* ---------- boot ---------- */
async function main() {
  await waitForModel();
  const [systems, samples] = await Promise.all([fetch('/systems').then((r) => r.json()), fetch('/samples').then((r) => r.json())]);
  S.systems = systems; S.samples = samples;
  systems.forEach((sys) => { S.dialPos[sys.system] = {}; sys.dials.forEach((d) => { S.dialPos[sys.system][d] = 0.5; }); });
  $('predict').onclick = predict;
  $('run').onclick = runField;
  window.addEventListener('resize', () => { if (S.frames) renderStage(); });
  selectSystem(systems[0]);
}
main();
