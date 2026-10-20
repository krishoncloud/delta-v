/* Delta-V study workflow: formatting and SVG chart helpers (no data, no model).
 *
 * Pure functions, no DOM access at load time, so tests can require() this file.
 * Every number drawn here comes from the /flow API or from the published
 * research evidence file. There is no browser model, no synthetic solver and no
 * fallback: if the API fails, callers show the error.
 */
(function (root) {
  "use strict";

  /* ---------- Formatting ---------- */
  function fmt(v, digits) {
    if (!Number.isFinite(v)) return "n/a";
    const d = digits == null ? 3 : digits;
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a >= 10000 || a < 0.001) return v.toExponential(d - 1);
    return Number(v.toPrecision(d)).toString();
  }
  const pct = (v, digits) => (Number.isFinite(v) ? (v * 100).toFixed(digits == null ? 1 : digits).replace(/\.0$/, "") + "%" : "n/a");
  const axisName = (name) => String(name).trim().replace(/_+/g, " ");
  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  function toCsv(header, rows) {
    const q = (v) => {
      const s = v == null ? "" : String(v);
      return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    return [header.map(q).join(",")].concat(rows.map((r) => r.map(q).join(","))).join("\n") + "\n";
  }

  /* ---------- SVG charts (colours come from flow.css classes) ---------- */
  function niceTicks(lo, hi, count) {
    if (!(hi > lo)) return [lo];
    const raw = (hi - lo) / (count || 5);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
    return out;
  }
  /* Layout is a small value per chart, computed from the width the chart will
   * actually be drawn at. The SVG is never stretched beyond that width (see
   * svgOpen), so 12px text in the viewBox is 12px on screen. */
  function layout(width, opts) {
    const o = opts || {};
    const W = Math.max(240, Math.round(width || 720));
    const narrow = W < 520;
    const M = { l: narrow ? 56 : 64, r: 24, t: (o.top || 0) + 16, b: 52 };
    let H;
    if (o.square) {
      // Equal pixels per unit on both axes: the inner plot is a square.
      const side = Math.min(W, o.maxSide || 560) - M.l - M.r;
      return { W: side + M.l + M.r, H: side + M.t + M.b, M, narrow, iw: side, ih: side };
    }
    H = (narrow ? 280 : Math.max(320, Math.min(400, Math.round(W * 0.42)))) + (o.top || 0);
    return { W, H, M, narrow, iw: W - M.l - M.r, ih: H - M.t - M.b };
  }
  function frame(L, xDom, yDom) {
    const xs = (v) => L.M.l + ((v - xDom[0]) / (xDom[1] - xDom[0] || 1)) * L.iw;
    const ys = (v) => L.M.t + (1 - (v - yDom[0]) / (yDom[1] - yDom[0] || 1)) * L.ih;
    return { xs, ys, iw: L.iw, ih: L.ih };
  }
  const n1 = (v) => v.toFixed(1);
  let clipSeq = 0;
  function svgOpen(L, aria, cls) {
    return '<svg class="flow-chart' + (cls ? " " + cls : "") + '" width="' + L.W + '" height="' + L.H + '" viewBox="0 0 ' + L.W + " " + L.H +
      '" style="max-width:' + L.W + 'px" role="img" aria-label="' + esc(aria) + '">';
  }
  function clipRect(L) {
    const id = "fc" + ++clipSeq;
    return { id, def: '<defs><clipPath id="' + id + '"><rect x="' + L.M.l + '" y="' + L.M.t + '" width="' + L.iw + '" height="' + L.ih + '"/></clipPath></defs>' };
  }
  function axes(L, f, xTicks, yTicks, xLabel, yLabel, xFmt, yFmt) {
    const { W, H, M } = L;
    let s = "";
    yTicks.forEach((t) => {
      s += '<line class="grid" x1="' + M.l + '" x2="' + (W - M.r) + '" y1="' + n1(f.ys(t)) + '" y2="' + n1(f.ys(t)) + '"/>';
      s += '<text class="tick" x="' + (M.l - 8) + '" y="' + n1(f.ys(t) + 4) + '" text-anchor="end">' + esc(yFmt(t)) + "</text>";
    });
    xTicks.forEach((t) => {
      s += '<line class="tickmark" x1="' + n1(f.xs(t)) + '" x2="' + n1(f.xs(t)) + '" y1="' + (H - M.b) + '" y2="' + (H - M.b + 5) + '"/>';
      s += '<text class="tick" x="' + n1(f.xs(t)) + '" y="' + (H - M.b + 20) + '" text-anchor="middle">' + esc(xFmt(t)) + "</text>";
    });
    s += '<line class="axis" x1="' + M.l + '" x2="' + (W - M.r) + '" y1="' + (H - M.b) + '" y2="' + (H - M.b) + '"/>';
    s += '<text class="axis-label" x="' + (M.l + f.iw / 2) + '" y="' + (H - 8) + '" text-anchor="middle">' + esc(xLabel) + "</text>";
    s += '<text class="axis-label" transform="translate(14 ' + (M.t + f.ih / 2) + ') rotate(-90)" text-anchor="middle">' + esc(yLabel) + "</text>";
    return s;
  }
  // Invisible hit targets carry the tooltip. They take keyboard focus only when a
  // chart has few of them; dense charts offer the data table instead of hundreds of tab stops.
  const hit = (x, y, tip, r, focusable) =>
    '<circle' + (focusable ? ' tabindex="0" role="button" aria-label="' + esc(tip) + '"' : "") + ' class="hit" data-tip="' + esc(tip) + '" cx="' + n1(x) + '" cy="' + n1(y) + '" r="' + (r || 12) + '"><title>' + esc(tip) + "</title></circle>";
  const MAX_FOCUSABLE = 40;
  // Marker shapes give each series an identity that does not depend on colour.
  function marker(shape, x, y, cls) {
    const c = ' class="dot ' + cls + '"';
    if (shape === "square") return "<rect" + c + ' x="' + n1(x - 4) + '" y="' + n1(y - 4) + '" width="8" height="8"/>';
    if (shape === "diamond") return "<path" + c + ' d="M' + n1(x) + " " + n1(y - 5.5) + "L" + n1(x + 5.5) + " " + n1(y) + "L" + n1(x) + " " + n1(y + 5.5) + "L" + n1(x - 5.5) + " " + n1(y) + 'Z"/>';
    return "<circle" + c + ' cx="' + n1(x) + '" cy="' + n1(y) + '" r="4.5"/>';
  }

  /**
   * Multi-series line chart. cfg: {width, series:[{name,cls,dash,shape,points:[{x,y,lo,hi,tip}]}], xTicks, xLabel, yLabel,
   * xFmt, yFmt, yDomain, hLine:{y,label}, aria}. Series are identified by the legend (legendHtml), never by
   * labels placed at line ends, so converging endpoints cannot collide.
   */
  function seriesChartSvg(cfg) {
    const L = layout(cfg.width);
    const all = cfg.series.flatMap((s) => s.points);
    const xv = all.map((p) => p.x);
    const xDom = [Math.min(...xv), Math.max(...xv)];
    const yDom = cfg.yDomain || [0, Math.max(...all.map((p) => (Number.isFinite(p.hi) ? p.hi : p.y)), cfg.hLine ? cfg.hLine.y : 0) * 1.08];
    const f = frame(L, xDom, yDom);
    const xFmt = cfg.xFmt || String;
    const yFmt = cfg.yFmt || ((v) => fmt(v, 2));
    const focusable = all.length <= MAX_FOCUSABLE;
    let s = svgOpen(L, cfg.aria || "Chart");
    s += axes(L, f, cfg.xTicks || xv, niceTicks(yDom[0], yDom[1], L.narrow ? 4 : 5), cfg.xLabel, cfg.yLabel, xFmt, yFmt);
    if (cfg.hLine) {
      s += '<line class="target" x1="' + L.M.l + '" x2="' + (L.W - L.M.r) + '" y1="' + n1(f.ys(cfg.hLine.y)) + '" y2="' + n1(f.ys(cfg.hLine.y)) + '"/>';
      s += '<text class="target-label" x="' + (L.M.l + 6) + '" y="' + n1(f.ys(cfg.hLine.y) - 6) + '">' + esc(cfg.hLine.label) + "</text>";
    }
    cfg.series.forEach((se) => {
      const pts = se.points.filter((p) => Number.isFinite(p.y));
      pts.forEach((p) => {
        if (Number.isFinite(p.lo) && Number.isFinite(p.hi))
          s += '<line class="whisker ' + se.cls + '" x1="' + n1(f.xs(p.x)) + '" x2="' + n1(f.xs(p.x)) + '" y1="' + n1(f.ys(p.lo)) + '" y2="' + n1(f.ys(p.hi)) + '"/>';
      });
      s += '<polyline class="line ' + se.cls + '"' + (se.dash ? ' stroke-dasharray="' + se.dash + '"' : "") + ' points="' +
        pts.map((p) => n1(f.xs(p.x)) + "," + n1(f.ys(p.y))).join(" ") + '"/>';
      pts.forEach((p) => {
        s += marker(se.shape, f.xs(p.x), f.ys(p.y), se.cls);
        s += hit(f.xs(p.x), f.ys(p.y), p.tip || se.name + ": " + yFmt(p.y), 13, focusable);
      });
    });
    return s + "</svg>";
  }

  /**
   * Predicted outcome along one control with an explicit interval band.
   * cfg: {width, xs, mean, lower, upper, bandClass, observed:[{x,y,tip}], projected:[{x,y,tip}],
   *       next:[{x,tip,label}], xLabel, yLabel, aria}
   * observed: runs that match the slice. projected: runs whose other controls differ, drawn only
   * when the caller asks for them, as hollow markers. The y-range covers every drawn point.
   */
  function bandChartSvg(cfg) {
    const next = cfg.next || [];
    const L = layout(cfg.width, { top: next.length ? 22 : 0 });
    const pts = (cfg.observed || []).concat(cfg.projected || []);
    const yMin = Math.min(...cfg.lower, ...pts.map((o) => o.y));
    const yMax = Math.max(...cfg.upper, ...pts.map((o) => o.y));
    const pad = (yMax - yMin) * 0.08 || Math.abs(yMax) * 0.1 || 1;
    const yDom = [yMin - pad, yMax + pad];
    const xDom = [cfg.xs[0], cfg.xs[cfg.xs.length - 1]];
    const f = frame(L, xDom, yDom);
    const focusable = pts.length + next.length + 13 <= MAX_FOCUSABLE;
    let s = svgOpen(L, cfg.aria || "Predicted outcome");
    s += axes(L, f, niceTicks(xDom[0], xDom[1], L.narrow ? 4 : 6), niceTicks(yDom[0], yDom[1], L.narrow ? 4 : 5), cfg.xLabel, cfg.yLabel, (v) => fmt(v, 3), (v) => fmt(v, 3));
    const up = cfg.xs.map((x, i) => n1(f.xs(x)) + "," + n1(f.ys(cfg.upper[i])));
    const dn = cfg.xs.map((x, i) => n1(f.xs(x)) + "," + n1(f.ys(cfg.lower[i]))).reverse();
    s += '<polygon class="band ' + (cfg.bandClass || "s-pre") + '" points="' + up.concat(dn).join(" ") + '"/>';
    s += '<polyline class="bound ' + (cfg.bandClass || "s-pre") + '" points="' + up.join(" ") + '"/>';
    s += '<polyline class="bound ' + (cfg.bandClass || "s-pre") + '" points="' + dn.join(" ") + '"/>';
    s += '<polyline class="line s-pre" points="' + cfg.xs.map((x, i) => n1(f.xs(x)) + "," + n1(f.ys(cfg.mean[i]))).join(" ") + '"/>';
    const every = Math.max(1, Math.floor(cfg.xs.length / 12));
    cfg.xs.forEach((x, i) => {
      if (i % every === 0 || i === cfg.xs.length - 1)
        s += hit(f.xs(x), f.ys(cfg.mean[i]), (cfg.tips && cfg.tips[i]) || fmt(x) + ": " + fmt(cfg.mean[i]), 9, focusable);
    });
    next.forEach((n, i) => {
      const x = f.xs(n.x);
      s += '<line class="nextline" x1="' + n1(x) + '" x2="' + n1(x) + '" y1="' + (L.M.t - 6) + '" y2="' + (L.H - L.M.b) + '"/>';
      s += '<text class="next-label" x="' + n1(x) + '" y="' + (L.M.t - 10) + '" text-anchor="middle">' + esc(n.label || i + 1) + "</text>";
      s += hit(x, L.M.t - 14, n.tip, 10, focusable);
    });
    (cfg.projected || []).forEach((o) => {
      s += '<circle class="proj" cx="' + n1(f.xs(o.x)) + '" cy="' + n1(f.ys(o.y)) + '" r="4"/>';
      s += hit(f.xs(o.x), f.ys(o.y), o.tip, 9, focusable);
    });
    (cfg.observed || []).forEach((o) => {
      s += '<circle class="obs" cx="' + n1(f.xs(o.x)) + '" cy="' + n1(f.ys(o.y)) + '" r="4.5"/>';
      s += hit(f.xs(o.x), f.ys(o.y), o.tip, 11, focusable);
    });
    return s + "</svg>";
  }

  /**
   * Reference (x) against prediction (y), with interval whiskers and a +/- tolerance corridor.
   * cfg: {width, truth, prediction, lower, upper, tolerance, xLabel, yLabel, aria, tips}
   * Both axes share one domain and one pixel scale, so the equality line is at 45 degrees.
   */
  function parityChartSvg(cfg) {
    const L = layout(cfg.width, { square: true, maxSide: cfg.maxSide });
    const vals = cfg.truth.concat(cfg.prediction, cfg.lower || [], cfg.upper || []).filter(Number.isFinite);
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const pad = (hi - lo) * 0.05 || 1;
    const dom = [lo - pad, hi + pad];
    const f = frame(L, dom, dom);
    const ticks = niceTicks(dom[0], dom[1], L.iw < 300 ? 4 : 5);
    const clip = clipRect(L);
    let s = svgOpen(L, cfg.aria || "Prediction against reference") + clip.def;
    s += axes(L, f, ticks, ticks, cfg.xLabel, cfg.yLabel, (v) => fmt(v, 3), (v) => fmt(v, 3));
    s += '<g clip-path="url(#' + clip.id + ')">';
    if (Number.isFinite(cfg.tolerance)) {
      const t = cfg.tolerance;
      s += '<polygon class="band s-gp" points="' + [[dom[0], dom[0] - t], [dom[1], dom[1] - t], [dom[1], dom[1] + t], [dom[0], dom[0] + t]]
        .map((p) => n1(f.xs(p[0])) + "," + n1(f.ys(p[1]))).join(" ") + '"/>';
    }
    s += '<line class="target" x1="' + n1(f.xs(dom[0])) + '" y1="' + n1(f.ys(dom[0])) + '" x2="' + n1(f.xs(dom[1])) + '" y2="' + n1(f.ys(dom[1])) + '"/></g>';
    cfg.truth.forEach((t, i) => {
      const p = cfg.prediction[i];
      if (cfg.lower && Number.isFinite(cfg.lower[i]))
        s += '<line class="whisker s-pre" x1="' + n1(f.xs(t)) + '" x2="' + n1(f.xs(t)) + '" y1="' + n1(f.ys(cfg.lower[i])) + '" y2="' + n1(f.ys(cfg.upper[i])) + '"/>';
      s += '<circle class="dot s-pre" cx="' + n1(f.xs(t)) + '" cy="' + n1(f.ys(p)) + '" r="3"/>';
      if (cfg.tips) s += hit(f.xs(t), f.ys(p), cfg.tips[i], 6, cfg.truth.length <= MAX_FOCUSABLE);
    });
    return s + "</svg>";
  }

  function legendHtml(items) {
    return (
      '<ul class="chart-legend">' +
      items.map((i) => '<li><span class="swatch ' + i.cls + '"' + (i.dash ? ' data-dash="1"' : "") + (i.shape ? ' data-shape="' + i.shape + '"' : "") + "></span>" + esc(i.name) + "</li>").join("") +
      "</ul>"
    );
  }

  /** Tooltip for any element carrying data-tip, on hover and keyboard focus. Escape hides it.
   * Positioned in document coordinates; flips below the target when there is no room above. */
  function attachTips(container) {
    if (typeof document === "undefined" || !container) return;
    if (container.dataset.tipsBound) return;
    container.dataset.tipsBound = "true";
    let tip = document.getElementById("flow-tip");
    if (!tip) {
      tip = document.createElement("div");
      tip.id = "flow-tip";
      tip.setAttribute("role", "tooltip");
      tip.hidden = true;
      document.body.appendChild(tip);
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") tip.hidden = true;
      });
    }
    const show = (e) => {
      const t = e.target.closest && e.target.closest("[data-tip]");
      if (!t) return;
      tip.textContent = t.getAttribute("data-tip");
      tip.hidden = false;
      const r = t.getBoundingClientRect();
      const w = tip.offsetWidth;
      const h = tip.offsetHeight;
      const left = Math.max(8, Math.min(document.documentElement.clientWidth - w - 8, r.left + r.width / 2 - w / 2));
      const above = r.top - h - 8;
      const top = above >= 8 ? above : r.bottom + 8;
      tip.style.left = window.scrollX + left + "px";
      tip.style.top = window.scrollY + top + "px";
    };
    const hide = () => {
      tip.hidden = true;
    };
    container.addEventListener("pointerover", show);
    container.addEventListener("focusin", show);
    container.addEventListener("pointerout", hide);
    container.addEventListener("focusout", hide);
  }

  /** Evenly spaced points along one control; the others held at `fixed`. */
  function sliceGrid(ranges, xIndex, fixed, m) {
    const count = m || 25;
    const xs = [];
    const points = [];
    for (let i = 0; i < count; i++) {
      const x = ranges[xIndex][0] + ((ranges[xIndex][1] - ranges[xIndex][0]) * i) / (count - 1);
      const row = fixed.slice();
      row[xIndex] = x;
      xs.push(x);
      points.push(row);
    }
    return { xs, points };
  }
  const midpoints = (ranges) => ranges.map((r) => (r[0] + r[1]) / 2);


  /* Draw a chart at its container's measured width and redraw it, from the data the
   * caller already holds, whenever that width changes. Resizing never re-queries the
   * model. Redraws are coalesced to one per animation frame. */
  const chartObservers = typeof WeakMap !== "undefined" ? new WeakMap() : null;
  function mountChart(container, draw) {
    if (!container) return;
    const old = chartObservers.get(container);
    if (old) old.disconnect();
    let last = 0;
    let queued = false;
    const paint = () => {
      queued = false;
      if (!container.isConnected) {
        observer && observer.disconnect();
        return;
      }
      const width = Math.round(container.clientWidth);
      if (width < 1 || width === last) return;
      last = width;
      container.innerHTML = draw(width);
      attachTips(container);
    };
    const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => {
      if (!queued) {
        queued = true;
        requestAnimationFrame(paint);
      }
    }) : null;
    if (observer) {
      chartObservers.set(container, observer);
      observer.observe(container);
    }
    paint();
  }
  /** Accessible numerical alternative to a chart, collapsed by default. */
  function dataTable(headers, rows, caption) {
    return '<details class="chart-data"><summary>' + esc(caption || "Chart data as a table") + '</summary><div class="table-scroll" tabindex="0" role="region" aria-label="' + esc(caption || "Chart data") +
      '"><table class="flow-table"><thead><tr>' + headers.map((h) => "<th>" + esc(h) + "</th>").join("") + "</tr></thead><tbody>" +
      rows.map((row) => "<tr>" + row.map((v) => '<td class="num">' + esc(typeof v === "number" ? fmt(v, 4) : v) + "</td>").join("") + "</tr>").join("") + "</tbody></table></div></details>";
  }
  /** Relative tolerance for "same setting" on a held control: a fraction of that control's range. */
  const SLICE_TOLERANCE = 0.01;
  /** True when every control other than the varied one is within SLICE_TOLERANCE of the held value. */
  function matchesSlice(controls, fixed, xIndex, ranges, tol) {
    const t = tol == null ? SLICE_TOLERANCE : tol;
    return controls.every((v, j) => j === xIndex || Math.abs(v - fixed[j]) <= Math.max(1e-12, Math.abs(ranges[j][1] - ranges[j][0]) * t));
  }

  const api = { SLICE_TOLERANCE, mountChart, dataTable, matchesSlice, fmt, pct, axisName, esc, toCsv, niceTicks, sliceGrid, midpoints, seriesChartSvg, bandChartSvg, parityChartSvg, legendHtml, attachTips };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DeltaFlow = api;
})(typeof window !== "undefined" ? window : globalThis);
