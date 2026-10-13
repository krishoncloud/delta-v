/* Delta-V "bring your runs" flow logic.  MOCK / FRONT-END ONLY.
 *
 * Pure functions, no DOM access at load time, so tests can require() this file.
 *
 * WHAT IS MOCK HERE
 *   - Fit / predict / next-run selection use a plain Gaussian process (kriging)
 *     running in the browser. It is a stand-in, NOT the Delta-V pretrained model.
 *   - The worked example and the evidence curves are synthetic fixtures.
 *   Every screen that uses them carries a MOCK label.
 *
 * BACKEND CONTRACT (for tasks 1-10; flip FLOW_MODE to "live" once these exist)
 *   POST /flow/validate  {csv, mapping}                  -> {ok, n_runs, params[], outcome, issues[]}
 *   POST /flow/fit       {params[], ranges[], X[][], y[], outcome} -> {model_id, loo_rel_l2}
 *   POST /flow/predict   {model_id, points[][]}          -> {mean[], std[]}
 *   POST /flow/next      {model_id, k}                   -> {points[][], mean[], std[]}
 *   static/worked-example.json            fixture, see "status"
 *   static/sample-efficiency-results.json fixture, see "status"
 *   "status" values: "mock" | "pending" | "measured". Only "measured" may
 *   put a number in the landing-page claim (see claimText).
 */
(function (root) {
  "use strict";

  const FLOW_MODE = "mock"; // "mock" | "live"
  const LIMITS = { minRuns: 3, fewRuns: 5, maxRuns: 200, maxParams: 3 };
  const ID_NAME = /^(run|run_?id|id|case|case_?id|sample|index|name)$/i;

  /* ---------- CSV ---------- */
  function parseCsv(text) {
    const s = String(text == null ? "" : text).replace(/^﻿/, "");
    const rows = [];
    let row = [];
    let cell = "";
    let q = false;
    for (let i = 0; i < s.length; i++) {
      const c = s[i];
      if (q) {
        if (c === '"') {
          if (s[i + 1] === '"') {
            cell += '"';
            i++;
          } else q = false;
        } else cell += c;
      } else if (c === '"' && cell === "") q = true;
      else if (c === ",") {
        row.push(cell);
        cell = "";
      } else if (c === "\n" || c === "\r") {
        if (c === "\r" && s[i + 1] === "\n") i++;
        row.push(cell);
        cell = "";
        rows.push(row);
        row = [];
      } else cell += c;
    }
    if (cell !== "" || row.length) {
      row.push(cell);
      rows.push(row);
    }
    return {
      rows: rows.filter((r) => !(r.length === 1 && r[0].trim() === "")),
      unterminated: q,
    };
  }

  const issue = (level, msg) => ({ level, msg });
  const hasError = (issues) => issues.some((i) => i.level === "error");

  /** Read the table and classify columns. Does not decide which are inputs. */
  function parseRuns(text) {
    const issues = [];
    const { rows, unterminated } = parseCsv(text);
    if (unterminated) {
      issues.push(issue("error", "A quoted cell is never closed. Check for a stray quotation mark."));
      return { header: [], columns: [], n: 0, issues };
    }
    if (!rows.length) {
      issues.push(issue("error", "The file is empty. Expected a header row and one row per run."));
      return { header: [], columns: [], n: 0, issues };
    }
    const header = rows[0].map((h) => h.trim());
    if (header.some((h) => h === ""))
      issues.push(issue("error", "Every column needs a name in the header row."));
    const seen = new Set();
    header.forEach((h) => {
      if (h && seen.has(h.toLowerCase()))
        issues.push(issue("error", 'Two columns are both named "' + h + '". Rename one.'));
      seen.add(h.toLowerCase());
    });
    const data = rows.slice(1);
    if (!data.length) {
      issues.push(issue("error", "The file has a header but no runs."));
      return { header, columns: [], n: 0, issues };
    }
    const bad = data.map((r, i) => [i, r.length]).filter(([, len]) => len !== header.length);
    if (bad.length) {
      const [i, len] = bad[0];
      issues.push(
        issue(
          "error",
          "Line " + (i + 2) + " has " + len + (len === 1 ? " cell" : " cells") + " but the header has " + header.length +
            (bad.length > 1 ? " (" + (bad.length - 1) + " more rows have the same problem)." : "."),
        ),
      );
      return { header, columns: [], n: data.length, issues }; // later checks would only repeat this problem
    }
    const columns = header.map((name, index) => {
      const cells = data.map((r) => (r[index] == null ? "" : String(r[index]).trim()));
      const blanks = cells.map((v, i) => (v === "" ? i : -1)).filter((i) => i >= 0);
      const nonNumeric = cells
        .map((v, i) => (v !== "" && !Number.isFinite(Number(v)) ? i : -1))
        .filter((i) => i >= 0);
      const numeric = !blanks.length && !nonNumeric.length;
      const uniqueLabels = nonNumeric.length > 0 && !blanks.length && new Set(cells).size === cells.length;
      let kind = numeric ? "numeric" : "text";
      if (ID_NAME.test(name) || uniqueLabels) kind = "id";
      else if (!nonNumeric.length && blanks.length) kind = "numeric-gaps";
      const values = kind === "numeric" ? cells.map(Number) : null;
      return {
        name,
        index,
        kind,
        blanks,
        nonNumeric,
        values,
        cells,
        min: values ? Math.min(...values) : null,
        max: values ? Math.max(...values) : null,
      };
    });
    columns.forEach((c) => {
      if (c.kind === "text" && c.nonNumeric.length) {
        const i = c.nonNumeric[0];
        issues.push(
          issue(
            "error",
            'Column "' + c.name + '" has a non-numeric value ("' + c.cells[i] + '") on line ' + (i + 2) +
              ". Design parameters and outcomes must be numbers. If this column only labels runs, name it run_id.",
          ),
        );
      }
      if (c.kind === "numeric-gaps") {
        issues.push(
          issue(
            "error",
            'Column "' + c.name + '" has ' + c.blanks.length + " empty cell" + (c.blanks.length > 1 ? "s" : "") +
              " (first on line " + (c.blanks[0] + 2) + "). Every run needs a value.",
          ),
        );
      }
    });
    return { header, columns, n: data.length, issues };
  }

  function defaultMapping(parsed) {
    const nums = parsed.columns.filter((c) => c.kind === "numeric").map((c) => c.name);
    if (nums.length < 2) return { params: nums.slice(0, 1), outcome: null };
    return { params: nums.slice(0, -1), outcome: nums[nums.length - 1] };
  }

  /** Apply a column mapping and design ranges; collect every problem found. */
  function buildDataset(parsed, mapping, designRanges) {
    const issues = [];
    const col = (name) => parsed.columns.find((c) => c.name === name);
    const params = (mapping.params || []).map(col);
    const out = mapping.outcome ? col(mapping.outcome) : null;
    if (!params.length || params.some((c) => !c || c.kind !== "numeric"))
      issues.push(issue("error", "Choose at least one numeric design-parameter column."));
    if (params.length > LIMITS.maxParams)
      issues.push(
        issue("error", "The mock workbench handles up to " + LIMITS.maxParams + " design parameters; you selected " + params.length + "."),
      );
    if (!out || out.kind !== "numeric") issues.push(issue("error", "Choose one numeric outcome column."));
    if (out && (mapping.params || []).includes(out.name))
      issues.push(issue("error", "The outcome column cannot also be a design parameter."));
    const n = parsed.n;
    if (n && n < LIMITS.minRuns)
      issues.push(issue("error", "Only " + n + " run" + (n === 1 ? "" : "s") + ". At least " + LIMITS.minRuns + " are needed to fit anything."));
    else if (n && n < LIMITS.fewRuns)
      issues.push(issue("warn", "Only " + n + " runs. Predictions will carry wide uncertainty until more runs are added."));
    if (n > LIMITS.maxRuns)
      issues.push(issue("error", "More than " + LIMITS.maxRuns + " runs is outside what the mock workbench accepts."));
    if (hasError(issues)) return { ok: false, issues };

    const X = [];
    for (let i = 0; i < n; i++) X.push(params.map((c) => c.values[i]));
    const y = out.values.slice();
    const dataRanges = params.map((c) => [c.min, c.max]);
    const ranges = (designRanges && designRanges.length === params.length ? designRanges : dataRanges).map((r) => [
      Number(r[0]),
      Number(r[1]),
    ]);
    ranges.forEach((r, j) => {
      if (!Number.isFinite(r[0]) || !Number.isFinite(r[1]) || r[0] >= r[1])
        issues.push(issue("error", 'Design range for "' + params[j].name + '" needs a minimum below its maximum.'));
      else if (r[0] > dataRanges[j][0] || r[1] < dataRanges[j][1])
        issues.push(issue("error", 'Design range for "' + params[j].name + '" must include all your runs (' + dataRanges[j][0] + " to " + dataRanges[j][1] + ")."));
    });
    params.forEach((c) => {
      if (c.min === c.max)
        issues.push(issue("warn", '"' + c.name + '" is the same in every run, so nothing can be learned about it.'));
    });
    if (out.min === out.max) issues.push(issue("warn", "The outcome is identical in every run."));
    const keys = X.map((r) => r.join("|"));
    if (new Set(keys).size < keys.length)
      issues.push(issue("warn", "Some runs repeat the same design point. Duplicates add no new information."));
    return {
      ok: !hasError(issues),
      issues,
      n,
      paramNames: params.map((c) => c.name),
      outcomeName: out.name,
      X,
      y,
      dataRanges,
      ranges,
    };
  }

  /* ---------- Gaussian process stand-in ---------- */
  function normX(X, ranges) {
    return X.map((row) => row.map((v, j) => (ranges[j][1] > ranges[j][0] ? (v - ranges[j][0]) / (ranges[j][1] - ranges[j][0]) : 0)));
  }
  function kern(a, b, ls) {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
      const d = (a[i] - b[i]) / ls;
      s += d * d;
    }
    return Math.exp(-0.5 * s);
  }
  function cholesky(K, n) {
    const L = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j <= i; j++) {
        let sum = K[i * n + j];
        for (let k = 0; k < j; k++) sum -= L[i * n + k] * L[j * n + k];
        if (i === j) {
          if (sum <= 1e-12) return null;
          L[i * n + i] = Math.sqrt(sum);
        } else L[i * n + j] = sum / L[j * n + j];
      }
    return L;
  }
  function solveLower(L, n, b) {
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= L[i * n + k] * x[k];
      x[i] = s / L[i * n + i];
    }
    return x;
  }
  function solveUpper(L, n, b) {
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = b[i];
      for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
      x[i] = s / L[i * n + i];
    }
    return x;
  }
  function factor(Xn, ls, noise) {
    const n = Xn.length;
    let nz = noise;
    for (let attempt = 0; attempt < 6; attempt++) {
      const K = new Float64Array(n * n);
      for (let i = 0; i < n; i++)
        for (let j = 0; j <= i; j++) {
          const v = kern(Xn[i], Xn[j], ls) + (i === j ? nz : 0);
          K[i * n + j] = v;
          K[j * n + i] = v;
        }
      const L = cholesky(K, n);
      if (L) return { L, noise: nz };
      nz *= 10;
    }
    throw new Error("The runs could not be fitted (the design points are numerically degenerate).");
  }

  function fitAt(Xn, y, ls, noise) {
    const n = Xn.length;
    const ym = y.reduce((a, b) => a + b, 0) / n;
    const sd = Math.sqrt(y.reduce((a, b) => a + (b - ym) * (b - ym), 0) / n) || 1;
    const ys = y.map((v) => (v - ym) / sd);
    const { L, noise: nz } = factor(Xn, ls, noise);
    const alpha = solveUpper(L, n, solveLower(L, n, ys));
    const diag = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const e = new Float64Array(n);
      e[i] = 1;
      const x = solveLower(L, n, e);
      let s = 0;
      for (let k = 0; k < n; k++) s += x[k] * x[k];
      diag[i] = s;
    }
    let se = 0;
    let sy = 0;
    let zz = 0;
    for (let i = 0; i < n; i++) {
      const r = (alpha[i] / diag[i]) * sd;
      se += r * r;
      sy += y[i] * y[i];
      zz += (alpha[i] * alpha[i]) / diag[i]; // squared leave-one-out z-score
    }
    // If leave-one-out errors are larger than the model claims, widen the band to match (never narrow it).
    const varScale = Math.max(1, zz / n);
    return { Xn, y, n, ls, noise: nz, ym, sd, L, alpha, varScale, looRelL2: sy > 0 ? Math.sqrt(se / sy) : null };
  }

  /** Fit, choosing the length scale by leave-one-out error. */
  function fit(X, y, ranges) {
    const Xn = normX(X, ranges);
    let best = null;
    for (const ls of [0.15, 0.25, 0.35, 0.5, 0.8, 1.2]) {
      const m = fitAt(Xn, y, ls, 1e-4);
      if (m.looRelL2 != null && (!best || m.looRelL2 < best.looRelL2)) best = m;
    }
    best = best || fitAt(Xn, y, 0.35, 1e-4);
    best.ranges = ranges;
    return best;
  }

  /** points are raw (not normalised) rows. Returns mean and std in outcome units. */
  function predict(model, points) {
    const Pn = normX(points, model.ranges);
    const mean = [];
    const std = [];
    Pn.forEach((p) => {
      const kv = new Float64Array(model.n);
      for (let i = 0; i < model.n; i++) kv[i] = kern(model.Xn[i], p, model.ls);
      let m = 0;
      for (let i = 0; i < model.n; i++) m += kv[i] * model.alpha[i];
      const v = solveLower(model.L, model.n, kv);
      let vv = 0;
      for (let i = 0; i < model.n; i++) vv += v[i] * v[i];
      mean.push(model.ym + model.sd * m);
      std.push(model.sd * Math.sqrt(Math.max(0, 1 - vv) * model.varScale));
    });
    return { mean, std };
  }

  function halton(i, base) {
    let f = 1;
    let r = 0;
    while (i > 0) {
      f /= base;
      r += f * (i % base);
      i = Math.floor(i / base);
    }
    return r;
  }
  const PRIMES = [2, 3, 5, 7];

  /** Greedy: add the point with the largest predictive std, k times. Returns raw points. */
  function suggestNext(model, k) {
    const d = model.ranges.length;
    const poolSize = model.n > 60 ? 250 : 500;
    const pool = [];
    for (let i = 1; i <= poolSize; i++) pool.push(PRIMES.slice(0, d).map((p) => halton(i + 17, p)));
    let design = model.Xn.slice();
    const chosen = [];
    for (let j = 0; j < k; j++) {
      const n = design.length;
      const { L } = factor(design, model.ls, model.noise);
      let bestI = -1;
      let bestV = -1;
      pool.forEach((p, idx) => {
        const kv = new Float64Array(n);
        for (let i = 0; i < n; i++) kv[i] = kern(design[i], p, model.ls);
        const v = solveLower(L, n, kv);
        let vv = 0;
        for (let i = 0; i < n; i++) vv += v[i] * v[i];
        const variance = 1 - vv;
        if (variance > bestV) {
          bestV = variance;
          bestI = idx;
        }
      });
      chosen.push(pool[bestI]);
      design = design.concat([pool[bestI]]);
      pool.splice(bestI, 1);
    }
    const raw = chosen.map((u) => u.map((v, jj) => model.ranges[jj][0] + v * (model.ranges[jj][1] - model.ranges[jj][0])));
    const pr = predict(model, raw);
    return { points: raw, mean: pr.mean, std: pr.std };
  }

  /* ---------- Synthetic stand-in solver, used only by the worked example ---------- */
  function wobble(a, b) {
    const x = Math.sin(a * 12.9898 + b * 78.233) * 43758.5453;
    return x - Math.floor(x) - 0.5;
  }
  const truths = {
    pipe_bend_v1: (p) => {
      const v = p[0];
      const r = p[1];
      return Math.round(11.5 * Math.pow(v, 1.85) * (1 + 2.4 / r) * (1 + 0.03 * wobble(v, r)) * 100) / 100;
    },
  };

  /* ---------- Evidence helpers ---------- */
  /** Interpolated runs needed to reach target error, or null if never reached. */
  function runsToTarget(ks, errs, target) {
    for (let i = 0; i < ks.length; i++) {
      if (!Number.isFinite(errs[i])) continue;
      if (errs[i] <= target) {
        if (i === 0) return ks[0];
        const e0 = errs[i - 1];
        const e1 = errs[i];
        const t = (e0 - target) / (e0 - e1);
        return Math.exp(Math.log(ks[i - 1]) + t * (Math.log(ks[i]) - Math.log(ks[i - 1])));
      }
    }
    return null;
  }

  /** Fail closed: only a complete "measured" file stays measured. */
  function normalizeEvidence(ev) {
    const empty = { status: "pending", label: "Sample-efficiency sweep not run yet." };
    if (!ev || typeof ev !== "object") return empty;
    const arms = ev.arms || {};
    const names = ["scratch", "pretrained", "gp"];
    const okArms = names.every((n) => {
      const a = arms[n];
      return a && Array.isArray(a.k) && Array.isArray(a.rel_l2) && a.k.length > 1 && a.k.length === a.rel_l2.length;
    });
    if (ev.status === "measured" && !okArms) return { ...empty, label: "Results file is incomplete, so no claim is made." };
    if (ev.status === "measured" || ev.status === "mock") return okArms ? ev : { ...empty };
    return { ...ev, status: "pending" };
  }

  function evidenceSummary(ev) {
    if (!ev || !ev.arms || !ev.arms.scratch) return null;
    const t = ev.target_rel_l2;
    const r = {};
    ["scratch", "pretrained", "gp"].forEach((n) => {
      r[n] = runsToTarget(ev.arms[n].k, ev.arms[n].rel_l2, t);
    });
    r.ratio = r.scratch && r.pretrained ? r.scratch / r.pretrained : null;
    return r;
  }

  /** The only place a runs-saved number may reach the landing page. */
  function claimText(ev) {
    const e = normalizeEvidence(ev);
    if (e.status === "measured") {
      const s = evidenceSummary(e);
      if (s && s.ratio)
        return {
          state: "measured",
          text:
            "Measured: reaches the " + Math.round(e.target_rel_l2 * 100) + "% error target with " + s.ratio.toFixed(1) +
            "x fewer runs than training from scratch.",
        };
      return { state: "pending", text: "Measured results are in, but the target was not reached by both arms." };
    }
    if (e.status === "mock")
      return { state: "mock", text: "Not yet measured. The curves on the Evidence page are mock." };
    return { state: "pending", text: "Not yet measured. The sweep that tests this has not run." };
  }

  /* ---------- Formatting ---------- */
  function fmt(v, digits) {
    if (!Number.isFinite(v)) return "n/a";
    const d = digits == null ? 3 : digits;
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a >= 10000 || a < 0.01) return v.toExponential(d - 1);
    return Number(v.toPrecision(d)).toString();
  }
  const pct = (v) => (Number.isFinite(v) ? (v * 100).toFixed(1) + "%" : "n/a");
  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  function toCsv(header, rows) {
    const q = (v) => {
      const s = String(v);
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
  // Layout is set per chart from the available width, so text stays a readable size on phones.
  let W = 720;
  let H = 340;
  let M = { l: 58, r: 118, t: 18, b: 46 };
  function setLayout(width) {
    W = Math.max(300, Math.min(860, Math.round(width || 720)));
    const narrow = W < 520;
    H = narrow ? 300 : 340;
    M = narrow ? { l: 50, r: 16, t: 18, b: 44 } : { l: 58, r: 118, t: 18, b: 46 };
    return narrow;
  }
  function frame(xDom, yDom, xLog) {
    const iw = W - M.l - M.r;
    const ih = H - M.t - M.b;
    const xs = xLog
      ? (v) => M.l + ((Math.log(v) - Math.log(xDom[0])) / (Math.log(xDom[1]) - Math.log(xDom[0]))) * iw
      : (v) => M.l + ((v - xDom[0]) / (xDom[1] - xDom[0] || 1)) * iw;
    const ys = (v) => M.t + (1 - (v - yDom[0]) / (yDom[1] - yDom[0] || 1)) * ih;
    return { xs, ys, iw, ih };
  }
  function axes(f, xTicks, yTicks, xLabel, yLabel, xFmt, yFmt) {
    let s = "";
    yTicks.forEach((t) => {
      s += '<line class="grid" x1="' + M.l + '" x2="' + (W - M.r) + '" y1="' + f.ys(t).toFixed(1) + '" y2="' + f.ys(t).toFixed(1) + '"/>';
      s += '<text class="tick" x="' + (M.l - 8) + '" y="' + (f.ys(t) + 4).toFixed(1) + '" text-anchor="end">' + esc(yFmt(t)) + "</text>";
    });
    xTicks.forEach((t) => {
      s += '<line class="tickmark" x1="' + f.xs(t).toFixed(1) + '" x2="' + f.xs(t).toFixed(1) + '" y1="' + (H - M.b) + '" y2="' + (H - M.b + 5) + '"/>';
      s += '<text class="tick" x="' + f.xs(t).toFixed(1) + '" y="' + (H - M.b + 19) + '" text-anchor="middle">' + esc(xFmt(t)) + "</text>";
    });
    s += '<line class="axis" x1="' + M.l + '" x2="' + (W - M.r) + '" y1="' + (H - M.b) + '" y2="' + (H - M.b) + '"/>';
    s += '<text class="axis-label" x="' + (M.l + f.iw / 2) + '" y="' + (H - 6) + '" text-anchor="middle">' + esc(xLabel) + "</text>";
    s += '<text class="axis-label" transform="translate(14 ' + (M.t + f.ih / 2) + ') rotate(-90)" text-anchor="middle">' + esc(yLabel) + "</text>";
    return s;
  }
  const watermark = (txt) =>
    txt
      ? '<text class="watermark" x="' + (M.l + (W - M.l - M.r) / 2) + '" y="' + (M.t + (H - M.t - M.b) / 2 + 14) + '" text-anchor="middle" transform="rotate(-10 ' + (M.l + (W - M.l - M.r) / 2) + " " + (M.t + (H - M.t - M.b) / 2) + ')">' + esc(txt) + "</text>"
      : "";

  /**
   * Multi-series line chart. cfg: {series:[{name,cls,dash,points:[{x,y,lo,hi,tip}]}], xLog, xTicks, xLabel, yLabel,
   * xFmt, yFmt, yMax, hLine:{y,label}, watermark, aria}
   */
  function seriesChartSvg(cfg) {
    const narrow = setLayout(cfg.width);
    const all = cfg.series.flatMap((s) => s.points);
    const xv = all.map((p) => p.x);
    const xDom = [Math.min(...xv), Math.max(...xv)];
    let yHi = Math.max(...all.map((p) => (Number.isFinite(p.hi) ? p.hi : p.y)), cfg.hLine ? cfg.hLine.y : 0);
    if (cfg.yMax) yHi = Math.min(yHi, cfg.yMax);
    const yDom = [0, yHi * 1.08];
    const f = frame(xDom, yDom, cfg.xLog);
    const xFmt = cfg.xFmt || String;
    const yFmt = cfg.yFmt || ((v) => fmt(v, 2));
    let s = '<svg class="flow-chart" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(cfg.aria || "Chart") + '">';
    s += axes(f, cfg.xTicks || xv, niceTicks(yDom[0], yDom[1], 5), cfg.xLabel, cfg.yLabel, xFmt, yFmt);
    if (cfg.hLine) {
      s += '<line class="target" x1="' + M.l + '" x2="' + (W - M.r) + '" y1="' + f.ys(cfg.hLine.y).toFixed(1) + '" y2="' + f.ys(cfg.hLine.y).toFixed(1) + '"/>';
      s += '<text class="target-label" x="' + (M.l + 6) + '" y="' + (f.ys(cfg.hLine.y) - 6).toFixed(1) + '">' + esc(cfg.hLine.label) + "</text>";
    }
    s += watermark(cfg.watermark);
    const labelY = {};
    if (cfg.series.length <= 4 && !narrow) {
      const ends = cfg.series
        .map((se, i) => ({ i, y: f.ys(se.points.filter((p) => Number.isFinite(p.y)).slice(-1)[0].y) + 4 }))
        .sort((a, b) => a.y - b.y);
      ends.forEach((e, k) => {
        if (k > 0 && e.y - ends[k - 1].y < 16) e.y = ends[k - 1].y + 16;
        labelY[e.i] = e.y;
      });
    }
    cfg.series.forEach((se, si) => {
      const pts = se.points.filter((p) => Number.isFinite(p.y));
      if (pts.some((p) => Number.isFinite(p.lo) && Number.isFinite(p.hi))) {
        pts.forEach((p) => {
          if (Number.isFinite(p.lo))
            s += '<line class="whisker ' + se.cls + '" x1="' + f.xs(p.x).toFixed(1) + '" x2="' + f.xs(p.x).toFixed(1) + '" y1="' + f.ys(p.lo).toFixed(1) + '" y2="' + f.ys(p.hi).toFixed(1) + '"/>';
        });
      }
      s += '<polyline class="line ' + se.cls + '"' + (se.dash ? ' stroke-dasharray="' + se.dash + '"' : "") + ' points="' +
        pts.map((p) => f.xs(p.x).toFixed(1) + "," + f.ys(p.y).toFixed(1)).join(" ") + '"/>';
      pts.forEach((p) => {
        s += '<circle class="dot ' + se.cls + '" cx="' + f.xs(p.x).toFixed(1) + '" cy="' + f.ys(p.y).toFixed(1) + '" r="4"/>';
        s += '<circle class="hit" data-tip="' + esc(p.tip || se.name + ": " + yFmt(p.y)) + '" cx="' + f.xs(p.x).toFixed(1) + '" cy="' + f.ys(p.y).toFixed(1) + '" r="13"><title>' + esc(p.tip || se.name + ": " + yFmt(p.y)) + "</title></circle>";
      });
      const last = pts[pts.length - 1];
      if (last && cfg.series.length <= 4 && !narrow)
        s += '<text class="direct" x="' + (f.xs(last.x) + 9).toFixed(1) + '" y="' + labelY[si].toFixed(1) + '">' + esc(se.name) + "</text>";
    });
    return s + "</svg>";
  }

  /**
   * Outcome-vs-parameter chart with a +/-2 sigma band.
   * cfg: {xs, mean, std, observed:[{x,y,tip}], next:[{x,tip}], xLabel, yLabel, watermark, aria}
   */
  function outcomeChartSvg(cfg) {
    const narrow = setLayout(cfg.width);
    const lo = cfg.mean.map((m, i) => m - 2 * cfg.std[i]);
    const hi = cfg.mean.map((m, i) => m + 2 * cfg.std[i]);
    const obsY = cfg.observed.map((o) => o.y);
    const yMin = Math.min(...lo, ...obsY);
    const yMax = Math.max(...hi, ...obsY);
    const pad = (yMax - yMin) * 0.06 || 1;
    const yDom = [yMin - pad, yMax + pad];
    const xDom = [cfg.xs[0], cfg.xs[cfg.xs.length - 1]];
    const f = frame(xDom, yDom, false);
    const yFmt = (v) => fmt(v, 3);
    let s = '<svg class="flow-chart" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(cfg.aria || "Predicted outcome") + '">';
    s += axes(f, niceTicks(xDom[0], xDom[1], narrow ? 4 : 6), niceTicks(yDom[0], yDom[1], 5), cfg.xLabel, cfg.yLabel, (v) => fmt(v, 3), yFmt);
    s += watermark(cfg.watermark);
    const up = cfg.xs.map((x, i) => f.xs(x).toFixed(1) + "," + f.ys(hi[i]).toFixed(1));
    const dn = cfg.xs.map((x, i) => f.xs(x).toFixed(1) + "," + f.ys(lo[i]).toFixed(1)).reverse();
    s += '<polygon class="band s-pre" points="' + up.concat(dn).join(" ") + '"/>';
    s += '<polyline class="line s-pre" points="' + cfg.xs.map((x, i) => f.xs(x).toFixed(1) + "," + f.ys(cfg.mean[i]).toFixed(1)).join(" ") + '"/>';
    // Stagger labels vertically when two suggestions sit close together on the x axis.
    const rows = {};
    let prev = { x: -1e9, row: 0 };
    cfg.next
      .map((n, i) => ({ px: f.xs(n.x), i }))
      .sort((a, b) => a.px - b.px)
      .forEach((n) => {
        const row = n.px - prev.x < 26 ? (prev.row + 1) % 3 : 0;
        rows[n.i] = row;
        prev = { x: n.px, row };
      });
    cfg.next.forEach((n, i) => {
      s += '<line class="nextline" x1="' + f.xs(n.x).toFixed(1) + '" x2="' + f.xs(n.x).toFixed(1) + '" y1="' + M.t + '" y2="' + (H - M.b) + '"/>';
      s += '<text class="next-label" x="' + f.xs(n.x).toFixed(1) + '" y="' + (M.t + 11 + rows[i] * 13) + '" text-anchor="middle">N' + (i + 1) + "</text>";
      s += '<circle class="hit" data-tip="' + esc(n.tip) + '" cx="' + f.xs(n.x).toFixed(1) + '" cy="' + (M.t + 8) + '" r="13"><title>' + esc(n.tip) + "</title></circle>";
    });
    cfg.observed.forEach((o) => {
      s += '<circle class="obs" cx="' + f.xs(o.x).toFixed(1) + '" cy="' + f.ys(o.y).toFixed(1) + '" r="5"/>';
      s += '<circle class="hit" data-tip="' + esc(o.tip) + '" cx="' + f.xs(o.x).toFixed(1) + '" cy="' + f.ys(o.y).toFixed(1) + '" r="13"><title>' + esc(o.tip) + "</title></circle>";
    });
    return s + "</svg>";
  }

  function legendHtml(items) {
    return (
      '<ul class="chart-legend">' +
      items.map((i) => '<li><span class="swatch ' + i.cls + '"' + (i.dash ? ' data-dash="1"' : "") + "></span>" + esc(i.name) + "</li>").join("") +
      "</ul>"
    );
  }

  /** Small hover tooltip for any element carrying data-tip. Call after inserting SVG. */
  function attachTips(container) {
    if (typeof document === "undefined" || !container) return;
    let tip = document.getElementById("flow-tip");
    if (!tip) {
      tip = document.createElement("div");
      tip.id = "flow-tip";
      tip.setAttribute("role", "tooltip");
      tip.hidden = true;
      document.body.appendChild(tip);
    }
    const show = (e) => {
      const t = e.target.closest && e.target.closest("[data-tip]");
      if (!t) return;
      tip.textContent = t.getAttribute("data-tip");
      tip.hidden = false;
      const r = t.getBoundingClientRect();
      tip.style.left = Math.max(8, Math.min(window.innerWidth - tip.offsetWidth - 8, r.left + r.width / 2 - tip.offsetWidth / 2)) + "px";
      tip.style.top = window.scrollY + r.top - tip.offsetHeight - 8 + "px";
    };
    const hide = () => {
      tip.hidden = true;
    };
    container.addEventListener("pointerover", show);
    container.addEventListener("focusin", show);
    container.addEventListener("pointerout", hide);
    container.addEventListener("focusout", hide);
  }

  /** Evenly spaced curve through the design range along one parameter; others held at `fixed`. */
  function sliceGrid(ranges, xIndex, fixed, m) {
    const count = m || 60;
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

  /** Parse, validate and fit a worked-example fixture through the same path a user upload takes. */
  function sessionFromCsv(csv, designRanges) {
    const parsed = parseRuns(csv);
    if (hasError(parsed.issues)) return { ok: false, issues: parsed.issues };
    const mapping = defaultMapping(parsed);
    const ds = buildDataset(parsed, mapping, designRanges);
    if (!ds.ok) return { ok: false, issues: ds.issues };
    return { ok: true, parsed, mapping, ds, model: fit(ds.X, ds.y, ds.ranges) };
  }

  const api = {
    FLOW_MODE, LIMITS, parseCsv, parseRuns, defaultMapping, buildDataset, hasError,
    fit, predict, suggestNext, truths, runsToTarget, normalizeEvidence, evidenceSummary, claimText,
    fmt, pct, esc, toCsv, niceTicks, sliceGrid, midpoints, sessionFromCsv, seriesChartSvg, outcomeChartSvg, legendHtml, attachTips,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DeltaFlow = api;
})(typeof window !== "undefined" ? window : globalThis);
