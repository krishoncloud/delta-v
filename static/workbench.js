/* Workbench: bring your runs. MOCK / FRONT-END ONLY.
 *
 * Real: CSV parsing, column mapping and every validation message (in this browser).
 * Mock: the fit, predictions, uncertainty band and next-run suggestions come from
 * a Gaussian-process stand-in (DeltaFlow), not from the Delta-V model.
 * To connect the real flow, replace fitNow(), predictCurve() and nextRuns() with
 * calls to the endpoints listed at the top of flow.js. */
(function () {
  "use strict";
  const F = window.DeltaFlow;
  const $ = (id) => document.getElementById(id);
  const esc = F.esc;
  const STEPS = ["Describe", "Upload runs", "Fit", "Predict", "Next runs", "Report"];
  const MAX_FILE_BYTES = 2000000;

  const fresh = () => ({
    step: 0,
    maxStep: 0,
    spec: { name: "My design study", blindSweep: 100, target: 0.15 },
    source: null,
    truth: null,
    csv: "",
    fileName: "",
    parsed: null,
    roles: {},
    rangeEdits: {},
    ds: null,
    model: null,
    xIndex: 0,
    fixed: null,
    nextK: 4,
    next: null,
    rounds: [],
    message: "",
  });
  let W = fresh();
  let ready = false;

  /* ---------- helpers ---------- */
  function download(name, text, mime) {
    const url = URL.createObjectURL(new Blob([text], { type: mime }));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const issuesHtml = (issues) =>
    issues.length
      ? '<ul class="issues">' +
        issues
          .map((i) => '<li class="' + i.level + '"><span class="lvl">' + (i.level === "error" ? "ERROR" : "NOTE") + "</span><span>" + esc(i.msg) + "</span></li>")
          .join("") +
        "</ul>"
      : "";
  const tag = '<span class="mock-tag">MOCK</span>';

  /* ---------- ingest / validate ---------- */
  function currentMapping() {
    const cols = W.parsed.columns.filter((c) => c.kind === "numeric");
    return {
      params: cols.filter((c) => W.roles[c.name] === "param").map((c) => c.name),
      outcomes: cols.filter((c) => W.roles[c.name] === "outcome").map((c) => c.name),
    };
  }
  function currentRanges(params) {
    return params.map((n) => {
      const c = W.parsed.columns.find((x) => x.name === n);
      const e = W.rangeEdits[n];
      return e ? [e[0], e[1]] : [c.min, c.max];
    });
  }
  function validate() {
    const issues = W.parsed.issues.slice();
    W.ds = null;
    if (F.hasError(issues)) return issues;
    const m = currentMapping();
    if (m.outcomes.length > 1) issues.push({ level: "error", msg: "Choose only one outcome column. You selected " + m.outcomes.length + "." });
    const ds = F.buildDataset(W.parsed, { params: m.params, outcome: m.outcomes[0] || null }, currentRanges(m.params));
    ds.issues.forEach((i) => issues.push(i));
    if (ds.ok && !F.hasError(issues)) W.ds = ds;
    return issues;
  }
  function ingest(csv, name, keepRoles) {
    W.csv = csv;
    W.fileName = name || "pasted.csv";
    W.parsed = F.parseRuns(csv);
    W.model = null;
    W.next = null;
    const prev = W.roles;
    W.roles = {};
    const dm = F.defaultMapping(W.parsed);
    W.parsed.columns.forEach((c) => {
      if (c.kind !== "numeric") return;
      if (keepRoles && prev[c.name]) W.roles[c.name] = prev[c.name];
      else W.roles[c.name] = dm.params.includes(c.name) ? "param" : dm.outcome === c.name ? "outcome" : "ignore";
    });
    if (!keepRoles) W.rangeEdits = {};
  }
  async function loadExample() {
    try {
      const ex = await window.DeltaPages.getJson("worked-example.json");
      W = fresh();
      W.source = "example";
      W.truth = F.truths[ex.truth] || null;
      W.spec = { name: ex.name, blindSweep: ex.spec.blind_sweep_runs, target: ex.spec.target_rel_l2 };
      ingest(ex.runs_csv, "worked-example.csv (mock)");
      const names = W.parsed.columns.filter((c) => W.roles[c.name] === "param").map((c) => c.name);
      names.forEach((n, i) => (W.rangeEdits[n] = ex.design_ranges[i]));
      W.step = 1;
      W.maxStep = 1;
      render();
    } catch (e) {
      W.message = "The worked example could not load. Reload the page and try again.";
      render();
    }
  }

  /* ---------- mock fit / predict / next ---------- */
  function fitNow() {
    // MOCK: replace with POST /flow/fit
    W.model = F.fit(W.ds.X, W.ds.y, W.ds.ranges);
    W.xIndex = 0;
    W.fixed = F.midpoints(W.ds.ranges);
    W.next = null;
    const r = { runs: W.ds.n, loo: W.model.looRelL2 };
    if (W.rounds.length && W.rounds[W.rounds.length - 1].runs === r.runs) W.rounds[W.rounds.length - 1] = r;
    else W.rounds.push(r);
  }
  function predictCurve() {
    // MOCK: replace with POST /flow/predict
    const g = F.sliceGrid(W.ds.ranges, W.xIndex, W.fixed, 70);
    return { xs: g.xs, ...F.predict(W.model, g.points) };
  }
  function nextRuns() {
    // MOCK: replace with POST /flow/next
    if (!W.next || W.next.k !== W.nextK) W.next = { k: W.nextK, result: F.suggestNext(W.model, W.nextK) };
    return W.next.result;
  }

  /* ---------- shared pieces ---------- */
  const panelWidth = () => Math.max(300, (($("wb-panel") && $("wb-panel").clientWidth) || 760) - 40);
  // Mirrors the .wb-split container query in flow.css: beside a 300px control column when the panel is wide.
  const chartWidth = () => (W.step === 3 && panelWidth() >= 860 ? panelWidth() - 300 - 28 : panelWidth());
  function chartHtml(showNext) {
    const c = predictCurve();
    const names = W.ds.paramNames;
    const nx = showNext ? nextRuns() : null;
    const obsTip = (x, i) =>
      "Run " + (i + 1) + ": " + names.map((n, j) => n + " " + F.fmt(x[j])).join(", ") + " gives " + F.fmt(W.ds.y[i]);
    const svg = F.outcomeChartSvg({
      width: chartWidth(),
      xs: c.xs,
      mean: c.mean,
      std: c.std,
      observed: W.ds.X.map((x, i) => ({ x: x[W.xIndex], y: W.ds.y[i], tip: obsTip(x, i) })),
      next: nx
        ? nx.points.map((p, i) => ({
            x: p[W.xIndex],
            tip: "Suggested run N" + (i + 1) + ": " + names.map((n, j) => n + " " + F.fmt(p[j])).join(", ") + ". Predicted " + F.fmt(nx.mean[i]) + " (spread " + F.fmt(2 * nx.std[i]) + ").",
          }))
        : [],
      xLabel: F.axisName(names[W.xIndex]),
      yLabel: F.axisName(W.ds.outcomeName),
      watermark: "MOCK DATA",
      aria: "Mock predicted " + W.ds.outcomeName + " against " + names[W.xIndex] + " with an approximate uncertainty band and your " + W.ds.n + " runs.",
    });
    const legend = F.legendHtml(
      [
        { name: "Predicted outcome (stand-in model)", cls: "s-pre" },
        { name: "Approximate ±2σ band", cls: "s-pre band-sw" },
        { name: "Your runs", cls: "obs-sw" },
      ].concat(nx ? [{ name: "Suggested next runs", cls: "next-sw" }] : []),
    );
    const worst = c.std.indexOf(Math.max(...c.std));
    const note =
      names.length > 1
        ? "Showing " + esc(names[W.xIndex]) + ", with " + names.filter((_, j) => j !== W.xIndex).map((n) => esc(n) + " held at " + F.fmt(W.fixed[names.indexOf(n)])).join(" and ") +
          ". Your runs are projected onto this axis, so they only sit on the curve if the held values match theirs."
        : "";
    return {
      html: svg + legend,
      note: note,
      least: "Least certain near " + names[W.xIndex] + " = " + F.fmt(c.xs[worst]) + ".",
    };
  }
  const BAND_NOTE =
    "The band is an approximate ±2σ from a stand-in model, widened to match leave-one-out error. Its coverage is not validated.";

  /* ---------- step views ---------- */
  function viewDescribe() {
    return (
      '<div class="card"><div class="card-head"><h2 tabindex="-1">1. Describe your study</h2></div><div class="pad">' +
      (W.message ? '<ul class="issues"><li class="error"><span class="lvl">ERROR</span><span>' + esc(W.message) + "</span></li></ul>" : "") +
      '<div class="form-grid">' +
      '<label class="field">Study name<input id="wb-name" type="text" maxlength="80" value="' + esc(W.spec.name) + '" /><small>Only used to title your report.</small></label>' +
      '<label class="field">Runs a blind sweep would need<input id="wb-blind" type="number" min="10" max="100000" step="1" value="' + W.spec.blindSweep + '" /><small>How many solver runs you would otherwise plan. Used for the runs-saved figure.</small></label>' +
      '<label class="field">Error target (relative L2, %)<input id="wb-target" type="number" min="1" max="50" step="1" value="' + Math.round(W.spec.target * 100) + '" /><small>The Model demo uses a self-imposed 15%. Set your own tolerance here.</small></label>' +
      '</div><div class="actions"><button class="btn primary" id="wb-go-upload" type="button">Continue to upload</button>' +
      '<button class="btn" id="wb-load-example" type="button">Load the worked example ' + tag + "</button></div></div></div>"
    );
  }

  function viewUpload() {
    return (
      '<div class="wb-grid"><div class="card"><div class="card-head"><h2 tabindex="-1">2. Upload your runs</h2><span class="tag">CSV</span></div><div class="pad">' +
      '<div class="dropzone" id="wb-drop"><strong>Drop a CSV here</strong><span>or</span>' +
      '<input id="wb-file" class="visually-hidden" type="file" accept=".csv,text/csv,text/plain" />' +
      '<label class="btn small" for="wb-file">Choose a file</label>' +
      (F.FLOW_MODE === "mock" ? '<span class="flow-note">The file is read in this browser. It is not uploaded anywhere.</span>' : "") +
      "</div>" +
      '<details style="margin-top:12px"><summary>Or paste CSV text</summary><div class="field" style="margin-top:10px"><textarea id="wb-paste" class="paste" rows="6" spellcheck="false" aria-label="Paste CSV text"></textarea></div>' +
      '<button class="btn small" id="wb-paste-go" type="button" style="margin-top:8px">Use pasted text</button></details>' +
      '<div id="wb-result"></div></div></div>' +
      '<aside class="card pad side-card"><h3>File format</h3><p>One row per run. A header row, then numbers.</p>' +
      '<pre class="code-sample">run_id,velocity,radius,pressure_drop\n1,2.25,5.33,75.27\n2,1.38,7.67,27.33</pre>' +
      "<ul><li>1 to 3 design parameters, one outcome.</li><li>3 to 200 runs.</li><li>A column named run_id, id or case is treated as a label.</li><li>No empty cells.</li></ul>" +
      '<p><button class="btn small" id="wb-template" type="button">Download a template</button></p></aside></div>'
    );
  }

  function resultHtml() {
    const p = W.parsed;
    if (!p) return "";
    if (!p.columns.length) return '<div id="wb-issues"></div>';
    const rows = p.columns
      .map((c) => {
        if (c.kind === "id") return "<tr><td>" + esc(c.name) + '</td><td>Run label (not used)</td><td class="num"></td><td class="num"></td><td class="num"></td><td class="num"></td></tr>';
        if (c.kind !== "numeric") return "<tr><td>" + esc(c.name) + '</td><td>Not numeric</td><td class="num"></td><td class="num"></td><td class="num"></td><td class="num"></td></tr>';
        const role = W.roles[c.name];
        const e = W.rangeEdits[c.name];
        const lo = e ? e[0] : c.min;
        const hi = e ? e[1] : c.max;
        const isP = role === "param";
        return (
          "<tr><td>" + esc(c.name) + '</td><td><select data-role="' + esc(c.name) + '" aria-label="Role of ' + esc(c.name) + '">' +
          [["param", "Design parameter"], ["outcome", "Outcome"], ["ignore", "Ignore"]].map(([v, l]) => '<option value="' + v + '"' + (role === v ? " selected" : "") + ">" + l + "</option>").join("") +
          '</select></td><td class="num">' + F.fmt(c.min) + '</td><td class="num">' + F.fmt(c.max) + '</td><td class="num">' +
          (isP ? '<input type="number" step="any" data-lo="' + esc(c.name) + '" value="' + lo + '" aria-label="Design range minimum of ' + esc(c.name) + '" />' : "") +
          '</td><td class="num">' +
          (isP ? '<input type="number" step="any" data-hi="' + esc(c.name) + '" value="' + hi + '" aria-label="Design range maximum of ' + esc(c.name) + '" />' : "") +
          "</td></tr>"
        );
      })
      .join("");
    return (
      '<p class="flow-note"><strong>' + esc(W.fileName) + "</strong>: " + p.n + " runs, " + p.columns.length + " columns.</p>" +
      '<div class="table-scroll"><table class="flow-table"><thead><tr><th>Column</th><th>Role</th><th class="num">Data min</th><th class="num">Data max</th><th class="num">Design min</th><th class="num">Design max</th></tr></thead><tbody>' +
      rows + "</tbody></table></div>" +
      '<p class="flow-note">The design range is the space you want predictions over. It must include all your runs. Widening it shows where the model has nothing to go on.</p>' +
      '<div id="wb-issues"></div><div class="actions" id="wb-fit-actions"></div>'
    );
  }
  function refreshValidation() {
    if (!W.parsed || !$("wb-issues")) return;
    const issues = validate();
    const ok = !!W.ds && !F.hasError(issues);
    $("wb-issues").innerHTML =
      issuesHtml(issues) || '<ul class="issues"><li class="ok"><span class="lvl">OK</span><span>The file passes every check.</span></li></ul>';
    const actions = $("wb-fit-actions");
    if (actions)
      actions.innerHTML =
        '<button class="btn primary" id="wb-fit" type="button"' + (ok ? "" : " disabled") + ">Fit " + (F.FLOW_MODE === "mock" ? "(mock)" : "") + "</button>";
  }

  function viewFit() {
    const m = W.model;
    const d = W.ds;
    return (
      '<div class="card"><div class="card-head"><h2 tabindex="-1">3. Fit ' + tag + '</h2></div><div class="pad">' +
      '<p>The stand-in model has been fitted to your <strong>' + d.n + " runs</strong> (" + esc(d.paramNames.join(", ")) + " to " + esc(d.outcomeName) + ").</p>" +
      '<div class="stat-row">' +
      [
        [String(d.n), "runs used"],
        [F.pct(m.looRelL2), "leave-one-out error on your runs"],
        [F.pct(W.spec.target), "your error target"],
      ].map(([v, l]) => '<div class="stat"><strong>' + esc(v) + tag + "</strong><span>" + l + "</span></div>").join("") +
      "</div>" +
      '<p class="flow-note">Leave-one-out error: each run is held out in turn and predicted from the others, then compared with its true value. In the live flow this step adapts the pretrained fluid model to your runs. Here it is a plain Gaussian process, so the number says nothing about Delta-V.</p>' +
      '<div class="actions"><button class="btn primary" id="wb-go-predict" type="button">See predictions</button></div></div></div>'
    );
  }

  function viewPredict() {
    const d = W.ds;
    const ch = chartHtml(false);
    const others = d.paramNames.map((n, j) => ({ n, j })).filter((o) => o.j !== W.xIndex);
    return (
      '<div class="card"><div class="card-head"><h2 tabindex="-1">4. Predict across your design range ' + tag + "</h2></div><div class=\"pad\">" +
      (d.paramNames.length > 1
        ? '<div class="form-grid single"><label class="field">Parameter on the horizontal axis<select id="wb-xsel">' +
          d.paramNames.map((n, j) => '<option value="' + j + '"' + (j === W.xIndex ? " selected" : "") + ">" + esc(n) + "</option>").join("") +
          "</select></label></div>"
        : "") +
      '<div class="wb-split"><div class="wb-split-main"><div id="wb-chart">' + ch.html + "</div>" +
      '<p class="flow-note" id="wb-chart-note">' + ch.note + " " + ch.least + " " + BAND_NOTE + "</p></div>" +
      '<div class="wb-split-side">' +
      (others.length
        ? '<div class="sliders" id="wb-sliders">' +
          others
            .map(
              (o) =>
                '<label class="field">' + esc(o.n) + ': <output id="wb-out-' + o.j + '">' + F.fmt(W.fixed[o.j]) + '</output><input type="range" data-fix="' + o.j + '" min="' + d.ranges[o.j][0] + '" max="' + d.ranges[o.j][1] +
                '" step="' + (d.ranges[o.j][1] - d.ranges[o.j][0]) / 200 + '" value="' + W.fixed[o.j] + '" /></label>',
            )
            .join("") + "</div>"
        : "") +
      "<h3>Predict at one design point</h3>" +
      '<div class="form-grid stacked">' +
      d.paramNames.map((n, j) => '<label class="field">' + esc(n) + '<input type="number" step="any" data-pt="' + j + '" value="' + F.fmt(F.midpoints(d.ranges)[j], 4) + '" /></label>').join("") +
      '</div><p id="wb-point-out" aria-live="polite" class="point-out"></p></div></div>' +
      '<div class="actions"><button class="btn primary" id="wb-go-next" type="button">Suggest the next runs</button></div></div></div>'
    );
  }
  function updateChart() {
    const ch = chartHtml(false);
    $("wb-chart").innerHTML = ch.html;
    F.attachTips($("wb-chart"));
    $("wb-chart-note").innerHTML = ch.note + " " + ch.least + " " + BAND_NOTE;
  }
  function updatePoint() {
    const vals = [...document.querySelectorAll("[data-pt]")].map((i) => Number(i.value));
    const out = $("wb-point-out");
    const bad = vals.findIndex((v, j) => !Number.isFinite(v) || v < W.ds.ranges[j][0] || v > W.ds.ranges[j][1]);
    if (bad >= 0) {
      out.textContent = "Enter " + W.ds.paramNames[bad] + " between " + F.fmt(W.ds.ranges[bad][0]) + " and " + F.fmt(W.ds.ranges[bad][1]) + " (your design range).";
      return;
    }
    const r = F.predict(W.model, [vals]);
    out.innerHTML = "<strong>" + F.fmt(r.mean[0]) + " ± " + F.fmt(2 * r.std[0]) + "</strong>Predicted " + esc(W.ds.outcomeName) + ", with an approximate 2σ spread from the mock stand-in.";
  }

  function viewNext() {
    const d = W.ds;
    const nx = nextRuns();
    const ch = chartHtml(true);
    const canMock = W.source === "example" && W.truth;
    const rows = nx.points
      .map(
        (p, i) =>
          "<tr><td>N" + (i + 1) + "</td>" + p.map((v) => '<td class="num">' + F.fmt(v, 4) + "</td>").join("") +
          '<td class="num">' + F.fmt(nx.mean[i]) + '</td><td class="num">±' + F.fmt(2 * nx.std[i]) + "</td></tr>",
      )
      .join("");
    return (
      '<div class="card"><div class="card-head"><h2 tabindex="-1">5. Run these next ' + tag + "</h2></div><div class=\"pad\">" +
      '<p>Simulate these design points in your own solver. They are where the model is least sure, one after another, assuming each earlier suggestion has been run.</p>' +
      '<div class="form-grid single"><label class="field">How many suggestions<select id="wb-k">' +
      [2, 3, 4, 5, 6, 8].map((k) => '<option value="' + k + '"' + (k === W.nextK ? " selected" : "") + ">" + k + "</option>").join("") +
      "</select></label></div>" +
      '<div class="table-scroll"><table class="flow-table"><thead><tr><th>Run</th>' +
      d.paramNames.map((n) => '<th class="num">' + esc(n) + "</th>").join("") +
      '<th class="num">Predicted</th><th class="num">Spread</th></tr></thead><tbody>' + rows + "</tbody></table></div>" +
      '<div id="wb-chart" style="margin-top:16px">' + ch.html + "</div>" +
      '<p class="flow-note" id="wb-chart-note">' + ch.note + " " + ch.least + " " + BAND_NOTE + "</p>" +
      '<div class="actions"><button class="btn" id="wb-dl-next" type="button">Download these as CSV</button>' +
      (canMock ? '<button class="btn" id="wb-mock-solve" type="button">Run them in the mock solver ' + tag + "</button>" : "") +
      '<button class="btn" id="wb-up2" type="button">Upload updated runs</button><input id="wb-file2" type="file" accept=".csv,text/csv,text/plain" hidden aria-label="Choose the updated CSV" />' +
      '<button class="btn primary" id="wb-go-report" type="button">See the report</button></div>' +
      '<p class="flow-note">' +
      (canMock
        ? "The mock solver evaluates the synthetic pipe-bend formula at these points, adds them to your runs and refits. It only exists for the worked example. "
        : "") +
      "After you run these, add the results to your original file and upload it again. The model refits and the error is checked again.</p></div></div>"
    );
  }

  function viewReport() {
    const d = W.ds;
    const m = W.model;
    const saved = Math.max(0, W.spec.blindSweep - d.n);
    const meets = m.looRelL2 != null && m.looRelL2 <= W.spec.target;
    return (
      '<div class="card"><div class="card-head"><h2 tabindex="-1">6. Report: ' + esc(W.spec.name) + " " + tag + "</h2></div><div class=\"pad\">" +
      '<div class="stat-row">' +
      [
        [d.n + " of " + W.spec.blindSweep, "runs used, against your blind-sweep plan"],
        [String(saved), "runs not yet needed (not a measured saving)"],
        [F.pct(m.looRelL2), "leave-one-out error, stand-in model"],
        [F.pct(W.spec.target), "your error target"],
      ].map(([v, l]) => '<div class="stat"><strong>' + esc(v) + tag + "</strong><span>" + l + "</span></div>").join("") +
      "</div>" +
      '<p><span class="chip ' + (meets ? "good" : "warn") + '">' + (meets ? "Stand-in error is under the target" : "Stand-in error is not yet under the target") + "</span></p>" +
      (W.rounds.length > 1
        ? "<h3>Error by round</h3>" +
          F.seriesChartSvg({
            width: panelWidth(),
            series: [{ name: "Leave-one-out error", cls: "s-pre", points: W.rounds.map((r) => ({ x: r.runs, y: r.loo, tip: r.runs + " runs: " + F.pct(r.loo) })) }],
            xLog: false,
            xTicks: W.rounds.map((r) => r.runs),
            xLabel: "Runs used",
            yLabel: "Leave-one-out error",
            yFmt: (v) => F.pct(v),
            hLine: { y: W.spec.target, label: F.pct(W.spec.target) + " target" },
            watermark: "MOCK DATA",
            aria: "Mock leave-one-out error after each round of runs, against the error target.",
          })
        : "") +
      "<h3>What this report does not say</h3><ul class=\"limits-list\">" +
      "<li>The \"runs not yet needed\" figure is your blind-sweep plan minus the runs you have used. It is not a measured saving. The measured comparison is on the Evidence page.</li>" +
      "<li>Nothing here comes from the Delta-V model. It is a Gaussian process stand-in.</li>" +
      "<li>Predictions narrow a design space. They do not replace the final verification run in your own solver.</li>" +
      "<li>2D fluid problems only, with data resampled to a fixed grid in the live flow.</li></ul>" +
      '<div class="actions"><button class="btn" id="wb-dl-report" type="button">Download report (JSON)</button>' +
      '<button class="btn" id="wb-back-next" type="button">Back to next runs</button>' +
      '<button class="btn" id="wb-restart" type="button">Start over</button></div></div></div>'
    );
  }

  /* ---------- render + events ---------- */
  function renderSteps() {
    $("wb-steps").innerHTML = STEPS.map(
      (s, i) =>
        '<li><button type="button" data-step="' + i + '"' + (i > W.maxStep ? " disabled" : "") + (i === W.step ? ' aria-current="step"' : "") +
        (i <= W.maxStep && i !== W.step ? ' class="done"' : "") + '><span class="n">' + (i + 1) + "</span>" + s + "</button></li>",
    ).join("");
  }
  function go(step, focus) {
    W.step = step;
    W.maxStep = Math.max(W.maxStep, step);
    W.message = "";
    render(focus !== false);
  }
  function render(focus) {
    if (!ready) return;
    renderSteps();
    const views = [viewDescribe, viewUpload, viewFit, viewPredict, viewNext, viewReport];
    $("wb-panel").innerHTML = views[W.step]();
    if (W.step === 1) {
      $("wb-result").innerHTML = resultHtml();
      refreshValidation();
    }
    if (W.step === 3 || W.step === 4) F.attachTips($("wb-chart"));
    bind();
    if (focus) {
      const h = $("wb-panel").querySelector("h2");
      if (h) h.focus({ preventScroll: false });
    }
  }
  async function readFile(file) {
    if (!file) return;
    if (file.size > MAX_FILE_BYTES) {
      W.parsed = { header: [], columns: [], n: 0, issues: [{ level: "error", msg: "That file is larger than 2 MB. A file of runs should be much smaller." }] };
      $("wb-result").innerHTML = resultHtml();
      refreshValidation();
      return;
    }
    W.source = "upload";
    W.truth = null;
    ingest(await file.text(), file.name);
    W.rounds = [];
    W.maxStep = 1;
    renderSteps();
    $("wb-result").innerHTML = resultHtml();
    refreshValidation();
    bindResult();
  }
  async function readUpdated(file) {
    if (!file) return;
    if (file.size > MAX_FILE_BYTES) return;
    ingest(await file.text(), file.name, true);
    const issues = validate();
    if (!W.ds || F.hasError(issues)) {
      W.step = 1;
      W.maxStep = 1;
      render(true);
      return;
    }
    fitNow();
    go(3);
  }
  function bindResult() {
    const root = $("wb-result");
    if (!root) return;
    root.querySelectorAll("[data-role]").forEach((s) => {
      s.onchange = () => {
        const name = s.dataset.role;
        W.roles[name] = s.value;
        if (s.value === "outcome")
          Object.keys(W.roles).forEach((n) => {
            if (n !== name && W.roles[n] === "outcome") W.roles[n] = "ignore";
          });
        delete W.rangeEdits[name];
        root.innerHTML = resultHtml();
        refreshValidation();
        bindResult();
        const again = root.querySelector('[data-role="' + CSS.escape(name) + '"]');
        if (again) again.focus();
      };
    });
    const edit = (attr, idx) => (i) => {
      i.onchange = () => {
        const name = i.dataset[attr];
        const c = W.parsed.columns.find((x) => x.name === name);
        const cur = W.rangeEdits[name] || [c.min, c.max];
        cur[idx] = i.value === "" ? NaN : Number(i.value);
        W.rangeEdits[name] = cur;
        refreshValidation();
        bindResult();
      };
    };
    root.querySelectorAll("[data-lo]").forEach(edit("lo", 0));
    root.querySelectorAll("[data-hi]").forEach(edit("hi", 1));
    const fit = $("wb-fit");
    if (fit)
      fit.onclick = () => {
        try {
          fitNow();
          go(2);
        } catch (e) {
          $("wb-issues").innerHTML = issuesHtml([{ level: "error", msg: e.message }]);
        }
      };
  }
  function bind() {
    const on = (id, fn) => {
      const n = $(id);
      if (n) n.onclick = fn;
    };
    if (W.step === 0) {
      $("wb-name").oninput = (e) => (W.spec.name = e.target.value);
      $("wb-blind").oninput = (e) => (W.spec.blindSweep = Math.max(10, Math.round(Number(e.target.value)) || 100));
      $("wb-target").oninput = (e) => (W.spec.target = Math.min(0.5, Math.max(0.01, Number(e.target.value) / 100 || 0.15)));
      on("wb-go-upload", () => go(1));
      on("wb-load-example", loadExample);
    }
    if (W.step === 1) {
      const drop = $("wb-drop");
      drop.ondragover = (e) => {
        e.preventDefault();
        drop.classList.add("over");
      };
      drop.ondragleave = () => drop.classList.remove("over");
      drop.ondrop = (e) => {
        e.preventDefault();
        drop.classList.remove("over");
        readFile(e.dataTransfer.files[0]);
      };
      $("wb-file").onchange = (e) => readFile(e.target.files[0]);
      on("wb-paste-go", () => {
        W.source = "upload";
        W.truth = null;
        ingest($("wb-paste").value, "pasted.csv");
        W.rounds = [];
        W.maxStep = 1;
        $("wb-result").innerHTML = resultHtml();
        refreshValidation();
        bindResult();
      });
      on("wb-template", () =>
        download("delta-v-runs-template.csv", F.toCsv(["run_id", "design_parameter_1", "design_parameter_2", "outcome"], [[1, "", "", ""], [2, "", "", ""], [3, "", "", ""]]), "text/csv"),
      );
      bindResult();
    }
    if (W.step === 2) on("wb-go-predict", () => go(3));
    if (W.step === 3) {
      const xs = $("wb-xsel");
      if (xs)
        xs.onchange = () => {
          W.xIndex = Number(xs.value);
          render(false);
        };
      document.querySelectorAll("[data-fix]").forEach((r) => {
        r.oninput = () => {
          const j = Number(r.dataset.fix);
          W.fixed[j] = Number(r.value);
          $("wb-out-" + j).textContent = F.fmt(W.fixed[j]);
          updateChart();
        };
      });
      document.querySelectorAll("[data-pt]").forEach((i) => (i.oninput = updatePoint));
      updatePoint();
      on("wb-go-next", () => go(4));
    }
    if (W.step === 4) {
      $("wb-k").onchange = (e) => {
        W.nextK = Number(e.target.value);
        render(false);
      };
      on("wb-dl-next", () => {
        const nx = nextRuns();
        download("delta-v-next-runs.csv", F.toCsv(W.ds.paramNames.concat([W.ds.outcomeName]), nx.points.map((p) => p.map((v) => Number(v.toPrecision(5))).concat([""]))), "text/csv");
      });
      on("wb-mock-solve", appendMock);
      on("wb-up2", () => $("wb-file2").click());
      $("wb-file2").onchange = (e) => readUpdated(e.target.files[0]);
      on("wb-go-report", () => go(5));
    }
    if (W.step === 5) {
      on("wb-dl-report", downloadReport);
      on("wb-back-next", () => go(4));
      on("wb-restart", () => {
        W = fresh();
        render(true);
      });
    }
  }

  function appendMock() {
    const nx = nextRuns();
    const p = W.parsed;
    const ids = p.columns.filter((c) => c.kind === "id");
    let nextId = Math.max(0, ...ids.flatMap((c) => c.cells.map(Number).filter(Number.isFinite))) + 1;
    const rows = p.columns[0].cells.map((_, i) => p.columns.map((c) => c.cells[i]));
    nx.points.forEach((pt) => {
      const vals = pt.map((v, j) => Math.min(W.ds.ranges[j][1], Math.max(W.ds.ranges[j][0], Math.round(v * 100) / 100)));
      const y = W.truth(vals);
      rows.push(
        p.columns.map((c) => {
          if (c.kind === "id") return String(nextId++);
          const pi = W.ds.paramNames.indexOf(c.name);
          if (pi >= 0) return String(vals[pi]);
          if (c.name === W.ds.outcomeName) return String(y);
          return "";
        }),
      );
    });
    ingest(F.toCsv(p.header, rows), "worked-example.csv (mock, extended)", true);
    const issues = validate();
    if (!W.ds || F.hasError(issues)) {
      go(1);
      return;
    }
    fitNow();
    go(3);
  }
  function downloadReport() {
    const d = W.ds;
    const m = W.model;
    const nx = W.next ? W.next.result : null;
    const report = {
      generated: new Date().toISOString(),
      mock: true,
      notice: "Mock output from a Gaussian process stand-in. Not produced by the Delta-V model. Not a measured saving.",
      study: { name: W.spec.name, blind_sweep_runs: W.spec.blindSweep, target_rel_l2: W.spec.target },
      columns: { design_parameters: d.paramNames, outcome: d.outcomeName },
      design_ranges: d.ranges,
      runs_used: d.n,
      loo_rel_l2: m.looRelL2,
      stand_in_error_under_target: m.looRelL2 != null && m.looRelL2 <= W.spec.target,
      rounds: W.rounds,
      suggested_next_runs: nx ? nx.points.map((pt, i) => ({ point: pt, predicted: nx.mean[i], approx_2sigma: 2 * nx.std[i] })) : [],
      limits: ["2D fluid problems only", "Narrows a design space; does not replace a final verification run", "Band coverage is not validated"],
    };
    download("delta-v-report-mock.json", JSON.stringify(report, null, 2) + "\n", "application/json");
  }

  function show() {
    if (!ready) {
      $("wb-steps").addEventListener("click", (e) => {
        const b = e.target.closest("button[data-step]");
        if (b && !b.disabled) go(Number(b.dataset.step));
      });
      ready = true;
    }
    let want = false;
    try {
      want = sessionStorage.getItem("deltav-load-example") === "1";
      sessionStorage.removeItem("deltav-load-example");
    } catch (e) {
      want = false;
    }
    if (want) loadExample();
    else render(false);
  }

  window.DeltaWorkbench = { show };
})();
