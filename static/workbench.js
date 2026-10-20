/* Workbench: adapt the pretrained FNO to a compatible study through the /flow API.
 *
 * Six steps: Describe, Upload, Adapt, Explore, Next runs, Report. Every number on
 * these screens comes from the server (validation, worker jobs, stored models and
 * the cost ledger). There is no browser model and no fallback: failures are shown.
 */
(function () {
  "use strict";
  const F = window.DeltaFlow;
  const A = window.DeltaFlowApi;
  const $ = (id) => document.getElementById(id);
  const esc = F.esc;
  const STEPS = ["Your study", "Predict", "Next runs", "Summary"];
  const CURVE_POINTS = 21;
  const client = A.createClient();

  const W = {
    step: 0,
    cap: null,
    view: null,
    runs: null,
    report: null,
    message: null,
    notice: null,
    issues: [],
    busy: false,
    outcome: 0,
    xIndex: 0,
    fixed: null,
    curve: null,
    point: null,
    next: null,
    nextOpts: { policy: null, batch: 8 },
    poll: null,
    ready: false,
    showProjected: false,
    showNext: true,
    uploading: null,
  };

  /* ---------- helpers ---------- */
  function errorHtml(msg, issues) {
    const list = (issues || []).filter((i) => i && i.message);
    return (
      '<ul class="issues" role="alert"><li class="error"><span class="lvl">Error</span><span>' + esc(msg) + "</span></li>" +
      list.map((i) => '<li class="' + (i.level === "error" ? "error" : "warn") + '"><span class="lvl">' + (i.level === "error" ? "Error" : "Note") +
        "</span><span>" + esc(i.message) + (i.fix ? ' <em class="fix">' + esc(i.fix) + "</em>" : "") + "</span></li>").join("") +
      "</ul>"
    );
  }
  const noteHtml = (msg) => '<ul class="issues"><li class="warn"><span class="lvl">Note</span><span>' + esc(msg) + "</span></li></ul>";
  const bullets = (items) => '<ul class="plain-list">' + items.map((i) => "<li>" + i + "</li>").join("") + "</ul>";
  const okHtml = (msg) => '<ul class="issues"><li class="ok"><span class="lvl">Done</span><span>' + esc(msg) + "</span></li></ul>";
  const dataTag = (kind) => '<span class="data-tag ' + esc(kind || "") + '">' + esc(A.DATA_LABEL[kind] || kind || "") + "</span>";
  // Studies opened from the worked example get a plain title, whatever the bundle calls them.
  const displayName = (st) => (st && st.example_id ? "Worked example" : (st && st.name) || "Untitled study");
  const EXAMPLE_ABOUT = "A generated 2D periodic flow study from our research, with recorded solver results and a model adapted in advance.";
  // Runs are referred to by role and position; exact run ids stay in tables and downloads.
  const runLabel = (r) => {
    const same = W.runs.runs.filter((x) => x.role === r.role);
    return ({ train: "Training run", development: "Development run", calibration: "Calibration run", verification: "Verification run" }[r.role] || "Run") + " " + (same.indexOf(r) + 1);
  };
  const ctlLabel = (n) => F.axisName(n).replace(/^./, (c) => c.toUpperCase());
  const lcFirst = (t) => t.charAt(0).toLowerCase() + t.slice(1);
  const short = (sha) => (sha ? String(sha).slice(0, 12) + "…" : "n/a");
  const latestVersion = () => (W.view ? W.view.versions[W.view.versions.length - 1] : null);
  const readyModels = () => (W.view ? W.view.models.filter((m) => m.status === "ready") : []);
  const model = () => readyModels().slice(-1)[0] || null;
  const fitJobs = () => (W.view ? W.view.jobs.filter((j) => j.kind === "fit") : []);
  const activeFit = () => fitJobs().filter((j) => j.status === "queued" || j.status === "running").slice(-1)[0] || null;
  const lastFit = () => fitJobs().slice(-1)[0] || null;

  function download(name, blob) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function downloadArtifact(name) {
    const res = await client.artifact(W.view.study_id, name);
    download(name, await res.blob());
  }

  function maxStep() {
    return W.view && model() ? 3 : 0;
  }
  function setStep(step, focus) {
    W.step = Math.min(step, maxStep());
    W.message = null;
    W.issues = [];
    render(focus !== false);
  }
  async function guard(fn) {
    if (W.busy) return;
    W.busy = true;
    W.message = null;
    W.issues = [];
    renderBusy();
    try {
      await fn();
    } catch (e) {
      W.message = e.message || String(e);
      W.issues = e.issues || [];
    } finally {
      W.busy = false;
      $("wb-panel").removeAttribute("aria-busy");
      render(false);
    }
  }
  // While a request runs, controls are disabled and the panel says what is happening.
  function renderBusy() {
    const panel = $("wb-panel");
    panel.setAttribute("aria-busy", "true");
    document.querySelectorAll("#wb-panel button, #wb-panel input, #wb-panel select").forEach((b) => (b.disabled = true));
    const drop = $("wb-drop");
    if (drop && W.uploading) drop.querySelector("strong").textContent = "Uploading and checking " + W.uploading + "…";
  }

  /* ---------- server state ---------- */
  async function loadStudy(id) {
    const oldStudy = W.view?.study_id;
    W.view = await client.study(id);
    if (oldStudy !== W.view.study_id) {
      W.point = null;
      W.next = null;
      W.report = null;
      W.artifacts = null;
      W.curve = null;
      W.pointRequest = (W.pointRequest || 0) + 1;
    }
    client.setStudyId(W.view.study_id);
    W.runs = await client.runs(W.view.study_id);
    const m = model();
    if (m && (!W.fixed || W.fixedModel !== m.model_id)) {
      W.fixed = F.midpoints(m.ranges);
      W.fixedModel = m.model_id;
      W.curve = null;
    }
    if (activeFit()) startPolling();
  }
  async function refreshStudy() {
    if (!W.view) return;
    await loadStudy(W.view.study_id);
  }
  function stopPolling() {
    if (W.poll) clearTimeout(W.poll);
    W.poll = null;
  }
  function startPolling() {
    stopPolling();
    const tick = async () => {
      const job = activeFit();
      if (!job || !document.body || document.body.dataset.page !== "workbench") {
        W.poll = null;
        return;
      }
      try {
        const fresh = await client.job(job.job_id);
        const i = W.view.jobs.findIndex((j) => j.job_id === fresh.job_id);
        W.view.jobs[i] = fresh;
        if (fresh.status === "queued" || fresh.status === "running") {
          if (W.step === 0) updateJobPanel();
          W.poll = setTimeout(tick, 2000);
          return;
        }
        W.poll = null;
        await refreshStudy();
        render(false);
      } catch (e) {
        W.message = "Lost contact with the server while following the job: " + e.message + " Reload to try again.";
        W.poll = null;
        render(false);
      }
    };
    W.poll = setTimeout(tick, 1500);
  }

  /* ---------- shared view pieces ---------- */
  // Each step is a page section with one heading, a short lede and plain dividers between parts.
  const head = (title, lede) => '<header class="wb-head"><h2 tabindex="-1">' + esc(title) + "</h2>" + (lede ? '<p class="wb-lede">' + lede + "</p>" : "") + "</header>";
  const facts = (rows) => '<dl class="facts">' + rows.filter(Boolean).map(([k, v]) => "<div><dt>" + esc(k) + "</dt><dd>" + v + "</dd></div>").join("") + "</dl>";
  const roleLine = (c) => c.train + " training · " + c.development + " development · " + c.calibration + " calibration" + (c.verification ? " · " + c.verification + " verification" : "");
  function availabilityLine() {
    const c = W.cap;
    return c.adapt.enabled
      ? "Adapting a new study runs on a GPU worker" + (c.worker.available ? " (connected)." : ". No worker is connected right now, so a job would wait in the queue.")
      : "Adapting a new study needs a GPU, which is off in this public demo. Uploads are still checked, and the worked example has its model ready.";
  }
  // Said the same way everywhere a newer study version exists than the model was fitted on.
  const modelVersionNote = (m) => {
    const mv = W.view.versions.find((x) => x.version_id === m.version_id);
    return "New runs added (version " + latestVersion().number + "). The model still uses version " + (mv ? mv.number : 1) + "; re-adapt on a GPU to include them.";
  };
  // What the model actually learned from, and how many runs arrived after it was fitted.
  function modelRuns(m, v) {
    const mv = W.view.versions.find((x) => x.version_id === m.version_id) || v;
    const used = m.role_counts_used || mv.role_counts;
    return { used, version: mv.number, added: Math.max(0, v.role_counts.train - mv.role_counts.train) };
  }
  const GPU_NOTE = "On a GPU, adapting the pretrained model to a study like this takes about 3 to 4 minutes; then you can predict and plan next runs. This public demo has no GPU attached, so your upload is checked and stored but not adapted here.";
  const NEW_PROBLEM = "A new problem here means a new set of runs in the supported family: a 2D periodic flow on a 64 × 64 grid, set by viscosity, forcing and initial mix, measured by kinetic energy and pressure RMS. Your own settings, ranges and runs are new to the model. Other geometries, 3D flows and other outcomes are not supported yet.";
  const STORAGE_NOTE = "Uploaded studies are checked and stored on this server. Only this browser can open them, because it holds the study access key; clearing site data loses that access. Delete a study from its Report.";

  /* ---------- Describe: start a study ---------- */
  function viewDescribe() {
    const c = W.cap;
    const ex = c.example;
    const ctl = c.controls.map((x) => "<strong>" + esc(F.axisName(x.name)) + "</strong> " + x.range[0] + "–" + x.range[1]).join(", ");
    return (
      head("Start a study", "Try the ready worked example, or upload your own runs if they fit the study type supported today.") +
      '<p class="status-line">' + esc(availabilityLine()) + "</p>" +
      '<div class="entry-grid">' +
      '<section class="entry"><h3>Worked example</h3>' +
      bullets([
        "A generated 2D flow study from our research, with recorded solver results.",
        c.preadapted ? "Its model was adapted in advance, so every step works here." : "Adapt it to unlock the later steps.",
        "Opens your own copy. No computation starts.",
      ]) +
      '<div class="actions"><button class="btn primary" id="wb-example" type="button"' + (ex.available ? "" : " disabled") + ">Explore the worked example</button>" +
      (ex.available ? '<a class="text-link" href="/flow/examples/worked/bundle" download>Download its ZIP</a>' : "") + "</div>" +
      (ex.available ? "" : noteHtml("The worked example is not available on this server.")) +
      "</section>" +
      '<section class="entry"><h3>Your own study</h3>' +
      bullets([
        "A 2D periodic flow on a fixed 64 × 64 grid, set by " + ctl + ".",
        "Each run supplies its starting and final fields.",
        "Predicts two outcomes: <strong>kinetic energy</strong> and <strong>pressure RMS</strong>.",
        "Not supported: other geometries, 3D, or arbitrary CFD output.",
      ]) +
      '<p class="flow-note">' + esc(NEW_PROBLEM) + " " + esc(GPU_NOTE) + "</p>" +
      '<p><a class="text-link" href="/downloads/delta-v-sample-study.zip" download>Download a sample study ZIP</a> <span class="flow-note">8 training runs from the same flow family, ready to upload.</span></p>' +
      dropZone(null) + (W.message ? errorHtml(W.message, W.issues) : "") +
      '<details><summary>What the ZIP holds</summary>' + zipContents() + "</details>" +
      "</section></div>" +
      '<section class="wb-section" id="wb-studies" hidden></section>'
    );
  }
  async function renderStudyList() {
    const box = $("wb-studies");
    if (!box) return;
    try {
      const list = await client.studies();
      if (!list.length) return;
      box.hidden = false;
      box.innerHTML = "<h3>Your studies on this server</h3>" +
        '<div class="table-scroll"><table class="flow-table"><thead><tr><th>Study</th><th>Data</th><th>Created</th><th><span class="visually-hidden">Open</span></th></tr></thead><tbody>' +
        list.map((s) => "<tr" + (W.view && W.view.study_id === s.study_id ? ' aria-current="true"' : "") + "><td>" + esc(displayName(s)) + "</td><td>" + dataTag(s.data_kind) + "</td><td>" +
          esc(new Date(s.created_at * 1000).toLocaleDateString()) + '</td><td class="num"><button class="btn small" data-open="' + esc(s.study_id) + '" type="button">Open</button></td></tr>').join("") +
        "</tbody></table></div>";
      box.querySelectorAll("[data-open]").forEach((b) => (b.onclick = () => guard(async () => {
        await loadStudy(b.dataset.open);
        W.step = model() ? 1 : 0;
      })));
    } catch (e) {
      box.hidden = true;
    }
  }

  /* ---------- Upload ---------- */
  function versionSummary(v) {
    const c = v.role_counts;
    const missing = A.missingRequirements(v, W.cap);
    return (
      facts([
        ["Study", "<strong>" + esc(displayName(W.view)) + "</strong> " + dataTag(v.data_kind)],
        ["Version", String(v.number) + (v.origin ? " (" + esc(v.origin) + ")" : "")],
        ["Runs in this version", roleLine(c)],
        model() ? (() => {
          const r = modelRuns(model(), v);
          return ["Used by the model", r.used.train + " training runs from version " + r.version + (r.added ? ". The " + r.added + " runs added since are saved but not yet used; re-adapt on a GPU to include them." : ".")];
        })() : null,
      ]) +
      '<div class="table-scroll"><table class="flow-table roles"><caption class="visually-hidden">Runs by role</caption><thead><tr><th>Role</th><th class="num">Runs</th><th>Used for</th></tr></thead><tbody>' +
      [["Training", c.train, "fine-tuning the model"], ["Development", c.development, "picking each copy's best checkpoint"],
        ["Calibration", c.calibration, "setting the 90% intervals"], ["Verification", c.verification, "checking the model afterwards; never used to fit it"]]
        .map((r) => "<tr><td>" + r[0] + '</td><td class="num">' + r[1] + "</td><td>" + r[2] + "</td></tr>").join("") +
      "</tbody></table></div>" +
      (missing.length ? missing.map(noteHtml).join("") : okHtml("File checks passed: structure, fields, roles and fingerprints. This is not a check of model accuracy.")) +
      (v.warnings || []).filter((w) => w.code !== "calibration_insufficient").map((w) => noteHtml(w.message)).join("") +
      '<details><summary>Data source and fingerprint</summary><p class="flow-note">' + esc(W.view.example_id ? EXAMPLE_ABOUT : v.provenance.description || "") +
      " Fingerprint <code>" + short(v.content_sha256) + "</code>.</p></details>"
    );
  }
  function dropZone(v) {
    return '<div class="dropzone' + (v ? " compact" : "") + '" id="wb-drop"><strong>' + (v ? "Upload a different study ZIP" : "Drop your study ZIP here") + "</strong>" +
      '<input id="wb-file" class="visually-hidden" type="file" accept=".zip,application/zip" />' +
      '<label class="btn' + (v ? " small" : "") + '" for="wb-file">Choose a file</label>' +
      '<span class="flow-note">ZIP up to 64 MB. ' + (W.cap.example.available ? 'The <a href="/flow/examples/worked/bundle" download>example ZIP</a> shows a valid layout.' : "") + "</span></div>";
  }
  const zipContents = () => bullets(["<code>study.json</code>: what the study is", "<code>runs.csv</code>: one row per run, with its role",
    "<code>fields/</code>: start and end state of each run", "<code>checksums.json</code>: file fingerprints"]) + '<p class="flow-note">' + esc(STORAGE_NOTE) + "</p>";
  async function uploadFile(file) {
    if (!file) return;
    if (file.size > 64 * 1024 * 1024) {
      W.message = "That file is larger than 64 MB.";
      render(false);
      return;
    }
    W.uploading = file.name;
    await guard(async () => {
      try {
        const view = await client.upload(file);
        await loadStudy(view.study_id);
        W.notice = "Checked and stored: " + displayName(view) + ".";
      } finally {
        W.uploading = null;
      }
    });
  }

  /* ---------- Adapt ---------- */
  function jobPanel(job) {
    if (!job) return "";
    const pct = Math.round(A.fitFraction(job) * 100);
    return (
      '<div class="job" id="wb-job" role="status"><p><strong>' + esc(A.jobText(job)) + "</strong>" +
      (job.profile && job.profile !== "task8-fno-lhs-v1" ? ' <span class="mock-tag">Test profile</span>' : "") + "</p>" +
      '<div class="progress" role="progressbar" aria-label="Adaptation progress from reported training steps" aria-valuemin="0" aria-valuemax="100" aria-valuenow="' + pct + '"><div style="width:' + pct + '%"></div></div>' +
      '<p class="flow-note">Progress comes from the training steps the worker reports.</p>' +
      (job.queue_note ? noteHtml(job.queue_note) : "") +
      '<div class="actions"><button class="btn" id="wb-cancel" type="button">Cancel adaptation</button></div></div>'
    );
  }
  function updateJobPanel() {
    const box = $("wb-job");
    const job = activeFit();
    if (box && job) {
      box.outerHTML = jobPanel(job);
      const c = $("wb-cancel");
      if (c) c.onclick = () => guard(async () => {
        await client.cancel(job.job_id);
        await refreshStudy();
      });
    }
  }
  function parityHtml(m, width) {
    const vp = m.verification_predictions;
    const truth = {};
    W.runs.runs.forEach((r) => (truth[r.run_id] = r.outcomes));
    const j = W.outcome;
    const label = A.OUTCOME_LABEL[W.runs.outcomes[j]];
    return (
      F.parityChartSvg({
        width,
        maxSide: 480,
        truth: vp.ids.map((id) => truth[id][j]),
        prediction: vp.mean.map((r) => r[j]),
        lower: vp.lower ? vp.lower.map((r) => r[j]) : null,
        upper: vp.upper ? vp.upper.map((r) => r[j]) : null,
        tolerance: m.tolerance[j],
        xLabel: "Solver reference",
        yLabel: "Prediction",
        aria: label + " predicted against the solver on " + vp.ids.length + " verification runs.",
        tips: vp.ids.map((id, i) => runLabel(W.runs.runs.find((r) => r.run_id === id)) + ": solver " + F.fmt(truth[id][j], 4) + ", predicted " + F.fmt(vp.mean[i][j], 4)),
      }) +
      F.legendHtml([{ name: "Prediction" + (vp.lower ? " with 90% interval" : ""), cls: "s-pre", shape: "circle" }, { name: "Tolerance ±" + m.tolerance[j], cls: "s-gp band-sw" }]) +
      F.dataTable(["Run", "Solver", "Prediction", "Lower", "Upper"], vp.ids.map((id, i) => [runLabel(W.runs.runs.find((r) => r.run_id === id)), truth[id][j], vp.mean[i][j], vp.lower ? vp.lower[i][j] : "n/a", vp.upper ? vp.upper[i][j] : "n/a"]), label + " values as a table")
    );
  }
  function modelPanel(m) {
    const v = latestVersion();
    const ver = W.view.versions.find((x) => x.version_id === m.version_id);
    const test = m.profile !== "task8-fno-lhs-v1";
    const check = m.verification;
    const used = m.role_counts_used || (ver ? ver.role_counts : null);
    const hasParity = check && m.verification_predictions && W.runs && m.verification_predictions.ids.every((id) => W.runs.runs.some((r) => r.run_id === id));
    return (
      (test ? '<div class="mock-banner" role="note"><span class="mock-tag">Test model</span><div>' + esc(m.preadapted ? m.preadapted.label : m.profile_label) + "</div></div>" : "") +
      
      facts([
        ["Status", '<span class="state ok">Ready</span>' + (m.preadapted ? " Adapted in advance on " + esc(m.preadapted.device_name) + "; served on CPU here." : " Adapted on this server.")],
        ["Data", "Study version " + (ver ? ver.number : "?") + (used ? ": " + roleLine({ train: used.train, development: used.development, calibration: used.calibration }) : "")],
        ["Intervals", esc(A.intervalText(m.calibration))],
        check ? ["Verification", "<strong>" + Math.round(check.joint_accuracy_success * check.n) + " of " + check.n + "</strong> held-out runs had both outcomes within tolerance (±" + check.tolerance.join(", ±") + ")." +
          (Number.isFinite(check.joint_coverage) ? " Intervals covered " + F.pct(check.joint_coverage, 1) + " of them." : "")] : null,
        m.readiness ? ["Interval width", esc(m.readiness.criterion_met ? "Intervals are narrower than the tolerance across the study's probe settings." :
          "On " + F.pct(1 - m.readiness.probe_ready_fraction, 0) + " of " + m.readiness.probes + " probe settings the interval is still wider than the tolerance.")] : null,
      ]) +
      (hasParity ?
        '<section class="wb-section"><div class="section-bar"><h3>Prediction vs. solver on held-out runs</h3><label class="field inline">Outcome<select id="wb-parity-outcome">' +
        W.runs.outcomes.map((o, k) => '<option value="' + k + '"' + (k === W.outcome ? " selected" : "") + ">" + esc(A.OUTCOME_LABEL[o]) + "</option>").join("") +
        '</select><small class="hint">Which result to compare with the solver.</small></label></div><p class="flow-note">Each dot is one of ' + check.n + " checking runs the model never saw: its real result (across) against the prediction (up). Dots on the diagonal are perfect; the grey band is the accuracy target. Nondimensional units.</p>" +
        '<div class="result-split"><figure class="chart-figure"><div class="chart-mount parity-mount" id="wb-parity"></div></figure>' +
        '<dl class="result-notes">' +
        "<div><dt>Both outcomes within tolerance</dt><dd>" + Math.round(check.joint_accuracy_success * check.n) + " / " + check.n +
        (check.joint_accuracy_ci ? "<small>95% CI " + F.pct(check.joint_accuracy_ci[0], 1) + "–" + F.pct(check.joint_accuracy_ci[1], 1) + "</small>" : "") + "</dd></div>" +
        (check.rmse ? W.runs.outcomes.map((o) => "<div><dt>RMSE, " + esc(lcFirst(A.OUTCOME_LABEL[o])) + "</dt><dd>" + F.fmt(check.rmse[o], 3) + "<small>tolerance ±" + check.tolerance[W.runs.outcomes.indexOf(o)] + "</small></dd></div>").join("") : "") +
        (Number.isFinite(check.joint_coverage) ? "<div><dt>Interval coverage</dt><dd>" + F.pct(check.joint_coverage, 1) +
          "<small>nominal 90%" + (check.joint_coverage_ci ? "; 95% CI " + F.pct(check.joint_coverage_ci[0], 1) + "–" + F.pct(check.joint_coverage_ci[1], 1) : "") + "</small></dd></div>" : "") +
        "</dl></div></section>" : "") +
      '<details class="wb-section"><summary>Model details</summary><div class="table-scroll"><table class="flow-table"><thead><tr><th>Copy (seed)</th><th class="num">Best step</th><th>Weights fingerprint</th></tr></thead><tbody>' +
      m.members.map((x) => "<tr><td>" + x.seed + '</td><td class="num">' + x.best_step + "</td><td><code>" + short(x.weights_sha256) + "</code></td></tr>").join("") +
      '</tbody></table></div><p class="flow-note">' + esc(profileName(m.profile, m.profile_label)) + "; source checkpoint <code>" + short(m.source_sha256) + "</code>; " +
      esc(m.environment.device_name) + ". Model <code>" + esc(m.model_id) + "</code>.</p></details>"
    );
  }
  // One screen for the open study: its runs, the model and how well it did. Starting a study lives here too.
  function viewStudy() {
    if (!W.view) return viewDescribe();
    const v = latestVersion();
    const job = activeFit();
    const last = lastFit();
    const m = model();
    const enabled = W.cap.adapt.enabled;
    const profiles = Object.entries(W.cap.profiles);
    const canFit = enabled && !job && v && !(m && m.version_id === v.version_id && m.profile === "task8-fno-lhs-v1");
    const check = m && m.verification;
    let lede;
    if (m && check) lede = "Model ready. Checked on " + check.n + " runs it never saw: " + Math.round(check.joint_accuracy_success * check.n) + " predicted within tolerance.";
    else if (m) lede = "Model ready for this study.";
    else if (job) lede = "Adapting the pretrained model to your study.";
    else if (enabled) lede = "Your runs passed the checks. Next, adapt the pretrained model to them.";
    else lede = "Your runs passed the checks. Adapting a model needs a GPU, which is off in this public demo.";
    const about = W.view.example_id
      ? '<p class="new-problem">This is a new flow problem for the model: its runs were never part of pretraining. The pretrained FNO has already been adapted to it on a GPU, so you can go straight to predictions.</p>'
      : !m && !job && !enabled ? '<p class="new-problem">' + GPU_NOTE + "</p>" : "";
    return (
      head("Your study", lede) + about +
      (W.message ? errorHtml(W.message, W.issues) : "") +
      (m ? '<div class="actions"><button class="btn primary" id="wb-go-explore" type="button">Predict outcomes</button></div>' : "") +
      (job ? jobPanel(job) : "") +
      (!job && last && last.status === "failed" ? errorHtml("The last adaptation stopped with an error: " + ((last.error && last.error.message) || "unknown error"), []) : "") +
      (!job && last && last.status === "cancelled" ? noteHtml("The last adaptation was cancelled.") : "") +
      (!enabled && !m ? '<div class="actions"><button class="btn primary" id="wb-example" type="button">Try the worked example instead</button></div>' : "") +
      (canFit ? '<div class="fit-form">' +
        (profiles.length > 1 ? '<label class="field">Profile<select id="wb-profile">' + profiles.map(([k, label]) => '<option value="' + esc(k) + '">' + esc(k === "task8-fno-lhs-v1" ? profileName(k, label) : "Test only: " + label) + "</option>").join("") + "</select></label>" : "") +
        (!v.calibrated_available ? '<label class="check"><input type="checkbox" id="wb-uncal" /> Fit without calibrated ranges (prediction only)</label>' : "") +
        (W.view.worker.available ? "" : noteHtml("No GPU worker is connected right now, so a new job will wait in the queue.")) +
        '<div class="actions"><button class="btn primary" id="wb-fit" type="button">Adapt the model</button></div></div>' : "") +
      '<section class="wb-section"><h3>Your runs</h3>' + versionSummary(v) + "</section>" +
      (m ? '<section class="wb-section"><h3>The model</h3>' + modelPanel(m) + "</section>" : "") +
      '<details class="wb-section"><summary>Use a different study</summary>' + dropZone(v) +
      (W.cap.example.available && !W.view.example_id ? '<div class="actions"><button class="btn" id="wb-example" type="button">Open the worked example</button></div>' : "") +
      '<div id="wb-studies" hidden></div></details>'
    );
  }

  /* ---------- Explore ---------- */
  function band(result, j) {
    if (result.lower) return { lower: result.lower.map((r) => r[j]), upper: result.upper.map((r) => r[j]), calibrated: true };
    return { lower: result.raw_lower.map((r) => r[j]), upper: result.raw_upper.map((r) => r[j]), calibrated: false };
  }
  async function requestCurve() {
    const m = model();
    const g = F.sliceGrid(m.ranges, W.xIndex, W.fixed, CURVE_POINTS);
    const key = JSON.stringify([m.model_id, W.xIndex, W.fixed]);
    if (W.curve && W.curve.key === key && !W.curve.error) return;
    // Keep the last good curve on screen, marked as updating, until the new one arrives.
    W.curve = { key, pending: true, xs: g.xs, prev: W.curve && W.curve.result ? W.curve : W.curve && W.curve.prev };
    renderCurve();
    try {
      const done = await timedPredict(m, g.points);
      if (W.curve && W.curve.key === key) W.curve = { key, xs: g.xs, fixed: W.fixed.slice(), xIndex: W.xIndex, result: done.result };
    } catch (e) {
      if (W.curve && W.curve.key === key) W.curve = { key, error: e.message };
    }
    renderCurve();
  }
  const ctlNames = () => W.cap.controls.map((x) => x.name);
  function sliceRuns(c) {
    const m = model();
    const names = ctlNames();
    const j = W.outcome;
    const tipFor = (r) => runLabel(r) + ": " + names.map((n, k) => F.axisName(n) + " " + F.fmt(r.controls[k], 3)).join(", ") + "; " + lcFirst(A.OUTCOME_LABEL[W.runs.outcomes[j]]) + " " + F.fmt(r.outcomes[j], 4);
    const train = W.runs.runs.filter((r) => r.role === "train");
    const match = train.filter((r) => F.matchesSlice(r.controls, c.fixed, c.xIndex, m.ranges));
    const other = train.filter((r) => !match.includes(r));
    return {
      observed: match.map((r) => ({ x: r.controls[c.xIndex], y: r.outcomes[j], tip: tipFor(r), run: r })),
      projected: W.showProjected ? other.map((r) => ({ x: r.controls[c.xIndex], y: r.outcomes[j], tip: "Not on this slice. " + tipFor(r), run: r })) : [],
      otherCount: other.length,
    };
  }
  function curveHtml(width) {
    const c = W.curve && (W.curve.result ? W.curve : W.curve.prev);
    if (W.curve && W.curve.error) return errorHtml("Predictions failed: " + W.curve.error);
    if (!c) return pending("Computing the trend…");
    const m = model();
    const j = W.outcome;
    const b = band(c.result, j);
    const names = ctlNames();
    const pts = sliceRuns(c);
    const rec = latestRecommendation();
    const next = rec && W.showNext ? rec.rows.map((r, i) => ({ r, i })).filter(({ r }) => F.matchesSlice(r.controls, c.fixed, c.xIndex, m.ranges))
      .map(({ r, i }) => ({ x: r.controls[c.xIndex], label: String(i + 1), tip: "Suggested run " + (i + 1) + ": " + names.map((n, k) => F.axisName(n) + " " + F.fmt(r.controls[k], 3)).join(", ") })) : [];
    const yName = A.OUTCOME_LABEL[W.runs.outcomes[j]];
    return (
      (W.curve.pending ? pending("Updating for the new settings…") : "") +
      '<div class="' + (W.curve.pending ? "is-stale" : "") + '">' +
      F.bandChartSvg({
        width,
        xs: c.xs,
        mean: c.result.mean.map((r) => r[j]),
        lower: b.lower,
        upper: b.upper,
        bandClass: b.calibrated ? "s-pre" : "s-scr",
        observed: pts.observed,
        projected: pts.projected,
        next,
        tips: c.xs.map((x, i) => F.axisName(names[c.xIndex]) + " " + F.fmt(x, 3) + ": predicted " + F.fmt(c.result.mean[i][j], 4) + " (" + F.fmt(b.lower[i], 4) + "–" + F.fmt(b.upper[i], 4) + ")"),
        xLabel: F.axisName(names[c.xIndex]),
        yLabel: yName,
        aria: "Predicted " + lcFirst(yName) + " as " + F.axisName(names[c.xIndex]) + " varies, other controls held fixed, with " + (b.calibrated ? "calibrated 90% intervals." : "raw ensemble spread."),
      }) + "</div>" +
      F.legendHtml([
        { name: "Prediction", cls: "s-pre" },
        { name: b.calibrated ? "Calibrated 90% interval" : "Raw spread (not calibrated)", cls: (b.calibrated ? "s-pre" : "s-scr") + " band-sw" },
        { name: "Training runs on this slice (" + pts.observed.length + ")", cls: "obs-sw" },
      ].concat(pts.projected.length ? [{ name: "Other training runs, projected (" + pts.projected.length + ")", cls: "proj-sw" }] : [])
        .concat(next.length ? [{ name: "Suggested next runs on this slice", cls: "next-sw" }] : [])) +
      '<p class="flow-note">' + (pts.observed.length ? pts.observed.length + " training run" + (pts.observed.length === 1 ? "" : "s") + " share" + (pts.observed.length === 1 ? "s" : "") + " the held settings"
        : "No training run shares the held settings") + " (within " + F.SLICE_TOLERANCE * 100 + "% of each control's range, one slider step). " +
        (W.showProjected ? "Hollow points have different held settings, so the curve is not expected to pass through them." : pts.otherCount + " other training runs are hidden.") + " Outcomes are nondimensional.</p>" +
      F.dataTable([F.axisName(names[c.xIndex]), "Prediction", "Lower", "Upper"], c.xs.map((x, i) => [x, c.result.mean[i][j], b.lower[i], b.upper[i]]), "Curve values as a table")
    );
  }
  function latestRecommendation() {
    const m = model();
    return W.view.recommendations.filter((r) => r.model_id === m.model_id).slice(-1)[0] || null;
  }
  function renderCurve() {
    const box = $("wb-curve");
    if (box) F.mountChart(box, curveHtml);
  }
  /* The Predict story: what the model learned from, that it is fitted, and how much it has
   * been used. The prediction count is the server's own usage ledger for this study. */
  // A compact line under the prediction: what the model learned from, how it was checked, and
  // how many settings it has evaluated (the server's ledger, including the automatic checks).
  // Pending messages carry a start time; one ticker shows elapsed seconds so a slow request
  // reads as working, not stuck. No invented progress.
  const pending = (label) => '<p class="chart-status" role="status" data-since="' + Date.now() + '">' + esc(label) + " <span class=\"elapsed\"></span></p>";
  if (typeof window !== "undefined")
    setInterval(() => {
      document.querySelectorAll("[data-since]").forEach((el) => {
        const s = Math.floor((Date.now() - Number(el.dataset.since)) / 1000);
        const e = el.querySelector(".elapsed");
        if (e && s >= 2) e.textContent = s + " s" + (s >= 20 ? ". The free server can be slow when busy." : "");
      });
    }, 1000);
  function storyHtml() {
    const m = model();
    const used = m.role_counts_used || latestVersion().role_counts;
    const per = W.predictTiming ? W.predictTiming.ms / W.predictTiming.points / 1000 : null;
    const check = m.verification;
    return (
      '<p class="story-line">Trained on <strong>' + used.train + " runs</strong> (plus " + used.development + " to choose checkpoints and " + used.calibration + " to calibrate intervals)" +
      (check ? "; both outcomes within tolerance on <strong>" + Math.round(check.joint_accuracy_success * check.n) + " of " + check.n + "</strong> held-out settings" : "") +
      '; <strong id="wb-pred-count">' + (W.predCount == null ? "…" : W.predCount.toLocaleString()) + "</strong> settings evaluated so far, including the automatic checks on this page" +
      (per ? " (about " + F.fmt(per, 2) + " s each on this server's CPU)" : "") + ".</p>"
    );
  }
  async function refreshCount() {
    try {
      const r = await client.report(W.view.study_id);
      W.predCount = r.usage.untested_settings_predicted;
      W.report = r;
    } catch (e) {
      return;
    }
    if ($("wb-story")) $("wb-story").innerHTML = storyHtml();
  }
  // Time model calls so the counter can say how quick a prediction is. Measured, never assumed.
  async function timedPredict(m, points, fields) {
    const t0 = performance.now();
    const done = await client.settle(await client.predict(m.model_id, points, fields));
    // Only batch calls are timed: a lone point is mostly network round trip, not model time.
    if (points.length >= 16) {
      const t = W.predictTiming || { ms: 0, points: 0 };
      W.predictTiming = { ms: t.ms + performance.now() - t0, points: t.points + points.length };
    }
    refreshCount();
    return done;
  }
  // Where is the model still unsure? A 4 x 4 x 4 grid over the whole study range, each 90% half-width
  // compared with the accuracy target. The widest spots are what the next runs should cover.
  async function computeUnsure() {
    const m = model();
    const L = limits();
    const steps = [0.125, 0.375, 0.625, 0.875];
    const pts = [];
    steps.forEach((a) => steps.forEach((b) => steps.forEach((c) => pts.push([a, b, c].map((t, j) => m.ranges[j][0] + t * (m.ranges[j][1] - m.ranges[j][0]))))));
    W.unsure = { model: m.model_id, pending: true };
    const box = () => $("wb-unsure") && ($("wb-unsure").innerHTML = unsureHtml());
    box();
    try {
      const res = (await timedPredict(m, pts)).result;
      const lo = res.lower || res.raw_lower;
      const hi = res.upper || res.raw_upper;
      const rows = pts.map((x, i) => ({ x, ratio: Math.max(...L.target.map((t, j) => (hi[i][j] - lo[i][j]) / 2 / t)), half: L.target.map((t, j) => (hi[i][j] - lo[i][j]) / 2) }));
      W.unsure = { model: m.model_id, n: pts.length, wide: rows.filter((r) => r.ratio > 1).length, top: rows.sort((a, b) => b.ratio - a.ratio).slice(0, 3), target: L.target.slice() };
    } catch (e) {
      W.unsure = { model: m.model_id, error: e.message };
    }
    box();
  }
  function unsureHtml() {
    const u = W.unsure;
    const names = ctlNames();
    let body;
    if (!u || u.pending) body = pending("Checking 64 probe settings across your study…");
    else if (u.error) body = noteHtml("The check could not run: " + u.error);
    else body =
      "<p>" + (u.wide
        ? "At <strong>" + u.wide + " of " + u.n + "</strong> probe settings across your study, the 90% interval is wider than the accuracy target (±" + u.target.join(", ±") + "). The widest intervals are at:"
        : "At all " + u.n + " probe settings across your study, the 90% interval is inside the accuracy target. A small confirming batch in your solver is still worth running. The widest intervals are at:") + "</p>" +
      '<ol class="unsure-list">' + u.top.map((r) => "<li>" + names.map((n, j) => esc(ctlLabel(n)) + " " + F.fmt(r.x[j], 3)).join(", ") +
        ' <span class="muted">interval ±' + r.half.map((h) => F.fmt(h, 2)).join(" / ±") + "</span></li>").join("") + "</ol>";
    return '<h3>Where the model is least certain</h3>' + body +
      '<p class="flow-note">Probe settings are model-only checks, not solver verification. Next: set a run budget and get a small batch to simulate. New results usually tighten the intervals after re-adaptation, though not at every setting.</p>';
  }
  function viewExplore() {
    const m = model();
    const names = ctlNames();
    const rec = latestRecommendation();
    const train = W.runs.runs.filter((r) => r.role === "train");
    const fixedText = names.map((n, j) => (j === W.xIndex ? null : F.axisName(n) + " " + F.fmt(W.fixed[j], 3))).filter(Boolean).join(", ");
    return (
      head("Predict outcomes", "Enter any setting inside your study's range. You get each outcome with a calibrated 90% prediction interval, without another solver run.") +

      (m.profile !== "task8-fno-lhs-v1" ? '<div class="mock-banner" role="note"><span class="mock-tag">Test model</span><div>' + esc(m.preadapted ? m.preadapted.label : m.profile_label) + "</div></div>" : "") +
      '<section class="predict-box" id="wb-point-form" aria-label="Predict one setting">' +
      '<div class="predict-inputs">' +
      names.map((n, j) => '<label class="field">' + esc(ctlLabel(n)) + ' <small>' + m.ranges[j][0] + " to " + m.ranges[j][1] +
        '</small><input type="number" step="any" min="' + m.ranges[j][0] + '" max="' + m.ranges[j][1] + '" data-pt="' + j + '" value="' + F.fmt((W.point && W.point.x[j]) || F.midpoints(m.ranges)[j], 4) + '" /></label>').join("") +
      '<div class="predict-go"><button class="btn primary" id="wb-predict" type="button">Predict</button></div></div>' +
      '<label class="check"><input type="checkbox" id="wb-fields" ' + (W.point && W.point.fields ? "checked" : "") + " /> Also show the predicted flow fields</label>" +
      '<div id="wb-point" aria-live="polite">' + pointHtml() + "</div>" +
      '<div class="story-strip" id="wb-story">' + storyHtml() + "</div></section>" +
      '<section class="wb-section"><div class="section-bar"><h3>See the trend</h3><button class="text-link link-btn" id="wb-point-to-trend" type="button">Use the setting above in the trend</button></div><p class="flow-note">How one outcome changes as one setting varies. The trend keeps its own held settings, shown on the right. ' +
      (m.calibration.status === "calibrated" ? "Shading is the calibrated 90% prediction interval." : "Shading is raw spread, not a calibrated range.") + "</p>" +
      '<div class="explore">' +
      '<div class="explore-main">' +
      '<div class="toolbar"><label class="field inline">Outcome to plot<select id="wb-outcome">' +
      W.runs.outcomes.map((o, j) => '<option value="' + j + '"' + (j === W.outcome ? " selected" : "") + ">" + esc(A.OUTCOME_LABEL[o]) + "</option>").join("") +
      '</select><small class="hint">' + esc(OUTCOME_HELP[W.runs.outcomes[W.outcome]] || "") + '</small></label><label class="field inline">Setting to vary<select id="wb-xsel">' +
      names.map((n, j) => '<option value="' + j + '"' + (j === W.xIndex ? " selected" : "") + ">" + esc(ctlLabel(n)) + "</option>").join("") +
      '</select><small class="hint">Moves along the bottom axis. The other two stay at the held values on the right.</small></label><p class="toolbar-note" id="wb-fixed-text">Held at ' + esc(fixedText) + "</p></div>" +
      '<figure class="chart-figure"><div class="chart-mount" id="wb-curve"></div></figure>' +
      '<details class="interval-help"><summary>What the interval means</summary><p>' + esc(A.intervalText(m.calibration)) + "</p>" +
      (m.calibration.scope ? '<p class="flow-note">' + esc(m.calibration.scope) + "</p>" : "") + "</details>" +
      "</div>" +
      '<aside class="explore-side">' +
      '<fieldset class="side-group"><legend>Held settings</legend>' +
      names.map((n, j) => (j === W.xIndex ? "" : '<label class="field">' + esc(ctlLabel(n)) + ' <output id="wb-out-' + j + '">' + F.fmt(W.fixed[j], 3) +
        '</output><input type="range" data-fix="' + j + '" min="' + m.ranges[j][0] + '" max="' + m.ranges[j][1] + '" step="' + (m.ranges[j][1] - m.ranges[j][0]) / 100 +
        '" value="' + W.fixed[j] + '" /><small>' + m.ranges[j][0] + " to " + m.ranges[j][1] + "</small></label>")).join("") +
      '<label class="field">Match a training run<select id="wb-hold-run"><option value="">Choose a run…</option>' +
      train.map((r) => '<option value="' + esc(r.run_id) + '">' + esc(runLabel(r)) + ": " + names.map((n, k) => (k === W.xIndex ? null : F.axisName(n) + " " + F.fmt(r.controls[k], 3))).filter((x) => x != null).join(", ") + "</option>").join("") +
      '</select><small class="hint">Sets the held values to one of your finished runs, so its real result appears as a dot on the curve. A quick way to see the prediction pass through a result you already know.</small></label></fieldset>' +
      '<fieldset class="side-group"><legend>Show on the chart</legend>' +
      '<label class="check"><input type="checkbox" id="wb-show-projected"' + (W.showProjected ? " checked" : "") + " /> All training runs (faded where settings differ)</label>" +
      (rec ? '<label class="check"><input type="checkbox" id="wb-show-next"' + (W.showNext ? " checked" : "") + " /> Suggested next runs</label>" : "") +
      "</fieldset>" +
      "</aside></div></section>" +
      '<section class="wb-section" id="wb-unsure">' + unsureHtml() + "</section>" +
      '<div class="actions wb-next-step"><button class="btn primary" id="wb-go-next" type="button">Plan next runs</button></div>'
    );
  }
  const OUTCOME_HELP = {
    kinetic_energy: "Kinetic energy: how much the flow is moving overall at the end of the run.",
    pressure_rms: "Pressure RMS: how strongly the pressure varies across the domain at the end of the run.",
  };
  function pointHtml() {
    const p = W.point;
    if (!p) return "";
    if (p.pending) return pending("Computing your prediction…");
    if (p.error) return errorHtml(p.error);
    const r = p.result;
    const names = ctlNames();
    return (
      '<div class="predict-result">' +
      W.runs.outcomes.map((o, j) => {
        const b = band(r, j);
        return '<div class="pr-item"><span class="pr-label">' + esc(A.OUTCOME_LABEL[o]) + '</span><strong class="pr-value">' + F.fmt(r.mean[0][j], 3) +
          '</strong><span class="pr-range">' + (r.lower ? "90% prediction interval: " : "Uncalibrated spread: ") + F.fmt(b.lower[0], 3) + " to " + F.fmt(b.upper[0], 3) + "</span></div>";
      }).join("") + "</div>" +
      (r.lower ? '<p class="confidence-note">Calibrated so that, on cases like the held-out ones, both outcomes fall inside together about 90% of the time. Not a guarantee for one setting.' +
        (model().verification && Number.isFinite(model().verification.joint_coverage) ? " Measured on this example: " + F.pct(model().verification.joint_coverage, 1) + " of " + model().verification.n + " held-out settings." : "") + "</p>" : "") +
      '<p class="flow-note">For ' + names.map((n, k) => F.axisName(n) + " " + F.fmt(p.x[k], 3)).join(", ") + ". Nondimensional units. A prediction, not a simulation.</p>" +
      (r.fields ? '<div class="field-maps" id="wb-maps"></div>' : "")
    );
  }
  function drawFields(fields) {
    const box = $("wb-maps");
    if (!box || !fields) return;
    box.innerHTML = "";
    // Signed fields on a shared, symmetric, zero-centred diverging scale per field, with a numeric colour bar.
    [[1, "Pressure"], [2, "Velocity u"], [3, "Velocity v"]].forEach(([c, name]) => {
      const grid = fields[0][c];
      let max = 0;
      grid.forEach((row) => row.forEach((v) => (max = Math.max(max, Math.abs(v)))));
      const canvas = document.createElement("canvas");
      canvas.width = 64;
      canvas.height = 64;
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", "Predicted " + name.toLowerCase() + " field at t = 2 on the 64 × 64 grid, colour scale −" + F.fmt(max, 3) + " to +" + F.fmt(max, 3));
      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(64, 64);
      for (let i = 0; i < 64; i++)
        for (let j = 0; j < 64; j++) {
          const t = max ? grid[63 - i][j] / max : 0;
          const k = (i * 64 + j) * 4;
          const rgb = COLORMAPS.BrBG[Math.max(0, Math.min(255, Math.round((t + 1) * 127.5)))];
          img.data[k] = rgb >> 16;
          img.data[k + 1] = (rgb >> 8) & 255;
          img.data[k + 2] = rgb & 255;
          img.data[k + 3] = 255;
        }
      ctx.putImageData(img, 0, 0);
      const bar = document.createElement("canvas");
      bar.width = 128;
      bar.height = 1;
      bar.className = "cbar-strip";
      bar.setAttribute("aria-hidden", "true");
      const bctx = bar.getContext("2d");
      const bimg = bctx.createImageData(128, 1);
      for (let i = 0; i < 128; i++) {
        const rgb = COLORMAPS.BrBG[i * 2];
        bimg.data.set([rgb >> 16, (rgb >> 8) & 255, rgb & 255, 255], i * 4);
      }
      bctx.putImageData(bimg, 0, 0);
      const fig = document.createElement("figure");
      fig.appendChild(canvas);
      const cap = document.createElement("figcaption");
      cap.innerHTML = "<strong>" + esc(name) + "</strong>";
      cap.appendChild(bar);
      const ticks = document.createElement("span");
      ticks.className = "cbar-ticks";
      ticks.innerHTML = "<span>−" + F.fmt(max, 2) + "</span><span>0</span><span>+" + F.fmt(max, 2) + "</span>";
      cap.appendChild(ticks);
      fig.appendChild(cap);
      box.appendChild(fig);
    });
  }
  async function predictPoint() {
    const m = model();
    const requestId = (W.pointRequest = (W.pointRequest || 0) + 1);
    const x = [...document.querySelectorAll("[data-pt]")].map((i) => Number(i.value));
    const bad = x.findIndex((v, j) => !Number.isFinite(v) || v < m.ranges[j][0] || v > m.ranges[j][1]);
    const fields = $("wb-fields").checked;
    if (bad >= 0) {
      W.point = { x, fields, error: "Enter " + F.axisName(W.cap.controls[bad].name) + " between " + m.ranges[bad][0] + " and " + m.ranges[bad][1] + " (the study's range)." };
      const input = document.querySelector('[data-pt="' + bad + '"]');
      if (input) input.setAttribute("aria-invalid", "true");
    } else {
      document.querySelectorAll("[data-pt]").forEach((i) => i.removeAttribute("aria-invalid"));
      W.point = { x, fields, pending: true };
      $("wb-point").innerHTML = pointHtml();
      $("wb-predict").disabled = true;
      try {
        const done = await timedPredict(m, [x], fields);
        if (requestId !== W.pointRequest || !model() || model().model_id !== m.model_id) return;
        W.point = { x, fields, result: done.result };
      } catch (e) {
        if (requestId !== W.pointRequest) return;
        W.point = { x, fields, error: "Prediction failed: " + e.message };
      }
    }
    if (!$("wb-point")) return;
    $("wb-predict").disabled = false;
    $("wb-point").innerHTML = pointHtml();
    if (W.point.result) drawFields(W.point.result.fields);
  }

  /* ---------- Next runs ---------- */
  // Display names describe what each backend policy does; the exact recipe stays visible underneath.
  const PROFILE_NAME = { "task8-fno-lhs-v1": "Standard fine-tuning recipe: three copies, calibrated intervals" };
  const profileName = (id, label) => PROFILE_NAME[id] || label;
  const POLICY_NAME = {
    "mapped-lhs-v1": "Even coverage (recommended)",
    "maximin-v1": "Even coverage, largest gaps first",
    "uncertainty-diversity-experimental": "Uncertainty-driven (experimental)",
  };
  const WHY = { "mapped-lhs-v1": "These are the next settings in this study's space-filling plan, which spreads runs evenly; it did best in our research.",
    "maximin-v1": "These fill the largest gaps between the runs you already have.",
    "uncertainty-diversity-experimental": "These are where the model's spread is widest, kept apart from each other. Experimental: it did worse than even coverage in our research." };
  // The user's limits for the next batch: where to look, how accurate is enough, how many runs.
  function limits() {
    const m = model();
    if (!W.limits || W.limits.model !== m.model_id)
      W.limits = { model: m.model_id, region: m.ranges.map((r) => r.slice()), target: m.tolerance.slice() };
    return W.limits;
  }
  const POLICY_HELP = {
    "mapped-lhs-v1": "Spreads runs evenly across your settings, following the study's own plan. Did best in our research.",
    "maximin-v1": "Spreads runs evenly, filling the biggest empty gaps between the runs you have first.",
    "uncertainty-diversity-experimental": "Picks where the model is least sure, kept apart from each other. Experimental: did worse than even coverage in our research.",
  };
  function viewNext() {
    const m = model();
    const v = latestVersion();
    const L = limits();
    const names = ctlNames();
    const policies = W.cap.recommendation_policies;
    const hasAcq = !!v.acquisition;
    const policy = W.nextOpts.policy || (hasAcq ? "mapped-lhs-v1" : "maximin-v1");
    const isExample = !!W.view.example_id;
    return (
      head("Plan next runs", "Tell Delta-V your limits, and it suggests the few runs worth simulating next, chosen from the adapted model. Run them in your own solver, then add the results.") +
      
      '<form class="limits" id="wb-limits" novalidate>' +
      '<fieldset class="limit-group"><legend>Run budget</legend>' +
      '<label class="field">Runs in this batch<select id="wb-batch">' +
      [2, 4, 8, 12, 16].map((k) => '<option value="' + k + '"' + (k === W.nextOpts.batch ? " selected" : "") + ">" + k + " runs</option>").join("") +
      '</select><small class="hint">The most runs you want to pay for in this round. Our research used 8; small batches of 4 to 8 let you stop sooner.</small></label>' +
      '<label class="field">How to choose them<select id="wb-policy">' +
      Object.entries(policies).filter(([k]) => k !== "mapped-lhs-v1" || hasAcq).map(([k, p]) => '<option value="' + k + '"' + (k === policy ? " selected" : "") + ">" + esc(POLICY_NAME[k] || p.label) + "</option>").join("") +
      '</select><small class="hint" id="wb-policy-help">' + esc(POLICY_HELP[policy] || "") + "</small></label>" +
      "</fieldset>" +
      '<details class="limits-more"><summary>Search range and accuracy target <span class="muted">' + esc(limitsSummary()) + '</span></summary><div class="limits-inner">' +
      '<fieldset class="limit-group"><legend>Search range</legend><p class="hint">Suggestions stay inside these limits. Default: your whole study range.</p>' +
      names.map((n, j) => '<div class="range-row"><span>' + esc(ctlLabel(n)) + '</span><label><span class="visually-hidden">' + esc(ctlLabel(n)) + ' low</span><input type="number" step="any" data-lo="' + j + '" min="' + m.ranges[j][0] + '" max="' + m.ranges[j][1] + '" value="' + L.region[j][0] + '" /></label><span class="to">to</span><label><span class="visually-hidden">' + esc(ctlLabel(n)) + ' high</span><input type="number" step="any" data-hi="' + j + '" min="' + m.ranges[j][0] + '" max="' + m.ranges[j][1] + '" value="' + L.region[j][1] + '" /></label></div>').join("") +
      "</fieldset>" +
      '<fieldset class="limit-group"><legend>Accuracy target</legend><p class="hint">Absolute error you can accept, in nondimensional outcome units. Used for the readiness check.</p>' +
      W.runs.outcomes.map((o, j) => '<div class="range-row"><span>' + esc(A.OUTCOME_LABEL[o]) + '</span><label><span class="visually-hidden">' + esc(A.OUTCOME_LABEL[o]) + ' target</span>± <input type="number" step="any" min="0" data-target="' + j + '" value="' + L.target[j] + '" /></label></div>').join("") +
      "</fieldset>" +
      '</div></details>' +
      '<div class="actions"><button class="btn primary" id="wb-suggest" type="submit">' + (W.next && W.next.pending ? "Finding next runs…" : "Suggest next runs") + "</button></div></form>" +
      (W.message ? errorHtml(W.message, W.issues) : "") +
      '<div id="wb-next" class="wb-section">' + nextHtml() + "</div>" +
      '<section class="wb-section"><h3>Add finished runs</h3>' +
      '<p class="flow-note">Adding results creates a new study version. ' + (W.cap.adapt.enabled ? "The model stays on its current version until you adapt it again." : "The model stays on its current version until it is re-adapted on a GPU, which is off in this demo.") + "</p>" +
      '<div class="actions"><input id="wb-add-file" class="visually-hidden" type="file" accept=".zip,application/zip" /><label class="btn" for="wb-add-file">Upload finished runs (ZIP)</label>' +
      (isExample && W.next && W.next.rows && W.next.rows.length ? '<button class="btn" id="wb-example-complete" type="button">Add the recorded solver results for these runs</button>' : "") +
      "</div>" +
      (isExample ? '<p class="flow-note">Worked example only: these are solver results recorded during our research, replayed here. No new simulation runs.</p>' : "") +
      "</section>"
    );
  }
  function limitsSummary() {
    const m = model();
    const L = limits();
    const whole = L.region.every((r, j) => r[0] <= m.ranges[j][0] && r[1] >= m.ranges[j][1]);
    return (whole ? "whole study range" : "custom range") + ", ±" + L.target.join(" / ±");
  }
  function targetHtml(t) {
    if (!t) return "";
    if (t.pending) return '<p class="chart-status" role="status">Checking against your accuracy target…</p>';
    if (t.error) return noteHtml("The accuracy check could not run: " + t.error);
    const pct = Math.round((t.wide / t.n) * 100);
    return t.wide === 0
      ? '<div class="verdict-box ok"><strong>Intervals are within your target at all ' + t.n + " probe settings in this range.</strong> You may have enough runs for this question; confirm with a few solver runs.</div>"
      : '<div class="verdict-box"><strong>Intervals still exceed your target at ' + t.wide + " of " + t.n + " probe settings (" + pct + "%).</strong> More solver runs are likely to help. Probe settings are model-only checks, not solver verification.</div>";
  }
  function nextHtml() {
    const r = W.next;
    if (!r) return '<p class="flow-note">Set your limits above, then choose Suggest next runs.</p>';
    if (r.pending) return pending("Finding next runs…");
    const names = ctlNames();
    const outcomes = W.runs.outcomes;
    const ci = r.rows.some((row) => row.calibrated_interval);
    return (
      targetHtml(W.targetCheck) +
      (r.rows.length ? '<h3>' + r.rows.length + " suggested run" + (r.rows.length === 1 ? "" : "s") + "</h3>" +
        '<p class="why">' + esc(WHY[r.policy] || r.policy_label) + " Run them in your solver, then upload the completed runs below.</p>" : "") +
      (r.notes || []).filter((n) => !/differs from the eight-run|prefix of the recorded order|batches of/.test(n)).map(noteHtml).join("") +
      (r.rows.length ? '<div class="table-bar"><span class="flow-note">Settings to run, with what the model expects you to see. Nondimensional units.</span><button class="btn small" id="wb-next-csv" type="button">Download CSV</button></div>' +
        '<div class="table-scroll" tabindex="0" role="region" aria-label="Suggested next runs"><table class="flow-table"><thead><tr><th class="num">#</th>' + names.map((n) => '<th class="num">' + esc(ctlLabel(n)) + "</th>").join("") +
        outcomes.map((o) => '<th class="num">Expected ' + esc(lcFirst(A.OUTCOME_LABEL[o])) + (ci ? " (90% range)" : "") + "</th>").join("") + "</tr></thead><tbody>" +
        r.rows.map((row, i) => '<tr><td class="num">' + (i + 1) + "</td>" + row.controls.map((x) => '<td class="num">' + F.fmt(x, 3) + "</td>").join("") +
          outcomes.map((o, j) => '<td class="num">' + F.fmt(row.predicted[j], 3) + (row.calibrated_interval ? ' <span class="muted">(' + F.fmt(row.calibrated_interval[0][j], 3) + "–" + F.fmt(row.calibrated_interval[1][j], 3) + ")</span>" : "") + "</td>").join("") + "</tr>").join("") + "</tbody></table></div>" : "")
    );
  }
  function readLimits() {
    const m = model();
    const L = limits();
    const names = ctlNames();
    const region = names.map((n, j) => [Number(document.querySelector('[data-lo="' + j + '"]').value), Number(document.querySelector('[data-hi="' + j + '"]').value)]);
    const target = W.runs.outcomes.map((o, j) => Number(document.querySelector('[data-target="' + j + '"]').value));
    const bad = region.findIndex(([lo, hi], j) => !(Number.isFinite(lo) && Number.isFinite(hi)) || lo >= hi || lo < m.ranges[j][0] || hi > m.ranges[j][1]);
    if (bad >= 0) throw new Error("Set " + F.axisName(names[bad]) + " from low to high, inside " + m.ranges[bad][0] + " to " + m.ranges[bad][1] + ".");
    if (target.some((t) => !(t > 0))) throw new Error("Each accuracy target must be a number above zero.");
    L.region = region;
    L.target = target;
    return L;
  }
  // Is the model already accurate enough? Predict a 4 x 4 x 4 grid inside the user's range and
  // compare each 90% confidence half-width with the user's target.
  async function checkTarget(m, L) {
    const steps = [0.125, 0.375, 0.625, 0.875];
    const pts = [];
    steps.forEach((a) => steps.forEach((b) => steps.forEach((c) => pts.push([a, b, c].map((t, j) => L.region[j][0] + t * (L.region[j][1] - L.region[j][0]))))));
    W.targetCheck = { pending: true };
    try {
      const done = await client.settle(await client.predict(m.model_id, pts));
      const res = done.result;
      const lo = res.lower || res.raw_lower;
      const hi = res.upper || res.raw_upper;
      const wide = pts.filter((_, i) => L.target.some((t, j) => (hi[i][j] - lo[i][j]) / 2 > t)).length;
      W.targetCheck = { n: pts.length, wide };
    } catch (e) {
      W.targetCheck = { error: e.message };
    }
  }
  async function suggest() {
    const m = model();
    const v = latestVersion();
    const L = readLimits();
    const policy = $("wb-policy").value;
    W.nextOpts = { policy, batch: Number($("wb-batch").value) };
    W.next = { pending: true };
    $("wb-next").innerHTML = nextHtml();
    $("wb-suggest").textContent = "Finding next runs…";
    const full = L.region.every((r, j) => r[0] <= m.ranges[j][0] && r[1] >= m.ranges[j][1]);
    const done = await client.settle(await client.next(m.model_id, Object.assign({ policy, batch: W.nextOpts.batch }, full ? {} : { region: L.region })));
    await checkTarget(m, L);
    W.next = done.result;
    await refreshStudy();
    if (v.version_id !== latestVersion().version_id) W.next = null;
  }
  async function addRuns(fileOrBlob) {
    const v = latestVersion();
    const out = await client.addRuns(W.view.study_id, v.version_id, fileOrBlob);
    await loadStudy(out.study_id);
    W.next = null;
    W.notice = "Version " + latestVersion().number + " created with " + (latestVersion().added_runs || []).length + " added runs." +
      (W.cap.adapt.enabled ? " Adapt again to include them in the model." : " The model still uses version " + (W.view.versions.find((x) => x.version_id === model().version_id) || {}).number + ".");
  }

  /* ---------- Report ---------- */
  function scenarioForm(sc) {
    const i = (sc && sc.inputs) || {};
    const val = (v) => (v == null ? "" : esc(String(v)));
    return (
      '<form id="wb-scenario" class="form-grid scenario-form">' +
      '<label class="field">Your plan counts<select name="basis"><option value="total_plan_including_existing"' + (i.basis !== "future_runs_only" ? " selected" : "") +
      '>All runs, including ones you have</option><option value="future_runs_only"' + (i.basis === "future_runs_only" ? " selected" : "") + ">Only future runs</option></select></label>" +
      '<label class="field">Planned solver runs<input name="planned_runs" type="number" min="0" step="1" inputmode="numeric" value="' + val(i.planned_runs) + '" /></label>' +
      '<label class="field">Cost per run<input name="cost_per_run" type="number" min="0" step="any" value="' + val(i.cost_per_run) + '" /></label>' +
      '<label class="field">Currency<input name="currency" maxlength="3" placeholder="USD" autocomplete="off" value="' + val(i.currency) + '" /></label>' +
      '<div class="actions"><button class="btn" type="submit">Compare with my plan</button></div></form>'
    );
  }
  function viewReport() {
    const r = W.report;
    if (!r) return head("Summary") + (W.message ? errorHtml(W.message, W.issues) : '<p class="chart-status" role="status">Loading the report…</p>');
    const u = r.usage;
    const sc = r.scenario;
    const m = r.model;
    const used = u.labels_consumed_by_latest_model;
    const by = u.supplied_by_role;
    const runtime = Object.entries(u.worker_runtime_seconds);
    const charges = Object.entries(u.known_charges);
    const verOk = m && m.verification ? Math.round(m.verification.joint_accuracy_success * m.verification.n) : null;
    const row = (k, v, note) => "<tr><th scope=\"row\">" + esc(k) + '</th><td class="num">' + v + "</td><td>" + (note || "") + "</td></tr>";
    return (
      head("Summary", "What this study has used and produced so far (version " + r.study.latest_version + ").") +
      '<div class="metric-strip">' +
      '<div><span>Model</span><strong>' + (m ? (m.calibration.status === "calibrated" ? "Ready, calibrated" : "Ready, uncalibrated") : "None yet") + "</strong></div>" +
      (m && m.verification ? "<div><span>Verification</span><strong>" + verOk + " / " + m.verification.n + "</strong><small>held-out runs within tolerance</small></div>" : "") +
      '<div><span>Runs used by the model</span><strong>' + (used ? used.train + used.development + used.calibration : 0) + "</strong>" + (used ? "<small>" + used.train + " + " + used.development + " + " + used.calibration + "</small>" : "") + "</div>" +
      '<div><span>Settings evaluated</span><strong>' + u.untested_settings_predicted + "</strong><small>by the model, including automatic checks</small></div>" +
      "</div>" +
      (m && m.readiness_wording ? '<p class="flow-note">' + esc(m.readiness_wording) + "</p>" : "") +

      '<section class="wb-section"><h3>Data and compute used</h3>' +
      '<div class="table-scroll"><table class="flow-table ledger"><thead><tr><th>Item</th><th class="num">Count</th><th>Note</th></tr></thead><tbody>' +
      row("Runs supplied", u.supplied_total, "All roles, across every version.") +
      row("Training", by.train, used ? "Used to fit the latest model: " + used.train + "." : "") +
      row("Development", by.development, used ? "Used to pick checkpoints: " + used.development + "." : "") +
      row("Calibration", by.calibration, used ? "Used to set intervals: " + used.calibration + "." : "") +
      row("Verification references", by.verification, "Benchmark only; never used to fit, select or calibrate.") +
      row("Next runs suggested", u.new_runs_requested, u.new_runs_completed + " added back so far.") +
      row("Settings evaluated by the model", u.untested_settings_predicted, esc(u.untested_settings_note)) +
      (u.runs_with_solver_seconds ? row("Solver time in your runs", F.fmt(u.supplied_solver_seconds / 60, 3) + " min", esc(u.solver_seconds_note) + " (" + u.runs_with_solver_seconds + " runs report it.)") : "") +
      (runtime.length ? runtime.map(([k, s]) => row("Worker time: " + k, F.fmt(s / 60, 2) + " min", "Measured jobs in this study only. Does not include the example's earlier GPU adaptation.")).join("") :
        row("Worker time", "none", "No worker jobs recorded for this study yet.")) +
      row("Charges", charges.length ? charges.map(([c, a]) => A.money(a, c)).join(", ") : "unknown", charges.length ? "Recorded by you." : "None recorded. Unknown is not zero.") +
      "</tbody></table></div>" +
      '<details><summary>Record a charge</summary><form id="wb-charge" class="form-grid"><label class="field">Amount<input name="amount" type="number" min="0" step="any" required /></label>' +
      '<label class="field">Currency<input name="currency" maxlength="3" placeholder="USD" required /></label><label class="field">Note<input name="note" maxlength="200" /></label>' +
      '<div class="actions"><button class="btn" type="submit">Record charge</button></div></form></details></section>' +

      '<section class="wb-section"><h3>Budget</h3><p class="flow-note">Compare these numbers with the solver runs you planned. The result is budget remaining under your assumptions, not verified savings.</p><p><a class="text-link" href="#how?budget">Edit your budget scenario</a></p></section>' +

      '<section class="wb-section"><h3>Files</h3>' +
      (W.message ? errorHtml(W.message, W.issues) : "") +
      '<div class="actions"><button class="btn" data-artifact="report.md" type="button">Download report</button>' +
      '<button class="btn" id="wb-artifacts" type="button">List all files</button></div><div id="wb-artifact-list"></div></section>' +

      '<section class="wb-section"><h3>Limits</h3>' +
      '<details><summary>Limitations of this tool</summary>' + bullets(r.limitations.map(esc)) + "</details>" +
      '<details><summary>Our research, separate from this study</summary>' + bullets(["Predictions were within tolerance on 597 of 600 runs the model never saw, across three independent repeats.",
        "That was one generated research study type, not customer data.", esc(r.research_evidence.scope), esc(r.research_evidence.details.task8b_savings.status), esc(r.research_evidence.details.task8_prospective_savings.status)]) + "</details></section>" +

      '<details class="wb-section danger-zone"><summary>Delete this study</summary><p class="flow-note">Removes the study, its versions and its models from this server. You will be asked to confirm. This cannot be undone.</p>' +
      '<button class="btn danger" id="wb-delete" type="button">Delete study…</button></details>'
    );
  }
  /* Budget scenario, shown on the How it works page for the study open in Try it. */
  function budgetHtml(r) {
    const sc = r.scenario;
    return (
      '<p class="flow-note">For <strong>' + esc(displayName(r.study)) + "</strong>. " + esc(r.verified_cost_savings) + "</p>" +
      '<div class="scenario">' + scenarioForm(sc) +
      '<div class="scenario-out" aria-live="polite">' + (sc.status === "not_set" ? '<p class="flow-note">' + esc(sc.message || "Enter your plan to compare.") + "</p>" :
        '<table class="flow-table compact"><tbody>' + A.budgetRows(sc).map(([a, b]) => '<tr><th scope="row">' + esc(a) + '</th><td class="num">' + esc(b) + "</td></tr>").join("") + "</tbody></table>") +
      (sc.message && sc.status !== "not_set" ? noteHtml(sc.message) : "") + (sc.money && sc.money.incomplete ? noteHtml(sc.money.incomplete) : "") + "</div></div>" +
      '<p class="flow-note">A scenario against your own plan. It shows budget not yet spent, not proven savings. Unknown inputs stay unknown, and going over plan shows as extra cost.</p>'
    );
  }
  async function renderBudget(box) {
    if (!box) return;
    const id = client.studyId();
    if (!id) {
      box.innerHTML = '<p>Open a study first. <a class="text-link" href="#workbench?start=example">Try the worked example</a></p>';
      return;
    }
    box.innerHTML = '<p class="chart-status" role="status">Loading your study…</p>';
    try {
      if (!W.view || W.view.study_id !== id) {
        W.cap = W.cap || (await client.capabilities());
        await loadStudy(id);
      }
      W.report = await client.report(id);
      const draw = (msg) => {
        box.innerHTML = budgetHtml(W.report) + (msg ? errorHtml(msg) : "");
        const sf = box.querySelector("#wb-scenario");
        sf.onsubmit = async (e) => {
          e.preventDefault();
          const btn = sf.querySelector("button");
          btn.disabled = true;
          btn.textContent = "Comparing…";
          try {
            W.report = await client.scenario(W.view.study_id, formJson(sf));
            draw();
          } catch (err) {
            draw(err.message);
          }
        };
      };
      draw();
    } catch (e) {
      box.innerHTML = errorHtml("Your study could not be loaded: " + e.message);
    }
  }
  async function loadReport() {
    W.report = await client.report(W.view.study_id);
  }
  function formJson(form) {
    const out = {};
    new FormData(form).forEach((v, k) => {
      if (v === "") return;
      out[k] = ["planned_runs", "concurrency"].includes(k) ? parseInt(v, 10) : ["cost_per_run", "wall_hours_per_run", "amount"].includes(k) ? Number(v) : k === "currency" ? String(v).toUpperCase() : v;
    });
    return out;
  }

  function openExample() {
    guard(async () => {
      const v = await client.startExample();
      await loadStudy(v.study_id);
      W.step = model() ? 1 : 0;
      W.notice = null;
    });
  }

  /* ---------- render + events ---------- */
  /* Step states come from server state (study, version, model, jobs, recommendations),
   * never from which step happens to be selected. */
  function stepStates() {
    const v = latestVersion();
    const m = model();
    const fit = activeFit();
    const rec = m && latestRecommendation();
    const noModel = fit ? "Adapting" : W.view ? "Needs a model" : "Open a study first";
    return [
      !W.view ? { state: "available" } : fit ? { state: "running", text: "Adapting" } : m ? (m.version_id === v.version_id ? { state: "done", text: "Model ready" } : { state: "done", text: "Model ready" }) :
        { state: "available", text: W.cap && W.cap.adapt.enabled ? "Not adapted" : "Needs a GPU" },
      m ? { state: "available" } : { state: "blocked", text: noModel },
      m ? (rec ? { state: "done", text: "Suggested" } : { state: "available" }) : { state: "blocked", text: noModel },
      m ? { state: "available" } : { state: "blocked", text: noModel },
    ];
  }
  function renderHeading() {
    const v = latestVersion();
    const m = model();
    const fit = activeFit();
    $("wb-scope").hidden = !!W.view;
    if (!W.view) {
      $("wb-eyebrow").textContent = "WORKBENCH";
      $("wb-title").textContent = "Workbench";
      $("wb-sub").textContent = "Bring compatible simulation runs, adapt the pretrained model, explore untested settings and plan the next batch.";
      return;
    }
    const status = fit ? "Adapting the model" : m ? (m.version_id === v.version_id ? (m.preadapted ? "Example model ready" : "Model ready") : "Model still uses version " + ((W.view.versions.find((x) => x.version_id === m.version_id) || {}).number || 1)) :
      W.cap && W.cap.adapt.enabled ? "Not adapted yet" : "Not adapted (needs a GPU, off in this demo)";
    $("wb-eyebrow").textContent = "WORKBENCH";
    $("wb-title").textContent = displayName(W.view);
    // One compact line: data label, version, model status. Counts live in Your study and Summary.
    const r = m ? modelRuns(m, v) : null;
    $("wb-sub").innerHTML = dataTag(v.data_kind) + " <span>Version " + v.number + "</span> <span class=\"state " + (fit ? "run" : m ? "ok" : "warn") + "\">" + esc(status) +
      (r ? " · trained on " + r.used.train + " runs" + (r.added ? ", " + r.added + " newer runs not yet used" : "") : "") + "</span>";
  }
  function renderSteps() {
    const max = maxStep();
    const states = stepStates();
    renderHeading();
    $("wb-step-label").textContent = "Step " + (W.step + 1) + " of " + STEPS.length;
    $("wb-step-select").innerHTML = STEPS.map((s, i) => '<option value="' + i + '"' + (i === W.step ? " selected" : "") + (i > max ? " disabled" : "") + ">" + (i + 1) + ". " + s +
      (states[i].text ? " (" + states[i].text.toLowerCase() + ")" : "") + "</option>").join("");
    $("wb-step-select").onchange = (e) => setStep(Number(e.target.value));
    $("wb-steps").innerHTML = STEPS.map((s, i) => {
      const st = states[i];
      const locked = i > max;
      return '<li class="is-' + st.state + '"><button type="button" data-step="' + i + '"' + (locked ? ' disabled aria-disabled="true"' : "") + (i === W.step ? ' aria-current="step"' : "") +
        (locked && st.text ? ' title="' + esc(st.text) + '"' : "") + '><span class="n" aria-hidden="true">' + (st.state === "done" ? "✓" : i + 1) + '</span><span class="lbl">' + s +
        (st.text ? '<small>' + esc(st.text) + "</small>" : "") + "</span></button></li>";
    }).join("");
  }
  function render(focus) {
    if (!W.ready) return;
    renderSteps();
    if (!W.cap) {
      $("wb-panel").innerHTML = W.message ? errorHtml(W.message, W.issues) : '<p class="chart-status" role="status">Connecting to the study API…</p>';
      return;
    }
    W.step = Math.min(W.step, maxStep());
    const views = [viewStudy, viewExplore, viewNext, viewReport];
    const previousFocus = document.activeElement && document.activeElement.id;
    $("wb-panel").innerHTML = (W.notice ? okHtml(W.notice) : "") + views[W.step]();
    W.notice = null;
    bind();
    if (W.step === 1) {
      if (W.predCount == null || W.countStudy !== W.view.study_id) {
        W.countStudy = W.view.study_id;
        refreshCount();
      }
      renderCurve();
      // Your own prediction first, then the trend, then the probe check; one server job at a time.
      (async () => {
        if (!W.point) await predictPoint();
        await requestCurve();
        if (!W.unsure || W.unsure.model !== model().model_id) await computeUnsure();
      })();
      if (W.point && W.point.result) drawFields(W.point.result.fields);
    }
    if (W.step === 0 && $("wb-parity")) F.mountChart($("wb-parity"), (w) => parityHtml(model(), w));
    if (W.step === 0) renderStudyList();
    if (W.step === 3 && !W.report && !W.message) guard(loadReport);
    if (!focus && previousFocus && $(previousFocus)) $(previousFocus).focus({ preventScroll: true });
    if (focus) {
      const h = $("wb-panel").querySelector("h2");
      if (h) h.focus({ preventScroll: false });
    }
  }
  function bind() {
    const on = (id, fn) => {
      const n = $(id);
      if (n) n.onclick = fn;
    };
    if (W.step === 0) {
      on("wb-example", openExample);
      const drop = $("wb-drop");
      drop.ondragover = (e) => {
        e.preventDefault();
        drop.classList.add("over");
      };
      drop.ondragleave = () => drop.classList.remove("over");
      drop.ondrop = (e) => {
        e.preventDefault();
        drop.classList.remove("over");
        uploadFile(e.dataTransfer.files[0]);
      };
      $("wb-file").onchange = (e) => uploadFile(e.target.files[0]);
      on("wb-fit", () => guard(async () => {
        const body = { allow_uncalibrated: !!($("wb-uncal") && $("wb-uncal").checked) };
        if ($("wb-profile")) body.profile = $("wb-profile").value;
        await client.fit(W.view.study_id, body);
        await refreshStudy();
      }));
      on("wb-cancel", () => guard(async () => {
        await client.cancel(activeFit().job_id);
        await refreshStudy();
      }));
      on("wb-go-explore", () => setStep(1));
      const po = $("wb-parity-outcome");
      if (po) po.onchange = () => {
        W.outcome = Number(po.value);
        render(false);
      };
    }
    if (W.step === 1) {
      $("wb-outcome").onchange = (e) => {
        W.outcome = Number(e.target.value);
        e.target.nextElementSibling.textContent = OUTCOME_HELP[W.runs.outcomes[W.outcome]] || "";
        renderCurve();
      };
      $("wb-xsel").onchange = (e) => {
        W.xIndex = Number(e.target.value);
        render(false);
      };
      const heldText = () => {
        const t = $("wb-fixed-text");
        if (t) t.textContent = "Held at " + ctlNames().map((n, j) => (j === W.xIndex ? null : F.axisName(n) + " " + F.fmt(W.fixed[j], 3))).filter(Boolean).join(", ");
      };
      let timer = null;
      document.querySelectorAll("[data-fix]").forEach((r) => {
        r.oninput = () => {
          const j = Number(r.dataset.fix);
          W.fixed[j] = Number(r.value);
          $("wb-out-" + j).textContent = F.fmt(W.fixed[j], 3);
          heldText();
        };
        // One request when the slider is released, not one per movement.
        r.onchange = () => {
          clearTimeout(timer);
          timer = setTimeout(requestCurve, 250);
        };
      });
      $("wb-hold-run").onchange = (e) => {
        const run = W.runs.runs.find((r) => r.run_id === e.target.value);
        if (!run) return;
        run.controls.forEach((v, k) => {
          if (k !== W.xIndex) W.fixed[k] = v;
        });
        render(false);
      };
      $("wb-show-projected").onchange = (e) => {
        W.showProjected = e.target.checked;
        renderCurve();
      };
      if ($("wb-show-next")) $("wb-show-next").onchange = (e) => {
        W.showNext = e.target.checked;
        renderCurve();
      };
      on("wb-predict", () => predictPoint());
      on("wb-point-to-trend", () => {
        const x = [...document.querySelectorAll("[data-pt]")].map((i) => Number(i.value));
        x.forEach((v, k) => {
          if (k !== W.xIndex && Number.isFinite(v)) W.fixed[k] = v;
        });
        render(false);
      });
      on("wb-go-next", () => setStep(2));
    }
    if (W.step === 2) {
      $("wb-policy").onchange = (e) => {
        W.nextOpts.policy = e.target.value;
        $("wb-policy-help").textContent = POLICY_HELP[e.target.value] || "";
      };
      $("wb-batch").onchange = (e) => (W.nextOpts.batch = Number(e.target.value));
      $("wb-limits").onsubmit = (e) => {
        e.preventDefault();
        guard(suggest);
      };
      on("wb-next-csv", () => guard(() => downloadArtifact("next-runs-" + W.next.recommendation_id + ".csv")));
      $("wb-add-file").onchange = (e) => e.target.files[0] && guard(() => addRuns(e.target.files[0]));
      on("wb-example-complete", () => guard(async () => {
        const res = await client.exampleCompletions(W.next.rows.map((r) => r.candidate_id));
        await addRuns(await res.blob());
      }));
      on("wb-example-dl", () => guard(async () => {
        const res = await client.exampleCompletions(W.next.rows.map((r) => r.candidate_id));
        download("deltav-example-completed-runs.zip", await res.blob());
      }));
    }
    if (W.step === 3) {
      const cf = $("wb-charge");
      if (cf) cf.onsubmit = (e) => {
        e.preventDefault();
        guard(async () => {
          W.report = await client.charge(W.view.study_id, formJson(cf));
        });
      };
      document.querySelectorAll("[data-artifact]").forEach((b) => (b.onclick = () => guard(() => downloadArtifact(b.dataset.artifact))));
      on("wb-artifacts", () => guard(async () => {
        const list = await client.artifacts(W.view.study_id);
        W.artifacts = list.artifacts;
      }));
      if (W.artifacts && $("wb-artifact-list")) {
        $("wb-artifact-list").innerHTML = '<div class="table-scroll"><table class="flow-table"><thead><tr><th>Artifact</th><th>Kind</th><th>SHA-256</th><th></th></tr></thead><tbody>' +
          W.artifacts.map((a) => "<tr><td>" + esc(a.name) + "</td><td>" + esc(a.kind) + (a.weights_location ? "<br><small>" + esc(a.weights_location) + "</small>" : "") + "</td><td><code>" + short(a.sha256) +
            '</code></td><td><button class="btn small" data-dl="' + esc(a.name) + '" type="button">Download</button></td></tr>').join("") + "</tbody></table></div>";
        document.querySelectorAll("[data-dl]").forEach((b) => (b.onclick = () => guard(() => downloadArtifact(b.dataset.dl))));
      }
      on("wb-delete", () => {
        if (!window.confirm("Delete \u201c" + displayName(W.view) + "\u201d, its versions and its models from the server? This cannot be undone.")) return;
        guard(async () => {
          await client.remove(W.view.study_id);
          client.setStudyId(null);
          W.view = null;
          W.report = null;
          W.step = 0;
          W.notice = "Study deleted.";
        });
      });
    }
  }

  async function show() {
    if (!W.ready) {
      $("wb-steps").addEventListener("click", (e) => {
        const b = e.target.closest("button[data-step]");
        if (b && !b.disabled) {
          if (Number(b.dataset.step) === 3) W.report = null;
          setStep(Number(b.dataset.step));
        }
      });
      W.ready = true;
    }
    render(false);
    try {
      W.cap = await client.capabilities();
      const id = client.studyId();
      if (id && !W.view) {
        try {
          await loadStudy(id);
          W.step = model() && !activeFit() ? 1 : 0;
        } catch (e) {
          // A saved study that no longer exists (deleted, or the demo server restarted) is not an
          // error for the visitor: forget it and start fresh. Other failures are still shown.
          client.setStudyId(null);
          W.view = null;
          if (e.status !== 404) W.message = e.message;
        }
      } else if (W.view) {
        await refreshStudy();
      }
    } catch (e) {
      W.message = "The study API is not reachable: " + e.message;
    }
    // Deep links from the home page: #workbench?start=example | upload.
    const start = (location.hash.split("?")[1] || "").match(/(?:^|&)start=(example|upload)/);
    if (start && W.cap) {
      window.history.replaceState(null, "", "#workbench");
      if (start[1] === "upload") {
        // Start a fresh study: forget the open one in this view (it stays saved on the server).
        W.view = null;
        client.setStudyId(null);
        W.step = 0;
      } else if (W.view && W.view.example_id && model()) W.step = 1;
      else if (W.cap.example.available) return openExample();
    }
    render(false);
  }

  window.DeltaWorkbench = { show, stop: stopPolling, renderBudget };
})();
