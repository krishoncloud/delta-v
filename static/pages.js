/* Landing and Evidence pages. Historical research evidence only, from
 * static/research-evidence.json (built from the published Task 8/8b tables by
 * scripts/build_site_evidence.py). Nothing here is a live model output, and the
 * pages say so where each number appears. */
(function () {
  "use strict";
  const F = window.DeltaFlow;
  const $ = (id) => document.getElementById(id);
  const esc = F.esc;
  let evidence = null;

  async function getEvidence() {
    if (!evidence)
      evidence = fetch("research-evidence.json", { cache: "no-store" }).then((r) => {
        if (!r.ok) throw new Error("research-evidence.json returned " + r.status);
        return r.json();
      });
    return evidence.catch((e) => {
      evidence = null;
      throw e;
    });
  }
  const card = (ev, id) => ev.cards.find((c) => c.id === id);

  /* ---------- Landing ---------- */
  async function renderHomeAvailability() {
    try {
      const r = await fetch("/flow/capabilities", { cache: "no-store" });
      if (!r.ok) return;
      const cap = await r.json();
      $("home-gpu-note").textContent = cap.adapt && cap.adapt.enabled
        ? "Adapting your own uploads runs on a GPU worker."
        : "Adapting your own uploads needs a GPU, which is off in this public demo. Uploads are still checked.";
    } catch (e) {
      /* Keep the static note. */
    }
  }
  async function renderHome() {
    renderHomeAvailability();
    const chart = $("home-chart");
    try {
      const ev = await getEvidence();
      const acc = card(ev, "task8_fno_lhs_accuracy");
            $("claim-text").textContent = acc.value.hits + " / " + acc.value.cases;
      $("claim-qual").textContent = "Generated research study · three repeats · 67 labelled runs per fit"; // Generated research data, not customer data yet.
      const rec = ev.recorded_predictions;
      const o = rec.outcomes.kinetic_energy;
      $("home-chart-caption").textContent = "One repeat: " + rec.cases + " held-out settings, 16 training runs (the 600 above combines three repeats). Bars are calibrated 90% prediction intervals; the grey corridor is the ±" + ev.curves.tolerance + " tolerance. Recorded research, not live.";
      F.mountChart(chart, (width) =>
        F.parityChartSvg({
          width,
          maxSide: 520,
          truth: o.truth,
          prediction: o.prediction,
          lower: o.lower,
          upper: o.upper,
          tolerance: ev.curves.tolerance,
          xLabel: "Solver reference",
          yLabel: "Prediction",
          aria: "Kinetic energy predicted against solver reference for " + rec.cases + " held-out research settings (one research study, 16 training runs). Recorded results.",
        }) +
        F.legendHtml([{ name: "Prediction with 90% prediction interval", cls: "s-pre", shape: "circle" }, { name: "Tolerance corridor", cls: "s-gp band-sw" }]) +
        F.dataTable(["Reference", "Prediction", "Lower", "Upper"], o.truth.map((v, i) => [v, o.prediction[i], o.lower[i], o.upper[i]]), "Kinetic energy values as a table"));
    } catch (e) {
      chart.textContent = "The research evidence file could not load.";
      $("claim-text").textContent = "Evidence unavailable.";
    }
  }

  /* ---------- Evidence ---------- */
  const POLICY = {
    fno_lhs: { name: "Delta-V, even coverage", cls: "s-pre", shape: "circle", dash: null },
    fno_active: { name: "Delta-V, uncertainty-driven", cls: "s-scr", shape: "square", dash: "7 4" },
    gp_active: { name: "Baseline (GP), uncertainty-driven", cls: "s-gp", shape: "diamond", dash: "2 4" },
  };
  function curveSeries(ev, metric, fmtY) {
    return Object.keys(POLICY).map((policy) => ({
      name: POLICY[policy].name,
      cls: POLICY[policy].cls,
      shape: POLICY[policy].shape,
      dash: POLICY[policy].dash,
      points: ev.curves.rows.filter((r) => r.policy === policy).map((r) => {
        const v = r[metric];
        const mean = v.reduce((a, b) => a + b, 0) / v.length;
        return { x: r.budget, y: mean, lo: Math.min(...v), hi: Math.max(...v), labels: r.charged_labels, reps: v,
          tip: POLICY[policy].name + ", " + r.budget + " training runs (" + r.charged_labels + " labelled runs incl. development and calibration): mean " + fmtY(mean) + "; repeats " + v.map(fmtY).join(", ") };
      }),
    }));
  }
  function curveChart(series, label, fmtY, yDomain, width) {
    return F.seriesChartSvg({
      width,
      series,
      xTicks: [8, 16, 24, 32],
      xLabel: "Training runs supplied",
      yLabel: label,
      yFmt: fmtY,
      yDomain,
      aria: label + " against training runs for three ways of choosing runs. Points are the mean of three independent repeats; vertical bars span the lowest and highest repeat.",
    });
  }

  async function renderEvidence() {
    const root = $("ev-root");
    const list = (items) => '<ul class="plain-list">' + items.map((i) => "<li>" + esc(i) + "</li>").join("") + "</ul>";
    try {
      const ev = await getEvidence();
      const acc = card(ev, "task8_fno_lhs_accuracy");
      const fmtY = (v) => F.pct(v, 0);
      const series = curveSeries(ev, "joint_accuracy_success", fmtY);
      root.innerHTML =
        '<section class="ev-lead"><p class="ev-figure"><strong>' + acc.value.hits + " of " + acc.value.cases + "</strong> held-out research settings predicted within tolerance</p>" +
        '<p class="ev-qual">' + esc(acc.claim) + " " + esc(acc.qualification) + "</p></section>" +
        '<section class="ev-section"><h2>Accuracy as training runs grow</h2>' +
        '<p class="flow-note">Share of held-out settings where both outcomes landed within ±' + ev.curves.tolerance + ' of the solver. Points are the mean of three independent repeats of the study; bars span the lowest and highest repeat. Each budget also used 12 development and 39 calibration runs.</p>' +
        '<figure class="chart-figure"><div id="evidence-accuracy" class="chart-mount"></div>' +
        F.legendHtml(series.map((s) => ({ name: s.name, cls: s.cls, dash: !!s.dash, shape: s.shape }))) +
        F.dataTable(["Method", "Training runs", "Labelled runs", "Mean", "Lowest", "Highest"], series.flatMap((s) => s.points.map((p) => [s.name, String(p.x), String(p.labels), fmtY(p.y), fmtY(p.lo), fmtY(p.hi)])), "Accuracy values as a table") +
        "</figure></section>" +
'<p class="flow-note ev-sources">Every number here comes from recorded research result tables kept with the project; none is a live model output.</p>';
      F.mountChart($("evidence-accuracy"), (width) => curveChart(series, "Share within tolerance", fmtY, [0, 1], width));
    } catch (e) {
      root.innerHTML = '<p role="alert">The research evidence could not load: ' + esc(e.message) + "</p>";
    }
  }

  /* ---------- Router hook ---------- */
  function onRoute(page) {
    delete document.body.dataset.mock;
    if (page === "home") renderHome();
    if (page === "evidence") renderEvidence();
    if (page === "how" && window.DeltaWorkbench) window.DeltaWorkbench.renderBudget($("how-budget"));
    // #how?budget (from Summary) lands on the budget tool, after the router's scroll to top.
    if (page === "how" && /[?&]budget/.test(location.hash))
      setTimeout(() => {
        const h = $("budget-title");
        h.setAttribute("tabindex", "-1");
        h.scrollIntoView({ block: "start" });
        h.focus({ preventScroll: true });
      }, 0);
    if (page === "workbench" && window.DeltaWorkbench) window.DeltaWorkbench.show();
    else if (window.DeltaWorkbench) window.DeltaWorkbench.stop();
  }

  window.DeltaPages = { onRoute, getEvidence };
})();
