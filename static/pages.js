/* Landing-page and Evidence-page rendering. MOCK / FRONT-END ONLY.
 * Reads static/worked-example.json and static/sample-efficiency-results.json.
 * Nothing from a mock source may reach the landing-page claim: see DeltaFlow.claimText. */
(function () {
  "use strict";
  const F = window.DeltaFlow;
  const MOCK_PAGES = ["home", "workbench", "evidence"];
  const cache = {};

  function getJson(url) {
    if (!cache[url])
      cache[url] = fetch(url, { cache: "no-store" }).then((r) => {
        if (!r.ok) throw new Error(url + " returned " + r.status);
        return r.json();
      });
    return cache[url].catch((e) => {
      delete cache[url];
      throw e;
    });
  }
  const $ = (id) => document.getElementById(id);

  /* ---------- Landing ---------- */
  async function renderClaim() {
    let ev = null;
    try {
      ev = await getJson("sample-efficiency-results.json");
    } catch (e) {
      ev = null; // unreadable file means no claim
    }
    const c = F.claimText(ev);
    $("claim-tag").textContent = { measured: "MEASURED", mock: "NOT MEASURED", pending: "PENDING" }[c.state];
    $("claim-text").textContent = c.text;
  }

  async function renderHomeExample() {
    const chart = $("home-chart");
    const stats = $("home-stats");
    try {
      const ex = await getJson("worked-example.json");
      const s = F.sessionFromCsv(ex.runs_csv, ex.design_ranges);
      if (!s.ok) throw new Error("example did not validate");
      const g = F.sliceGrid(s.ds.ranges, 0, F.midpoints(s.ds.ranges), 60);
      const pr = F.predict(s.model, g.points);
      const nx = F.suggestNext(s.model, 4);
      chart.innerHTML = F.outcomeChartSvg({
        width: chart.clientWidth,
        xs: g.xs,
        mean: pr.mean,
        std: pr.std,
        observed: s.ds.X.map((x, i) => ({
          x: x[0],
          y: s.ds.y[i],
          tip: "Run " + (i + 1) + ": " + s.ds.paramNames.map((n, j) => n + " " + F.fmt(x[j])).join(", ") + " gives " + F.fmt(s.ds.y[i]),
        })),
        next: nx.points.map((p, i) => ({ x: p[0], tip: "Suggested run N" + (i + 1) + ": " + p.map((v) => F.fmt(v)).join(", ") })),
        xLabel: F.axisName(s.ds.paramNames[0]),
        yLabel: F.axisName(s.ds.outcomeName),
        watermark: "MOCK DATA",
        aria: "Mock pipe-bend example: predicted pressure drop against inlet velocity with an uncertainty band, eight runs and four suggested next runs.",
      });
      F.attachTips(chart);
      stats.innerHTML = [
        [s.ds.n + " of " + ex.spec.blind_sweep_runs, "runs used, against a blind sweep of " + ex.spec.blind_sweep_runs],
        [F.pct(s.model.looRelL2), "leave-one-out error, stand-in model (target " + F.pct(ex.spec.target_rel_l2) + ")"],
        ["4", "next runs suggested, where the model is least sure"],
      ]
        .map(([v, l]) => '<div class="stat"><strong>' + F.esc(v) + '<span class="mock-tag">MOCK</span></strong><span>' + F.esc(l) + "</span></div>")
        .join("");
    } catch (e) {
      chart.textContent = "The mock example could not load.";
      stats.innerHTML = "";
    }
  }

  /* ---------- Evidence ---------- */
  const ARMS = [
    { key: "pretrained", name: "Pretrained start", cls: "s-pre" },
    { key: "scratch", name: "From scratch", cls: "s-scr", dash: "7 4" },
    { key: "gp", name: "Gaussian process", cls: "s-gp", dash: "2 4" },
  ];

  function evidenceChart(ev) {
    const series = ARMS.map((a) => {
      const arm = ev.arms[a.key];
      return {
        name: a.name,
        cls: a.cls,
        dash: a.dash,
        points: arm.k.map((k, i) => {
          const sd = arm.std && Number.isFinite(arm.std[i]) ? arm.std[i] : null;
          return {
            x: k,
            y: arm.rel_l2[i],
            lo: sd == null ? null : Math.max(0, arm.rel_l2[i] - sd),
            hi: sd == null ? null : arm.rel_l2[i] + sd,
            tip: a.name + ", " + k + " trajectories: " + F.pct(arm.rel_l2[i]) + (sd == null ? "" : " (spread " + F.pct(sd) + ")"),
          };
        }),
      };
    });
    return (
      F.seriesChartSvg({
        width: Math.max(300, ($("ev-root").clientWidth || 760) - 40),
        series,
        xLog: true,
        xTicks: ev.arms.pretrained.k,
        xLabel: "Training trajectories (k)",
        yLabel: "Relative L2 error",
        yFmt: (v) => F.pct(v),
        hLine: { y: ev.target_rel_l2, label: F.pct(ev.target_rel_l2) + " target" },
        watermark: ev.status === "mock" ? "MOCK DATA" : "",
        aria: "Relative error against number of training trajectories for pretrained start, training from scratch and a Gaussian process, with the error target marked.",
      }) +
      F.legendHtml(ARMS.map((a) => ({ name: a.name, cls: a.cls, dash: !!a.dash })))
    );
  }

  async function renderEvidence() {
    const root = $("ev-root");
    let raw = null;
    try {
      raw = await getJson("sample-efficiency-results.json");
    } catch (e) {
      raw = null;
    }
    const ev = F.normalizeEvidence(raw);
    const tag = $("ev-tag");
    tag.hidden = ev.status === "measured";
    tag.textContent = ev.status === "mock" ? "MOCK" : "PENDING";
    if (!ev.arms) {
      root.innerHTML =
        '<div class="card pad"><h2>Not run yet</h2><p>' + F.esc(ev.label) +
        " This page will show the sweep as soon as results exist. Nothing is shown in the meantime, on purpose.</p>" + methodHtml(ev) + "</div>";
      return;
    }
    const sum = F.evidenceSummary(ev);
    const isMock = ev.status === "mock";
    const mockTag = isMock ? '<span class="mock-tag">MOCK</span>' : "";
    const runs = (v) => (v == null ? "not reached" : F.fmt(v, 2));
    root.innerHTML =
      (isMock
        ? '<div class="mock-banner" role="note"><span class="mock-tag">MOCK</span><div><strong>Mock data.</strong> ' + F.esc(ev.label) +
          " The real sweep (training from scratch, a pretrained start and a Gaussian process, split by trajectory) replaces this file. Until then, no claim on this site depends on it.</div></div>"
        : "") +
      '<div class="card"><div class="card-head"><h2>Error against training trajectories ' + mockTag + '</h2></div><div class="pad">' +
      evidenceChart(ev) +
      '<p class="flow-note">Whiskers show the spread across ' + F.esc(String(ev.seeds)) + " seeds. Metric: " + F.esc(ev.metric) +
      ". Split: by " + F.esc(ev.split) + ". " + F.esc(ev.system) + ".</p></div></div>" +
      '<div class="stat-row">' +
      [
        [runs(sum.pretrained), "trajectories to reach target, pretrained start"],
        [runs(sum.scratch), "trajectories to reach target, from scratch"],
        [runs(sum.gp), "trajectories to reach target, Gaussian process"],
        [sum.ratio ? sum.ratio.toFixed(1) + "x" : "n/a", "scratch divided by pretrained"],
      ]
        .map(([v, l]) => '<div class="stat"><strong>' + F.esc(v) + mockTag + "</strong><span>" + F.esc(l) + "</span></div>")
        .join("") +
      "</div>" +
      '<div class="card pad"><details><summary>View the data as a table</summary><div class="table-scroll"><table class="flow-table"><thead><tr><th class="num">k</th>' +
      ARMS.map((a) => '<th class="num">' + F.esc(a.name) + "</th>").join("") +
      "</tr></thead><tbody>" +
      ev.arms.pretrained.k
        .map((k, i) => '<tr><td class="num">' + k + "</td>" + ARMS.map((a) => '<td class="num">' + F.pct(ev.arms[a.key].rel_l2[i]) + "</td>").join("") + "</tr>")
        .join("") +
      "</tbody></table></div></details></div>" +
      '<div class="card pad">' + methodHtml(ev) + "</div>";
    F.attachTips(root);
  }

  function methodHtml(ev) {
    return (
      "<h3>What this tests</h3><ul class=\"limits-list\">" +
      "<li>The claim is about runs, not accuracy at large data. With plenty of data a model trained from scratch is expected to catch up, so the gap should close on the right of the chart.</li>" +
      "<li>The Gaussian process is the fair competitor at 5 to 20 runs. A pretrained start has to beat it, not only beat training from scratch.</li>" +
      "<li>Train and test trajectories are separate. Windows from one trajectory never appear on both sides.</li>" +
      "<li>The error target (" + F.pct(ev.target_rel_l2 || 0.15) + " relative L2) is self-imposed, as on the Model demo page. It is not a validated engineering tolerance.</li>" +
      "<li>One fluid lane on The Well. This says nothing about other physics or about your own solver data.</li></ul>"
    );
  }

  /* ---------- Router hook ---------- */
  function onRoute(page) {
    if (MOCK_PAGES.includes(page)) document.body.dataset.mock = "1";
    else delete document.body.dataset.mock;
    if (page === "home") {
      renderClaim();
      renderHomeExample();
    }
    if (page === "evidence") renderEvidence();
    if (page === "workbench" && window.DeltaWorkbench) window.DeltaWorkbench.show();
  }

  document.addEventListener("click", (e) => {
    if (e.target.closest && e.target.closest("[data-load-example]")) {
      try {
        sessionStorage.setItem("deltav-load-example", "1");
      } catch (err) {
        /* storage may be blocked; the workbench has its own example button */
      }
    }
  });

  window.DeltaPages = { onRoute, getJson };
})();
