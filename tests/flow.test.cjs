// Tests for the "bring your runs" flow logic (static/flow.js) and page wiring.
// MOCK: the model under test is a Gaussian process stand-in, not the Delta-V model.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const F = require("../static/flow.js");

const read = (f) => fs.readFileSync(path.join(__dirname, "../static", f), "utf8");
const example = JSON.parse(read("worked-example.json"));
const evidence = JSON.parse(read("sample-efficiency-results.json"));

/* ---------- CSV ---------- */
test("parseCsv handles BOM, CRLF, quotes and escaped quotes", () => {
  const r = F.parseCsv('﻿a,b\r\n"x,1","he said ""hi"""\r\n');
  assert.deepEqual(r.rows, [["a", "b"], ["x,1", 'he said "hi"']]);
  assert.equal(r.unterminated, false);
  assert.equal(F.parseCsv('a\n"open').unterminated, true);
});

test("parseRuns classifies id, numeric and bad columns with line numbers", () => {
  const ok = F.parseRuns("run_id,v,dp\n1,2,3\n2,3,4\n3,4,5\n");
  assert.deepEqual(ok.columns.map((c) => c.kind), ["id", "numeric", "numeric"]);
  assert.equal(ok.issues.length, 0);

  const text = F.parseRuns("regime,v,dp\nlaminar,1,2\nlaminar,2,3\nturb,3,4\n");
  assert.match(text.issues[0].msg, /non-numeric value \("laminar"\) on line 2/);

  const gap = F.parseRuns("v,r,dp\n1,2,3\n2,,4\n");
  assert.match(gap.issues[0].msg, /1 empty cell \(first on line 3\)/);

  const labels = F.parseRuns("case,v,dp\nA,1,2\nB,2,3\nC,3,4\n"); // unique text labels are ids
  assert.equal(labels.columns[0].kind, "id");
});

test("structural problems are reported once, not repeated per column", () => {
  const ragged = F.parseRuns("v,r,dp\n1,2,3\n2,3\n3,4,5\n");
  assert.equal(ragged.issues.length, 1);
  assert.match(ragged.issues[0].msg, /Line 3 has 2 cells but the header has 3/);
  const open = F.parseRuns('v,dp\n"1,2\n');
  assert.equal(open.issues.length, 1);
  assert.equal(F.parseRuns("").issues[0].level, "error");
  assert.match(F.parseRuns("a,b\n").issues[0].msg, /no runs/);
  assert.match(F.parseRuns("a,a\n1,2\n").issues[0].msg, /two columns are both named|Two columns are both named/);
});

/* ---------- dataset limits ---------- */
const table = (n, cols) => {
  const header = Array.from({ length: cols }, (_, j) => (j === cols - 1 ? "y" : "p" + j));
  const rows = Array.from({ length: n }, (_, i) => Array.from({ length: cols }, (_, j) => (j === cols - 1 ? i * 1.3 + 1 : i + j * 0.37 + 1)));
  return F.toCsv(header, rows);
};
const build = (csv, ranges) => {
  const p = F.parseRuns(csv);
  return F.buildDataset(p, F.defaultMapping(p), ranges);
};

test("run-count and parameter limits", () => {
  assert.equal(build(table(2, 2)).ok, false);
  assert.match(build(table(2, 2)).issues[0].msg, /At least 3/);
  const few = build(table(4, 2));
  assert.equal(few.ok, true);
  assert.equal(few.issues[0].level, "warn");
  assert.equal(build(table(F.LIMITS.maxRuns + 1, 2)).ok, false);
  assert.equal(build(table(10, 5)).ok, false); // 4 parameters
  assert.equal(build(table(10, 4)).ok, true); // 3 parameters
});

test("design range must contain the runs; outcome cannot be a parameter", () => {
  const csv = table(6, 3);
  const p = F.parseRuns(csv);
  const m = F.defaultMapping(p);
  const tooNarrow = F.buildDataset(p, m, [[2, 3], [1, 9]]);
  assert.equal(tooNarrow.ok, false);
  assert.match(tooNarrow.issues.map((i) => i.msg).join(" "), /must include all your runs/);
  const clash = F.buildDataset(p, { params: ["p0", "y"], outcome: "y" }, null);
  assert.equal(clash.ok, false);
  const dup = build("a,y\n1,2\n1,2\n2,3\n3,4\n4,5\n5,6\n");
  assert.ok(dup.issues.some((i) => /repeat the same design point/.test(i.msg)));
});

/* ---------- stand-in model ---------- */
test("fit tracks its runs closely and is more uncertain far from them", () => {
  const s = F.sessionFromCsv(example.runs_csv, example.design_ranges);
  assert.equal(s.ok, true);
  const atRuns = F.predict(s.model, s.ds.X);
  // The stand-in smooths slightly (it adds noise when its kernel is ill-conditioned), so not exact.
  atRuns.mean.forEach((m, i) => assert.ok(Math.abs(m - s.ds.y[i]) / s.ds.y[i] < 0.08));
  const far = F.predict(s.model, [[4, 3.0], [0.5, 10]]);
  const meanNear = atRuns.std.reduce((a, b) => a + b, 0) / atRuns.std.length;
  assert.ok(Math.max(...far.std) > 2 * meanNear);
  assert.ok(Number.isFinite(s.model.looRelL2) && s.model.looRelL2 < 0.3);
  assert.ok(s.model.varScale >= 1); // the band is only ever widened
});

test("suggestNext returns distinct in-range points", () => {
  const s = F.sessionFromCsv(example.runs_csv, example.design_ranges);
  const nx = F.suggestNext(s.model, 5);
  assert.equal(nx.points.length, 5);
  assert.equal(new Set(nx.points.map((p) => p.join("|"))).size, 5);
  nx.points.forEach((p) => p.forEach((v, j) => assert.ok(v >= s.ds.ranges[j][0] && v <= s.ds.ranges[j][1])));
});

test("the synthetic solver is deterministic and the example round-trips", () => {
  assert.equal(F.truths.pipe_bend_v1([2.25, 5.33]), 75.27);
  assert.equal(example.status, "mock");
  assert.match(example.label, /Synthetic/);
});

/* ---------- evidence and the claim gate ---------- */
test("runsToTarget interpolates and never invents a crossing", () => {
  assert.equal(F.runsToTarget([1, 2, 4], [0.1, 0.05, 0.02], 0.15), 1);
  assert.equal(F.runsToTarget([1, 2, 4], [0.5, 0.4, 0.3], 0.15), null);
  const v = F.runsToTarget([1, 2, 4, 8], [0.4, 0.3, 0.2, 0.1], 0.15);
  assert.ok(v > 4 && v < 8);
});

test("only a complete, measured file can put a number in the claim", () => {
  assert.equal(F.claimText(evidence).state, "mock");
  assert.doesNotMatch(F.claimText(evidence).text, /\d/);
  assert.equal(F.claimText(null).state, "pending");
  assert.equal(F.claimText(undefined).state, "pending");
  assert.doesNotMatch(F.claimText({ status: "pending" }).text, /\d/);

  const measured = { ...evidence, status: "measured" };
  const c = F.claimText(measured);
  assert.equal(c.state, "measured");
  assert.match(c.text, /fewer runs/);

  // fail closed: a "measured" file missing an arm is treated as pending, with no number
  const broken = { ...measured, arms: { scratch: measured.arms.scratch, pretrained: measured.arms.pretrained } };
  const b = F.claimText(broken);
  assert.equal(b.state, "pending");
  assert.doesNotMatch(b.text, /\d/);
  // mismatched lengths also fail closed
  const ragged = JSON.parse(JSON.stringify(measured));
  ragged.arms.gp.rel_l2.pop();
  assert.equal(F.claimText(ragged).state, "pending");
});

/* ---------- output safety ---------- */
test("escaping and csv output", () => {
  assert.equal(F.esc('<img src=x onerror="a">&\''), "&lt;img src=x onerror=&quot;a&quot;&gt;&amp;&#39;");
  assert.equal(F.toCsv(["a", "b"], [["x,y", 'q"r']]), 'a,b\n"x,y","q""r"\n');
});

test("charts carry a MOCK watermark when asked and are valid svg strings", () => {
  const s = F.sessionFromCsv(example.runs_csv, example.design_ranges);
  const g = F.sliceGrid(s.ds.ranges, 0, F.midpoints(s.ds.ranges), 30);
  const pr = F.predict(s.model, g.points);
  const svg = F.outcomeChartSvg({
    width: 400, xs: g.xs, mean: pr.mean, std: pr.std,
    observed: s.ds.X.map((x, i) => ({ x: x[0], y: s.ds.y[i], tip: "<b>run</b>" })),
    next: [], xLabel: "x", yLabel: "y", watermark: "MOCK DATA", aria: "t",
  });
  assert.match(svg, /MOCK DATA/);
  assert.match(svg, /^<svg[\s\S]*<\/svg>$/);
  assert.doesNotMatch(svg, /<b>run<\/b>/); // tooltips are escaped
  assert.doesNotMatch(svg, /NaN|undefined/);
  const narrow = F.seriesChartSvg({
    width: 360, xLog: true, series: [{ name: "a", cls: "s-pre", points: [{ x: 1, y: 0.4 }, { x: 8, y: 0.1 }] }],
    xLabel: "k", yLabel: "e", hLine: { y: 0.15, label: "t" },
  });
  assert.doesNotMatch(narrow, /NaN|undefined/);
});

/* ---------- page wiring ---------- */
test("every element id the existing scripts look up still exists in index.html", () => {
  const html = read("index.html");
  const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
  const missing = [];
  for (const f of ["app.js", "launch.js", "rollout.js", "pages.js"]) {
    for (const m of read(f).matchAll(/\$\(\s*["']([A-Za-z0-9_-]+)["']\s*\)/g)) if (!ids.has(m[1])) missing.push(f + ": " + m[1]);
  }
  for (const m of read("workbench.js").matchAll(/\$\(\s*["'](wb-steps|wb-panel)["']\s*\)/g)) if (!ids.has(m[1])) missing.push("workbench.js: " + m[1]);
  assert.deepEqual([...new Set(missing)], []);
});

test("navigation: old simulator links keep working and credits are as requested", () => {
  const html = read("index.html");
  const app = read("app.js");
  assert.match(html, /id="page-simulator"/);
  assert.match(html, /href="#simulator"[^>]*data-nav="simulator"/);
  assert.match(html, /Model demo/);
  assert.match(app, /"simulator",/);
  assert.match(app, /location\.replace\(\s*"#simulator"/); // #demo alias keeps the query string
  assert.doesNotMatch(html, /Built with The Well/);
  assert.doesNotMatch(html, /Adya/);
  assert.match(html, /<p>Created by Krish Malik\.<\/p>/);
  assert.match(html, /<a class="btn primary small" href="#workbench">Bring your runs<\/a>/);
});

test("mock labelling is present on every mock surface", () => {
  const html = read("index.html");
  assert.match(html, /mock-badge/);
  assert.match(html, /id="page-workbench"[\s\S]*?mock-banner/);
  assert.equal(example.status, "mock");
  assert.equal(evidence.status, "mock");
  assert.equal(F.FLOW_MODE, "mock");
});
