// Unit checks execute the production functions with a tiny DOM stand-in.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");
const fs = require("node:fs");
const path = require("node:path");
const source = fs
  .readFileSync(path.join(__dirname, "../static/app.js"), "utf8")
  .split("initLaunch();")[0];
function harness() {
  const nodes = new Map();
  const element = () => ({
    innerHTML: "",
    textContent: "",
    children: [],
    dataset: {},
    style: {},
    setAttribute() {},
    append(el) {
      this.children.push(el);
    },
    replaceChildren(...els) {
      this.children = els;
    },
    querySelector(s) {
      return (this.parts[s] ||= element());
    },
    querySelectorAll() {
      return [];
    },
    parts: {},
  });
  const storage = new Map();
  const timers = new Map();
  let timerId = 0;
  const ctx = vm.createContext({
    console,
    AbortController,
    AbortSignal,
    Uint8Array,
    Float32Array,
    TextDecoder,
    performance,
    URLSearchParams,
    navigator: { doNotTrack: "1" },
    Date,
    btoa,
    atob,
    localStorage: { getItem: () => null },
    sessionStorage: {
      getItem: (k) => storage.get(k),
      setItem: (k, v) => storage.set(k, v),
    },
    document: {
      getElementById: (id) => {
        if (!nodes.has(id)) nodes.set(id, element());
        return nodes.get(id);
      },
      createElement: element,
    },
    setTimeout: (fn, ms) => {
      const id = ++timerId;
      timers.set(id, { fn, ms });
      return id;
    },
    clearTimeout: (id) => timers.delete(id),
    clearInterval() {},
    window: { innerWidth: 1200 },
    fetch: async () => {
      throw new Error("Unexpected network");
    },
  });
  const run = (code) => vm.runInContext(code, ctx);
  run(fs.readFileSync(path.join(__dirname, "../static/launch.js"), "utf8"));
  run(source);
  run(
    'stopPlayback = syncControls = renderStage = renderTimeline = selectionInfo = () => {}; saveRun = r => saved.push(r); comparisonPreview = () => "";',
  );
  ctx.saved = [];
  return { ctx, run, nodes, timers };
}
function setupField(h) {
  h.run(
    'S.frames = new Uint8Array(1); S.sample = {id:"sample-a",dials:{}}; S.active = {system:"shear_flow",display_name:"Shear",channels:CH_CANON}; S.cacheRevision="rev1";',
  );
}
test("shared links encode identifiers, field and frame without arrays", () => {
  const h = harness();
  setupField(h);
  h.ctx.location = { origin: "https://example.test", pathname: "/", hash: "" };
  h.run("S.ch=2; S.tIdx=3");
  assert.equal(
    h.run("shareLink()"),
    "https://example.test/#simulator?system=shear_flow&sample=sample-a&field=velocity_x&frame=3",
  );
  assert.equal(
    h.run('sharedSelection("#simulator?system=shear_flow&frame=3").frame'),
    3,
  );
  h.run("LAUNCH.research=true");
  assert.equal(
    h.run("shareLink()"),
    "https://example.test/#simulator?system=euler_multi_quadrants_openBC",
  );
});
test("anonymous tracking respects privacy signals and sends only an event name", async () => {
  const h = harness();
  const calls = [];
  h.ctx.fetch = async (url, options) => {
    calls.push({ url, options });
  };
  h.run('track("landing_viewed")');
  assert.equal(calls.length, 0);
  h.ctx.navigator.doNotTrack = "0";
  h.ctx.navigator.globalPrivacyControl = true;
  h.run('track("landing_viewed")');
  assert.equal(calls.length, 0);
  h.ctx.navigator.globalPrivacyControl = false;
  h.run('track("landing_viewed")');
  assert.deepEqual(JSON.parse(calls[0].options.body), {
    event: "landing_viewed",
  });
  h.ctx.localStorage.getItem = () => "false";
  h.run('track("landing_viewed")');
  assert.equal(calls.length, 1);
});
test("quality uses explicit server booleans and includes SSIM without inventing old scores", () => {
  const h = harness();
  assert.match(
    h.run("qualityTarget({passes:false,threshold:.15})"),
    /quality-fail.*Outside target/,
  );
  assert.match(
    h.run("qualityTarget({passes:true,threshold:.15})"),
    /quality-pass.*Meets target/,
  );
  assert.match(
    h.run(
      'qualityTarget({passes:null,threshold:null,note:"Directional only"})',
    ),
    /No pass\/fail/,
  );
  assert.match(h.run("qualityTarget({})"), /unavailable/);
  assert.match(
    h.run('qualityHtml({scalar:{ssim:.8765,rel_l2:.27}},["tracer"])'),
    /27.0%.*SSIM\) 0.876/,
  );
});
test("request keeps caller cancellation and maps API errors, not private detail strings", async () => {
  const h = harness();
  h.ctx.fetch = async (_url, options) => {
    h.ctx.signal = options.signal;
    return {
      ok: false,
      status: 422,
      json: async () => ({ detail: "private traceback" }),
    };
  };
  await assert.rejects(
    h.run(
      'request("/x", {signal:(fieldAbortController=new AbortController()).signal})',
    ),
    (e) =>
      e.userMessage.includes("supported range") &&
      !e.userMessage.includes("traceback"),
  );
  h.run("fieldAbortController.abort()");
  assert.equal(h.ctx.signal.aborted, true);
  assert.doesNotMatch(
    h.run('friendlyError(new Error("private traceback"))'),
    /traceback/,
  );
});
test("session cache rehydrates bytes, expires, and invalidates across model revisions", () => {
  const h = harness();
  h.run(
    'S.cacheRevision="a"; cachePrediction(predictionKey("sample"), {data:new Uint8Array(524288),quality:{}}); predictionMemory.clear();',
  );
  assert.equal(
    h.run('cachedPrediction(predictionKey("sample")).data.length'),
    524288,
  );
  assert.equal(
    h.run('S.cacheRevision="b"; cachedPrediction(predictionKey("sample"))'),
    null,
  );
  h.run(
    'S.cacheRevision="a"; predictionMemory.get(predictionKey("sample")).savedAt=0;',
  );
  assert.equal(h.run('cachedPrediction(predictionKey("sample"))'), null);
});
test("repeated field runs use local results without fetch", async () => {
  const h = harness();
  setupField(h);
  h.run(
    'cachePrediction(predictionKey("sample-a"), {data:new Uint8Array(524288),errScales:[1,1,1,1],quality:{scalar:{rel_l2:.27}},seconds:1.7,epoch:59});',
  );
  await h.run("runField()");
  await h.run("runField()");
  assert.equal(h.run("S.resultSource"), "Session-cached prediction");
  assert.equal(h.ctx.saved.length, 2);
  assert.equal(h.run("S.running"), false);
});
test("a second field run aborts the first and stale completion cannot replace it", async () => {
  const h = harness();
  setupField(h);
  const pending = [];
  h.ctx.fetch = (_u, options) =>
    new Promise((resolve, reject) => {
      pending.push({ resolve, signal: options.signal });
      options.signal.addEventListener("abort", () =>
        reject(Object.assign(new Error("cancel"), { name: "AbortError" })),
      );
    });
  const first = h.run("runField()");
  const second = h.run("runField()");
  assert.equal(pending[0].signal.aborted, true);
  await first;
  assert.equal(h.run("S.running"), true);
  h.run("fieldAbortController.abort(); S.loadVersion++; S.running=false;");
  await second;
  assert.equal(h.ctx.saved.length, 0);
  assert.equal(h.nodes.get("notice").textContent, "");
});
test("dial burst schedules one automatic POST after 150ms and cancels older estimate", async () => {
  const h = harness();
  h.run(
    'S.active={system:"shear",dials:["reynolds"],dial_ranges:{reynolds:[1,100]},metrics:[]}; S.dialPos.shear={reynolds:.5}; renderDials();',
  );
  const input = h.nodes.get("dials").children[0].parts.input;
  input.oninput({ target: { value: 600 } });
  input.oninput({ target: { value: 610 } });
  input.oninput({ target: { value: 620 } });
  assert.equal(h.timers.size, 1);
  const timer = [...h.timers.values()][0];
  assert.equal(timer.ms, 150);
  let calls = 0;
  h.ctx.fetch = async () => {
    calls++;
    return { ok: true, json: async () => ({ metrics: {} }) };
  };
  await timer.fn();
  assert.equal(calls, 1);
  assert.equal(h.run("S.dialPos.shear.reynolds"), 0.62);
});
