/* Frontend adapter for the proposed binary rollout-v1 contract. No inference here. */
"use strict";
const ROLL = {
  controller: null,
  timer: null,
  result: null,
  frame: 4,
  busy: false,
};
function rolloutRange(first, last, cap) {
  first = Number(first);
  last = Number(last);
  if (
    !Number.isInteger(first) ||
    !Number.isInteger(last) ||
    first < 4 ||
    last < first ||
    last > 3 + cap
  )
    throw new Error(
      "Choose whole frames from f4 to f" +
        (3 + cap) +
        ", with the last frame at or after the first.",
    );
  // To display f7 onward, the model still has to generate f4, f5 and f6.
  return { first, last, n_steps: last - 3 };
}
function rolloutCapability(system) {
  const c = system?.rollout;
  return c?.contract === "deltav-rollout-v1" &&
    c.available === true &&
    Number.isInteger(c.max_steps) &&
    c.max_steps > 0 &&
    c.max_steps <= 32
    ? c
    : null;
}
function stopRolloutPlayback() {
  if (ROLL.timer) clearInterval(ROLL.timer);
  ROLL.timer = null;
  if ($("rollout-play")) $("rollout-play").textContent = "Play predictions";
}
function resetRollout() {
  ROLL.controller?.abort();
  ROLL.controller = null;
  stopRolloutPlayback();
  ROLL.result = null;
  ROLL.busy = false;
  if ($("rollout-playback")) $("rollout-playback").hidden = true;
}
function updateRolloutControls() {
  if (!$("rollout")) return;
  $("rollout").hidden = LAUNCH.research;
  const capability = rolloutCapability(S.active);
  $("rollout-capability").textContent = capability
    ? "AUTOREGRESSIVE PLAYBACK · CONNECTED"
    : "AUTOREGRESSIVE PLAYBACK · BACKEND CONNECTION PENDING";
  const cap = capability?.max_steps || 32;
  $("rollout-start").max = $("rollout-end").max = String(3 + cap);
  $("rollout-run").disabled =
    !capability || !S.frames || S.loading || ROLL.busy;
  $("rollout-start").disabled = $("rollout-end").disabled = ROLL.busy;
  $("rollout-reference").textContent = S.shape
    ? "Loaded reference: f0–f" +
      (S.shape[0] - 1) +
      ". Later steps without matching reference data will be labeled “ground truth unavailable”; no error will be invented."
    : "Reference availability is checked from the loaded sample, not assumed from the requested range.";
  if (!ROLL.busy && !ROLL.result)
    $("rollout-status").textContent = capability
      ? "Choose a range. Every intermediate prediction from f4 is generated; only the requested range is played. The default is 7 steps. A 30-step CPU request measured about 85 seconds including transfer; slower requests may time out."
      : "Range setup is ready, but this system has no connected rollout backend. The existing one-step comparison above remains available. Euler also requires a saved fine-tuned checkpoint.";
}
function validateRollout(parsed, meta, expected) {
  if (
    parsed.shape.join(",") !== [expected.n_steps, 4, 256, 256].join(",") ||
    !(parsed.data instanceof Float32Array) ||
    parsed.data.length !== expected.n_steps * 4 * 256 * 256 ||
    !parsed.data.every(Number.isFinite)
  )
    throw new Error(
      "Invalid rollout arrays: expected finite float32 canonical fields.",
    );
  if (
    meta.contract !== "deltav-rollout-v1" ||
    meta.system !== expected.system ||
    meta.sample_id !== expected.sample ||
    meta.cache_revision !== expected.revision ||
    meta.first_frame !== 4 ||
    !Array.isArray(meta.steps) ||
    meta.steps.length !== expected.n_steps
  )
    throw new Error(
      "Rollout identity or model revision does not match this selection.",
    );
  meta.steps.forEach((step, i) => {
    if (step.frame !== i + 4)
      throw new Error("Rollout frames are not consecutive.");
    for (const score of Object.values(step.quality || {}))
      for (const key of ["rel_l2", "rmse", "ssim"])
        if (score[key] != null && !Number.isFinite(score[key]))
          throw new Error("Invalid rollout quality score.");
  });
  return {
    data: parsed.data,
    meta,
    first: expected.first,
    last: expected.last,
  };
}
async function runRollout() {
  const cap = rolloutCapability(S.active);
  if (!cap || !S.frames || ROLL.busy) return;
  let expected;
  try {
    expected = {
      ...rolloutRange(
        $("rollout-start").value,
        $("rollout-end").value,
        cap.max_steps,
      ),
      system: S.active.system,
      sample: S.sample.id,
      revision: S.cacheRevision,
    };
  } catch (e) {
    $("rollout-status").textContent = e.message;
    return;
  }
  resetRollout();
  const controller = (ROLL.controller = new AbortController());
  ROLL.busy = true;
  updateRolloutControls();
  $("rollout-status").textContent =
    "Generating autoregressive frames from f4 to f" + expected.last + "…";
  try {
    const r = await request("/predict/rollout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        system: expected.system,
        sample_id: expected.sample,
        n_steps: expected.n_steps,
        revision: expected.revision,
      }),
      signal: controller.signal,
    });
    if (!r.headers.get("Content-Type")?.includes("application/octet-stream"))
      throw new Error(
        "The rollout backend has not supplied the agreed binary response.",
      );
    const meta = JSON.parse(r.headers.get("X-Rollout-Meta") || "null");
    if (!meta) throw new Error("Missing rollout metadata.");
    const result = validateRollout(
      parseNpy(await r.arrayBuffer()),
      meta,
      expected,
    );
    if (
      ROLL.controller !== controller ||
      controller.signal.aborted ||
      S.sample?.id !== expected.sample ||
      S.active?.system !== expected.system
    )
      return;
    ROLL.result = result;
    ROLL.frame = expected.first;
    $("rollout-scrubber").min = String(expected.first);
    $("rollout-scrubber").max = String(expected.last);
    $("rollout-status").textContent =
      "Generated f" +
      expected.first +
      "–f" +
      expected.last +
      ". Model computation: " +
      Number(meta.inference_seconds).toFixed(1) +
      " s. Playback FPS is independent of inference speed.";
    renderRollout();
  } catch (e) {
    if (ROLL.controller === controller && e.name !== "AbortError")
      $("rollout-status").textContent =
        "Rollout unavailable: " + (e.userMessage || e.message);
  } finally {
    if (ROLL.controller === controller) {
      ROLL.busy = false;
      $("rollout-run").disabled = false;
      $("rollout-start").disabled = $("rollout-end").disabled = false;
    }
  }
}
function rolloutScores(result, channel) {
  return result.meta.steps
    .filter((s) => s.frame >= result.first && s.frame <= result.last)
    .map((s) => ({
      frame: s.frame,
      value: s.quality?.[channel]?.rel_l2 ?? null,
    }));
}
function chartMarkup(points, current) {
  const valid = points.filter((p) => Number.isFinite(p.value));
  const max = Math.max(0.15, ...valid.map((p) => p.value));
  const first = points[0]?.frame || 4,
    last = points.at(-1)?.frame || first;
  const x = (f) => 55 + ((f - first) / Math.max(1, last - first)) * 555,
    y = (v) => 175 - (v / max) * 150;
  let previous = null,
    shapes = "";
  for (const p of points) {
    if (!Number.isFinite(p.value)) {
      previous = null;
      continue;
    }
    if (previous)
      shapes +=
        '<path d="M' +
        x(previous.frame) +
        " " +
        y(previous.value) +
        "L" +
        x(p.frame) +
        " " +
        y(p.value) +
        '" fill="none" stroke="var(--accent)" stroke-width="2"/>';
    shapes +=
      '<circle cx="' +
      x(p.frame) +
      '" cy="' +
      y(p.value) +
      '" r="' +
      (p.frame === current ? 6 : 4) +
      '" fill="var(--accent)"/>';
    previous = p;
  }
  return (
    '<path d="M55 15V175H625" fill="none" stroke="currentColor"/><text x="3" y="25" fill="currentColor">' +
    (max * 100).toFixed(1) +
    '%</text><text x="15" y="179" fill="currentColor">0%</text><text x="55" y="196" fill="currentColor">f' +
    first +
    '</text><text x="600" y="196" fill="currentColor">f' +
    last +
    '</text><text x="225" y="207" fill="currentColor">Predicted frame · relative L2</text>' +
    shapes
  );
}
function renderRollout() {
  if (!$("rollout-chart")) return;
  const r = ROLL.result;
  if (!r) {
    $("rollout-chart").innerHTML = chartMarkup([], 4);
    $("rollout-chart-note").textContent =
      "No rollout results yet. No error curve is drawn without measured data.";
    return;
  }
  const c = S.ch,
    frame = ROLL.frame,
    step = r.meta.steps[frame - 4],
    offset = ((frame - 4) * 4 + c) * 256 * 256;
  const { lo, hi } = S.scales[c],
    quant = (v) =>
      Math.max(0, Math.min(255, Math.round(((v - lo) / (hi - lo || 1)) * 255)));
  const size = panelSize(
    Math.max(160, ($("rollout-panels").clientWidth || 800) / 3 - 20),
    270,
  );
  const panels = [
    makePanel(
      "Autoregressive prediction",
      "f" + frame,
      256,
      256,
      (i, j) => quant(r.data[offset + i * 256 + j]),
      lo,
      hi,
      lut(c),
      size,
    ),
  ];
  if (frame < S.shape[0]) {
    const [, H, W] = S.shape;
    panels.push(
      makePanel(
        "Recorded reference",
        "f" + frame,
        H,
        W,
        (i, j) => nat(frame, i, j, c),
        lo,
        hi,
        lut(c),
        size,
      ),
    );
    // Display-only bilinear resampling of quantized reference values. Never used as scientific scores.
    const diff = new Float32Array(256 * 256);
    let limit = 0;
    for (let i = 0; i < 256; i++)
      for (let j = 0; j < 256; j++) {
        const x = Math.max(0, Math.min(H - 1, ((i + 0.5) * H) / 256 - 0.5)),
          y = Math.max(0, Math.min(W - 1, ((j + 0.5) * W) / 256 - 0.5));
        const a = Math.floor(x),
          b = Math.floor(y),
          dx = x - a,
          dy = y - b;
        const v =
          (1 - dx) *
            ((1 - dy) * nat(frame, a, b, c) +
              dy * nat(frame, a, Math.min(W - 1, b + 1), c)) +
          dx *
            ((1 - dy) * nat(frame, Math.min(H - 1, a + 1), b, c) +
              dy *
                nat(frame, Math.min(H - 1, a + 1), Math.min(W - 1, b + 1), c));
        diff[i * 256 + j] =
          r.data[offset + i * 256 + j] - (lo + (v / 255) * (hi - lo));
        limit = Math.max(limit, Math.abs(diff[i * 256 + j]));
      }
    panels.push(
      makePanel(
        "Display difference",
        "prediction − quantized reference; approximate",
        256,
        256,
        (i, j) =>
          Math.round(127.5 + (127.5 * diff[i * 256 + j]) / (limit || 1)),
        -limit,
        limit,
        COLORMAPS.coolwarm,
        size,
      ),
    );
  } else {
    const missing = document.createElement("p");
    missing.textContent =
      "Ground truth unavailable for f" +
      frame +
      ". A difference image cannot be computed from this five-frame sample.";
    panels.push(missing);
  }
  $("rollout-playback").hidden = false;
  $("rollout-panels").replaceChildren(...panels);
  $("rollout-scrubber").value = String(frame);
  $("rollout-frame-label").textContent =
    "f" +
    frame +
    " · " +
    (Number.isFinite(step.inference_seconds)
      ? step.inference_seconds.toFixed(2) + " s model computation"
      : "model timing unavailable");
  const points = rolloutScores(r, CH_CANON[c]);
  $("rollout-chart").innerHTML = chartMarkup(points, frame);
  $("rollout-chart").setAttribute(
    "aria-label",
    "Relative L2 by predicted frame. " +
      points
        .map(
          (p) =>
            "f" +
            p.frame +
            ": " +
            (p.value == null ? "unavailable" : pct(p.value)),
        )
        .join("; "),
  );
  $("rollout-chart-note").textContent =
    points
      .map(
        (p) =>
          "f" +
          p.frame +
          ": " +
          (p.value == null
            ? "error unavailable"
            : pct(p.value) + " relative L2"),
      )
      .join(" · ") +
    ". Scores come from full-precision backend evaluation; missing scores are gaps, never zero.";
}
function initRollout() {
  $("rollout-run").onclick = runRollout;
  $("rollout-scrubber").oninput = (e) => {
    stopRolloutPlayback();
    ROLL.frame = Number(e.target.value);
    renderRollout();
  };
  $("rollout-play").onclick = () => {
    if (ROLL.timer) {
      stopRolloutPlayback();
      return;
    }
    if (!ROLL.result) return;
    stopPlayback();
    $("rollout-play").textContent = "Pause predictions";
    ROLL.timer = setInterval(
      () => {
        ROLL.frame =
          ROLL.frame >= ROLL.result.last ? ROLL.result.first : ROLL.frame + 1;
        renderRollout();
      },
      1000 / Number($("rollout-fps").value),
    );
  };
  $("rollout-fps").onchange = () => {
    if (ROLL.timer) {
      stopRolloutPlayback();
      $("rollout-play").click();
    }
  };
}
