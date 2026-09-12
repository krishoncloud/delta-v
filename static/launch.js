/* Launch flow, plain-language context, sharing and anonymous aggregate events. */
"use strict";
const EULER_SYSTEM = "euler_multi_quadrants_openBC";
const SAMPLE_NAMES = {
  "trl_tcool_0.03": "Rapid-cooling turbulent layer",
  "trl_tcool_1.00": "Slower-cooling turbulent layer",
  rb_Ra1e8_Pr1: "Convection at Rayleigh 100 million",
  "rb_Ra1e10_Pr0.1": "Convection at Rayleigh 10 billion",
  sf_Re1e5_Sc1: "Shear mixing at Reynolds 100,000",
  "sf_Re5e5_Sc0.1": "Shear mixing at Reynolds 500,000",
};
const SYSTEM_NAMES = {
  turbulent_radiative_layer_2D: "Cooling-driven turbulence",
  rayleigh_benard: "Heat-driven convection",
  shear_flow: "Shear-driven mixing",
};
const FIELD_NAMES = {
  density: "Density — how concentrated the fluid is",
  buoyancy: "Buoyancy — tendency to rise or sink",
  tracer: "Tracer — how the fluids mix",
  pressure: "Pressure",
  velocity_x: "Horizontal velocity",
  velocity_y: "Vertical velocity",
};
const LAUNCH = {
  research: false,
  example: false,
  completed: false,
  restoreFrame: null,
  selectedAt: null,
  firstMs: null,
  repeatMs: null,
  readyMs: null,
  started: performance.now(),
};
function sampleName(sample) {
  if (sample?.provenance?.split === "official-test")
    return "Official-test rollout · reference through f10";
  return (
    SAMPLE_NAMES[sample?.id || sample] ||
    sample?.id ||
    sample ||
    "No example selected"
  );
}
function track(event) {
  if (!event) return;
  if (
    navigator.doNotTrack === "1" ||
    navigator.globalPrivacyControl ||
    storageGet("deltav-analytics", true) === false
  )
    return;
  fetch("/events", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ event }),
    keepalive: true,
  }).catch(() => {});
}
function sharedSelection(hash = location.hash) {
  const q = new URLSearchParams(hash.split("?")[1] || "");
  return {
    system: q.get("system"),
    sample: q.get("sample"),
    field: q.get("field"),
    frame: q.has("frame") ? Number(q.get("frame")) : null,
    example: q.get("tour") === "1" || q.get("example") === "1",
  };
}
async function applySelectionLink() {
  const link = sharedSelection();
  LAUNCH.example = link.example;
  LAUNCH.completed = false;
  if (
    link.system === EULER_SYSTEM &&
    !S.systems.some((s) => s.system === EULER_SYSTEM)
  ) {
    showResearch();
    return;
  }
  const sys =
    S.systems.find((s) => s.system === link.system) ||
    S.systems.find((s) => s.system === "shear_flow") ||
    S.systems[0];
  const sample = S.samples.find(
    (s) => s.id === link.sample && s.system === sys.system,
  );
  S.ch = Math.max(0, CH_CANON.indexOf(link.field));
  LAUNCH.restoreFrame =
    Number.isInteger(link.frame) && link.frame >= 0 && link.frame <= 4
      ? link.frame
      : null;
  await selectSystem(sys, sample?.id);
}
function showResearch() {
  resetRollout();
  fieldAbortController?.abort();
  sampleAbortController?.abort();
  parametricController?.abort();
  clearTimeout(parametricDebounce);
  stopPlayback();
  S.loadVersion++;
  S.paramVersion++;
  LAUNCH.research = true;
  S.frames = null;
  S.pred = null;
  S.quality = null;
  S.loading = false;
  S.running = false;
  $("systems").value = EULER_SYSTEM;
  $("euler-research").hidden = false;
  $("rollout").hidden = true;
  $("parametric").hidden = true;
  $("field-workspace").hidden = true;
  $("example-guide").hidden = true;
  ["channels", "samples", "time-select", "run"].forEach(
    (id) => ($(id).disabled = true),
  );
  ["channels", "samples", "time-select"].forEach((id) => {
    $(id).replaceChildren(new Option("Not available yet", ""));
  });
  $("sample-description").textContent =
    "Euler shock interactions · research plan, not a served prediction";
  $("share-box").hidden = true;
  $("run").textContent = "Static probe results";
  $("selection-info").textContent =
    "Euler shock interactions · static probe results · no saved checkpoint";
  $("load-status").textContent =
    "Fourth system: recorded two-arm results, not live inference.";
  $("share-result").disabled = false;
  $("binary-result").hidden = true;
  if (window.history?.replaceState && page === "simulator")
    window.history.replaceState(null, "", shareLink());
  track("system_selected");
}
function showServedSystem() {
  LAUNCH.research = false;
  $("euler-research").hidden = true;
  $("rollout").hidden = false;
  $("parametric").hidden = false;
  $("field-workspace").hidden = false;
}
function updateLaunchUI() {
  syncSelectionURL();
  updateRolloutControls();
  if (!S.active || LAUNCH.research) return;
  $("load-status").textContent = S.loading
    ? "Loading recorded simulation frames…"
    : S.running
      ? "Fetching cached prediction…"
      : S.pred
        ? "Comparison ready. All four fields have measured errors below."
        : "Select an example to load its comparison.";
  $("share-result").disabled = !S.sample || !S.frames || S.loading || S.running;
  $("example-guide").hidden =
    !LAUNCH.example || S.sample?.id !== "sf_Re1e5_Sc1";
  $("example-complete").hidden = !LAUNCH.completed;
  $("example-progress").textContent = LAUNCH.completed
    ? "Example ready — compare the three panels, then tell us what you learned."
    : "Preparing your example. Loading time depends on the connection.";
  $("sample-description").textContent =
    sampleName(S.sample) + (S.sample ? " · ID: " + S.sample.id : "");
  $("binary-result").hidden = !S.pred;
  if (S.sample)
    $("binary-result").href =
      "/samples/" +
      encodeURIComponent(S.sample.id) +
      "/predict?revision=" +
      encodeURIComponent(S.cacheRevision);
  const q = S.quality?.[CH_CANON[S.ch]];
  $("result-explanation").hidden = !q || S.running;
  if (q)
    $("result-explanation").textContent =
      q.passes === false
        ? "This field’s relative error is " +
          pct(q.rel_l2) +
          ", above the self-imposed 15% release target. Inspect the difference panel before using this approximation. This score alone does not establish that the physical structures are correct."
        : q.passes === true
          ? "This field’s relative error is " +
            pct(q.rel_l2) +
            ", below the self-imposed 15% release target on this example. " +
            (S.sample?.provenance?.split === "official-test"
              ? "This is a small official-test clip, not a comprehensive generalization benchmark."
              : "This training example does not measure generalization. See About for the separate official-test check.")
          : "This velocity field has " +
            pct(q.rel_l2) +
            " relative error. There is no quantitative acceptance target for velocity in this release; the score is provided for inspection.";
  $("timing-summary").textContent =
    "Ready after page open: " +
    (LAUNCH.readyMs === null ? "waiting" : Math.round(LAUNCH.readyMs) + " ms") +
    " · first comparison including sample load: " +
    (LAUNCH.firstMs === null ? "pending" : Math.round(LAUNCH.firstMs) + " ms") +
    " · latest repeat retrieval: " +
    (LAUNCH.repeatMs === null
      ? "not requested"
      : Math.round(LAUNCH.repeatMs) + " ms") +
    ". These measure this browser session, separately from model computation.";
}
async function decodeDisplay(response, signal) {
  if (!response.headers.get("Content-Type")?.includes("image/png"))
    return parseNpy(await response.arrayBuffer());
  const shape = JSON.parse(response.headers.get("X-Npy-Shape"));
  const size = shape.reduce((a, b) => a * b, 1);
  if (
    shape.length !== 4 ||
    !shape.every((n) => Number.isInteger(n) && n > 0) ||
    size > 6_000_000
  )
    throw new Error("Unsupported display dimensions");
  const blob = await response.blob();
  if (signal?.aborted) throw new DOMException("Cancelled", "AbortError");
  const bitmap = await createImageBitmap(blob, {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  if (bitmap.width * bitmap.height !== size) {
    bitmap.close();
    throw new Error("Display size mismatch");
  }
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const data = new Uint8Array(size);
  if (response.headers.get("X-Pixel-Layout") === "TCHW") {
    const [T, H, W, C] = shape;
    for (let t = 0; t < T; t++)
      for (let c = 0; c < C; c++)
        for (let p = 0; p < H * W; p++)
          data[(t * H * W + p) * C + c] = pixels[((t * C + c) * H * W + p) * 4];
  } else for (let i = 0; i < size; i++) data[i] = pixels[i * 4];
  return { shape, data };
}
async function displayRequest(url, fallback, signal) {
  try {
    return await request(url, { signal });
  } catch (e) {
    if (signal.aborted) throw e;
    return request(fallback, { signal });
  }
}
function shareLink() {
  // The controls are the displayed selection. Never reuse the incoming hash.
  const uiSystem = $("systems").value || S.active?.system;
  const uiSample = $("samples").value || S.sample?.id;
  const uiChannel =
    $("channels").value === undefined ? S.ch : Number($("channels").value);
  const uiFrame =
    $("time-select").value === undefined
      ? S.tIdx
      : Number($("time-select").value);
  const q = new URLSearchParams(
    LAUNCH.research
      ? { system: EULER_SYSTEM }
      : {
          system: uiSystem,
          sample: uiSample,
          field: CH_CANON[uiChannel] || CH_CANON[S.ch],
          frame: String(Number.isInteger(uiFrame) ? uiFrame : S.tIdx),
        },
  );
  if (LAUNCH.example && !LAUNCH.research) q.set("tour", "1");
  return location.origin + location.pathname + "#simulator?" + q;
}
function syncSelectionURL() {
  if (
    !window.history?.replaceState ||
    page !== "simulator" ||
    !S.sample ||
    S.loading ||
    S.running ||
    LAUNCH.research
  )
    return;
  const link = shareLink();
  window.history.replaceState(null, "", link);
  if (!$("share-box").hidden) {
    $("share-url").value = link;
    $("share-label").textContent =
      "Current selection link — copy again after changing a control.";
  }
}
function initLaunch() {
  initRollout();
  loadHeldOut();
  $("menu-toggle").onclick = () => {
    const open = $("workspace").classList.toggle("menu-open");
    $("menu-toggle").setAttribute("aria-expanded", String(open));
  };
  $("share-result").onclick = async () => {
    if (!LAUNCH.research && (!S.frames || S.loading || S.running)) return;
    const link = shareLink();
    if (window.history?.replaceState)
      window.history.replaceState(null, "", link);
    $("share-url").value = link;
    $("share-box").hidden = false;
    try {
      await navigator.clipboard.writeText(link);
      $("share-label").textContent =
        "Link copied — anyone can open this selection.";
    } catch {
      $("share-label").textContent = "Copy this link to share the selection.";
    }
  };
  document
    .querySelectorAll(
      'a[href="https://github.com/krishoncloud/delta-v/issues"]',
    )
    .forEach((a) => {
      a.href = "#feedback";
      a.removeAttribute("target");
      a.textContent = "Give feedback →";
    });
  $("feedback-linkedin").oninput = (e) => {
    const value = e.target.value.trim();
    let looksLinkedIn = !value;
    try {
      const host = new URL(
        value.includes("://") ? value : "https://" + value,
      ).hostname.toLowerCase();
      looksLinkedIn = host === "linkedin.com" || host.endsWith(".linkedin.com");
    } catch {}
    $("linkedin-note").textContent = looksLinkedIn
      ? "Optional professional profile link. Unusual URL formats are allowed."
      : "This does not look like a linkedin.com URL. Check it if needed; you can still prepare feedback.";
  };
  $("feedback-form").onsubmit = async (e) => {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    const body = [
      "Delta-V prototype feedback",
      ...Array.from(form, ([k, v]) => k + ": " + v),
    ].join("\n\n");
    $("feedback-output").value = body;
    $("feedback-options").hidden = false;
    $("feedback-github").href =
      "https://github.com/krishoncloud/delta-v/issues/new?title=" +
      encodeURIComponent("Prototype feedback") +
      "&body=" +
      encodeURIComponent(body);
    const blob = new Blob([body], { type: "text/plain" });
    if ($("feedback-download").dataset.blob)
      URL.revokeObjectURL($("feedback-download").dataset.blob);
    const url = URL.createObjectURL(blob);
    $("feedback-download").href = url;
    $("feedback-download").dataset.blob = url;
  };
  $("feedback-copy").onclick = async () => {
    try {
      await navigator.clipboard.writeText($("feedback-output").value);
      $("feedback-copy").textContent = "Copied";
    } catch {
      $("feedback-output").focus();
      $("feedback-output").select();
    }
  };
  $("analytics-toggle").checked =
    storageGet("deltav-analytics", true) !== false;
  $("analytics-toggle").onchange = (e) => {
    try {
      localStorage.setItem(
        "deltav-analytics",
        JSON.stringify(e.target.checked),
      );
    } catch {}
  };
}

async function loadHeldOut() {
  try {
    const response = await request("heldout-results.json");
    const report = await response.json();
    if (report.results?.length !== 6) throw new Error("Incomplete test report");
    let passed = 0;
    $("heldout-body").innerHTML = report.results
      .map((sample) =>
        CH_CANON.map((channel, c) => {
          const q = sample.quality[channel];
          if (q.passes === true) passed++;
          const scalar = {
            shear_flow: "Tracer",
            rayleigh_benard: "Buoyancy",
            turbulent_radiative_layer_2D: "Density",
          }[sample.system];
          return (
            "<tr><td>" +
            escapeHtml(SYSTEM_NAMES[sample.system]) +
            '<small class="mono">' +
            escapeHtml(sample.id) +
            '</small></td><th scope="row">' +
            escapeHtml(c === 0 ? scalar : FIELD_NAMES[channel]) +
            '</th><td class="mono">' +
            pct(q.rel_l2) +
            '</td><td class="mono">' +
            fmtSci(q.rmse) +
            '</td><td class="mono">' +
            (Number.isFinite(q.ssim) ? q.ssim.toFixed(3) : "—") +
            "</td><td>" +
            qualityTarget(q) +
            "</td></tr>"
          );
        }).join(""),
      )
      .join("");
    $("heldout-status").textContent =
      passed +
      " of 12 scalar/pressure results meet the 15% target. All 12 velocity results are reported without pass/fail. Limited to these six one-step windows.";
  } catch {
    $("heldout-status").textContent =
      "Published test scores could not load. No substitute results are shown.";
  }
}
