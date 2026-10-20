/* Delta-V /flow API client and display logic for the study workbench.
 *
 * The browser keeps only a session token and the open study id (per-viewer
 * conveniences). Every study, job, model, prediction, recommendation and report
 * is read back from the server, so a refresh restores the real state. Failures
 * are thrown as ApiError and shown; nothing is computed locally instead.
 */
(function (root) {
  "use strict";

  const TOKEN_KEY = "deltav-flow-token";
  const STUDY_KEY = "deltav-flow-study";

  class ApiError extends Error {
    constructor(status, message, issues, payload) {
      super(message);
      this.status = status;
      this.issues = issues || [];
      this.payload = payload || null;
    }
  }

  function safeStore(store) {
    return {
      get: (k) => {
        try {
          return store ? store.getItem(k) : null;
        } catch (e) {
          return null;
        }
      },
      set: (k, v) => {
        try {
          if (store) v == null ? store.removeItem(k) : store.setItem(k, v);
        } catch (e) {
          /* storage blocked: the session lasts for this page only */
        }
      },
    };
  }

  async function errorFrom(res) {
    let body = null;
    try {
      body = await res.json();
    } catch (e) {
      body = null;
    }
    const detail = body && (typeof body.detail === "string" ? body.detail : Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : null);
    const fallback = {
      401: "Your session is not recognised by this server.",
      404: "Not found for this session. It may have been deleted, or created in another browser.",
      409: "That conflicts with the current state. Reload and try again.",
      410: "This model is no longer available. Refit the study version.",
      413: "The upload is larger than the server accepts (64 MB).",
      429: "The server is busy or rate-limited. Wait a moment and try again.",
      503: "The server is not ready for this action.",
    }[res.status];
    return new ApiError(res.status, detail || fallback || "The server returned an error (" + res.status + ").", body && body.issues, body);
  }

  function createClient(opts) {
    const fetchFn = (opts && opts.fetch) || root.fetch.bind(root);
    const store = safeStore(opts && "storage" in opts ? opts.storage : root.localStorage);
    const sleep = (opts && opts.sleep) || ((ms) => new Promise((r) => setTimeout(r, ms)));
    let token = store.get(TOKEN_KEY);

    async function session() {
      const res = await fetchFn("/flow/session", { method: "POST" });
      if (!res.ok) throw await errorFrom(res);
      token = (await res.json()).token;
      store.set(TOKEN_KEY, token);
      return token;
    }

    async function request(path, o) {
      o = o || {};
      if (!token) await session();
      const send = () =>
        fetchFn(path, {
          method: o.method || "GET",
          body: o.body,
          headers: Object.assign({ Authorization: "Bearer " + token }, o.json !== undefined ? { "Content-Type": "application/json" } : {}, o.headers || {}),
          ...(o.json !== undefined ? { body: JSON.stringify(o.json) } : {}),
          signal: o.signal,
        });
      let res = await send();
      if (res.status === 401 && !o.noRetry) {
        // Server data was reset or the token expired: start a new, empty session.
        store.set(STUDY_KEY, null);
        await session();
        res = await send();
      }
      if (!res.ok) throw await errorFrom(res);
      if (o.raw) return res;
      return res.status === 204 ? null : res.json();
    }

    async function job(jobId, o) {
      return request("/flow/jobs/" + encodeURIComponent(jobId), o);
    }

    /** Resolve a 200/202 job response to its finished job, polling the status URL. */
    async function settle(view, o) {
      o = o || {};
      let current = view;
      while (current.status === "queued" || current.status === "running") {
        if (o.onUpdate) o.onUpdate(current);
        await sleep(o.interval || 1500);
        if (o.signal && o.signal.aborted) throw new ApiError(0, "Cancelled");
        current = await job(current.job_id, { signal: o.signal });
      }
      if (current.status === "failed") throw new ApiError(409, (current.error && current.error.message) || "The job failed.", [], current);
      if (current.status === "cancelled") throw new ApiError(409, "The job was cancelled.", [], current);
      return current;
    }

    return {
      request,
      job,
      settle,
      get token() {
        return token;
      },
      studyId: () => store.get(STUDY_KEY),
      setStudyId: (id) => store.set(STUDY_KEY, id),
      capabilities: () => request("/flow/capabilities"),
      example: () => request("/flow/examples/worked"),
      startExample: () => request("/flow/examples/worked/start", { method: "POST" }),
      upload: (file) => request("/flow/studies", { method: "POST", body: file, headers: { "Content-Type": "application/zip" } }),
      study: (id) => request("/flow/studies/" + encodeURIComponent(id)),
      studies: () => request("/flow/studies"),
      runs: (id) => request("/flow/studies/" + encodeURIComponent(id) + "/runs"),
      fit: (id, body) => request("/flow/studies/" + encodeURIComponent(id) + "/fits", { method: "POST", json: body || {} }),
      cancel: (jobId) => request("/flow/jobs/" + encodeURIComponent(jobId) + "/cancel", { method: "POST" }),
      predict: (modelId, points, fields) =>
        request("/flow/models/" + encodeURIComponent(modelId) + "/predict", { method: "POST", json: { points, fields: !!fields } }),
      next: (modelId, body) => request("/flow/models/" + encodeURIComponent(modelId) + "/next", { method: "POST", json: body }),
      addRuns: (id, parent, file) =>
        request("/flow/studies/" + encodeURIComponent(id) + "/versions?parent_version_id=" + encodeURIComponent(parent), {
          method: "POST",
          body: file,
          headers: { "Content-Type": "application/zip" },
        }),
      exampleCompletions: (ids) => request("/flow/examples/worked/completions?ids=" + encodeURIComponent(ids.join(",")), { raw: true }),
      report: (id) => request("/flow/studies/" + encodeURIComponent(id) + "/report"),
      scenario: (id, body) => request("/flow/studies/" + encodeURIComponent(id) + "/scenario", { method: "PUT", json: body }),
      charge: (id, body) => request("/flow/studies/" + encodeURIComponent(id) + "/charges", { method: "POST", json: body }),
      artifacts: (id) => request("/flow/studies/" + encodeURIComponent(id) + "/artifacts"),
      artifact: (id, name) => request("/flow/studies/" + encodeURIComponent(id) + "/artifacts/" + encodeURIComponent(name), { raw: true }),
      remove: (id) => request("/flow/studies/" + encodeURIComponent(id), { method: "DELETE" }),
    };
  }

  /* ---------- display logic (pure) ---------- */
  const STATE_TEXT = {
    queued: "Queued",
    loading: "Loading verified inputs and source checkpoint",
    training: "Training ensemble members",
    validating: "Checking saved members",
    calibrating: "Calibrating intervals",
    ready: "Ready",
    done: "Done",
    failed: "Failed",
    cancelled: "Cancelled",
  };
  const DATA_LABEL = {
    generated_research_simulation: "Generated research simulation data",
    customer_simulation: "Customer simulation data",
    other_simulation: "Other simulation data",
  };
  const OUTCOME_LABEL = { kinetic_energy: "Kinetic energy", pressure_rms: "Pressure RMS" };

  function jobText(job) {
    if (!job) return "";
    let text = STATE_TEXT[job.state] || job.state;
    const p = job.progress || {};
    if (job.state === "training" && p.member) {
      text += ": member " + p.member + " of " + p.members;
      if (Number.isFinite(p.step)) text += ", step " + p.step + " of " + p.steps;
    }
    return text;
  }

  /** Fraction of a fit done: members weigh equally; calibration finishes it. Never invents progress. */
  function fitFraction(job) {
    if (!job) return 0;
    if (job.state === "ready") return 1;
    const p = job.progress || {};
    if (job.state === "calibrating" || job.state === "validating") return 0.97;
    if (!p.member || !p.members) return 0;
    const done = p.member_complete ? p.member : p.member - 1 + (Number.isFinite(p.step) && p.steps ? p.step / p.steps : 0);
    return Math.max(0, Math.min(0.96, (done / p.members) * 0.96));
  }

  function intervalText(calibration) {
    if (!calibration) return "No model";
    if (calibration.status === "calibrated")
      return "Calibrated joint " + Math.round(calibration.level * 100) + "% interval (normalized split conformal, " + calibration.n +
        " calibration runs). Marginal over future queries like the calibration runs; not a per-query guarantee.";
    return "Uncalibrated: raw ensemble spread only (" + calibration.n + " calibration runs supplied, 39 needed). Not a confidence interval.";
  }

  /** Requirements still missing before a calibrated fit, from the server's role counts and limits. */
  function missingRequirements(version, capability) {
    const out = [];
    const roles = capability.roles;
    const c = version.role_counts;
    if (c.train < roles.train.min) out.push("Add " + (roles.train.min - c.train) + " more train runs (minimum " + roles.train.min + ").");
    if (c.development < roles.development.min)
      out.push("Add " + (roles.development.min - c.development) + " more development runs (minimum " + roles.development.min + ").");
    if (c.calibration < roles.calibration.min_for_calibrated)
      out.push("Calibrated intervals need " + roles.calibration.min_for_calibrated + " calibration runs; " + c.calibration +
        " supplied. You can still request an explicitly uncalibrated, prediction-only fit.");
    return out;
  }

  function labelsSupplied(c) {
    return c.train + c.development + c.calibration + c.verification;
  }

  /** Rows for the budget panel. Negative differences are shown as added cost; unknowns stay unknown. */
  function budgetRows(sc) {
    if (!sc || sc.status === "not_set") return [];
    const rows = [["Runs you had at the start", String(sc.E_existing_runs)], ["Runs added since", String(sc.A_added_runs)]];
    if (Number.isFinite(sc.baseline_future_runs)) {
      rows.push(["Planned runs still to do", String(sc.baseline_future_runs)]);
      rows.push(["Against your plan", signed(sc.run_difference, "run")]);
    }
    if (sc.money) {
      const m = sc.money;
      rows.push([m.budget_difference >= 0 ? "Budget not yet spent" : "Extra cost", money(Math.abs(m.budget_difference), m.currency) + (m.incomplete ? " (incomplete)" : "")]);
    }
    return rows;
  }
  function signed(v, unit) {
    const n = Math.abs(v);
    const word = n === 1 ? unit : unit + "s";
    return v >= 0 ? n + " fewer " + word + " so far" : n + " more " + word + " than planned";
  }
  function money(v, currency) {
    const s = Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 2 });
    return (v < 0 ? "−" : "") + s + " " + currency;
  }

  const api = { ApiError, createClient, jobText, fitFraction, intervalText, missingRequirements, labelsSupplied, budgetRows, money, DATA_LABEL, OUTCOME_LABEL, STATE_TEXT };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DeltaFlowApi = api;
})(typeof window !== "undefined" ? window : globalThis);
