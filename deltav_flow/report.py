"""Study report, measured usage ledger and the optional customer budget scenario.

Accounting rules (prompt section 10):
  E = completed runs supplied in the first study version (before this workflow).
  B = the customer's planned total runs including E ("total_plan_including_existing"),
      or planned future runs only ("future_runs_only"). Bases are never mixed.
  A = completed runs added after the first version, in every data role.
  baseline future runs = max(B - E, 0) for a total plan, B for a future-only plan.
  scenario budget difference = (baseline future runs - A) * cost per run - known incremental charges.
Negative differences are added cost and are shown, never clamped. Missing prices
stay unknown, never zero. This is a budget comparison, not verified avoided work.
"""
from __future__ import annotations

import csv
import io
import math
import time
from collections import defaultdict

from . import contract as C
from .evidence import EVIDENCE

# Plain names for fit profiles in user-facing text; ids stay in the JSON report.
PROFILE_DISPLAY = {"task8-fno-lhs-v1": "standard FNO fine-tuning recipe (three copies, calibrated intervals)"}

LIMITATIONS = [
    "One generated 2D periodic forced incompressible flow family on a fixed 64x64 grid. Not customer CFD, OpenFOAM, "
    "other geometries, 3D or arbitrary fluids.",
    "Predictions narrow where to simulate next; they are not simulations and do not replace verification in your solver.",
    "Calibrated intervals are marginal over iid future queries like the calibration runs. They are not a per-query "
    "guarantee and are not a percentage chance that an individual prediction is right.",
    "Recommendation locations and repeated calibration-based stopping do not inherit the calibration guarantee.",
    "Verified cost savings are not established for this study. Budget numbers are scenario comparisons against your plan.",
    "Validation on a real customer solver, customer data or OpenFOAM has not been performed.",
]


def _latest_ready(models):
    ready = [m for m in models if m["status"] == "ready"]
    return ready[-1] if ready else None


def usage(view, ledger, queried_settings):
    versions = view["versions"]
    first, last = versions[0], versions[-1]
    added = [rid for v in versions[1:] for rid in v.get("added_runs", [])]
    recommended = []
    for r in view["recommendations"]:
        recommended += [row["candidate_id"] for row in r["rows"]]
    recommended = list(dict.fromkeys(recommended))
    added_set = set(added)
    model = _latest_ready(view["models"])
    runtime = defaultdict(float)
    for row in ledger:
        if row["unit"] == "seconds":
            runtime[row["kind"].replace("worker_", "")] += row["amount"]
    charges = defaultdict(float)
    for row in ledger:
        if row["kind"] == "charge":
            charges[row["unit"]] += row["amount"]
    return {
        "supplied_by_role": last["role_counts"],
        "supplied_total": last["runs"],
        "existing_at_start": {"runs": first["runs"], "by_role": first["role_counts"]},
        "labels_consumed_by_latest_model": (model["role_counts_used"] if model else None),
        "labels_consumed_note": "Unique labelled settings used to fit, select and calibrate the latest model. "
                                "Ensemble members reuse the same runs and are not extra simulations.",
        "new_runs_requested": len(recommended),
        "new_runs_completed": len(added),
        "recommended_runs_completed": len(added_set & set(recommended)),
        "pending_recommendations": [c for c in recommended if c not in added_set],
        "untested_settings_predicted": queried_settings,
        "untested_settings_note": "Distinct settings evaluated by the model, including automatic checks such as trend curves and probe scans. These are predictions, not simulations avoided.",
        "worker_runtime_seconds": dict(runtime),
        "worker_runtime_note": "Measured worker wall time per job type. Its price is unknown unless you record the charge.",
        "known_charges": dict(charges),
        "supplied_solver_seconds": last.get("solver_seconds_supplied"),
        "runs_with_solver_seconds": last.get("runs_with_solver_seconds"),
        "solver_seconds_note": "Serial solver seconds from your runs.csv; not elapsed wall-clock time.",
    }


def scenario(inputs, measured):
    if not inputs:
        return {"status": "not_set", "message": "Add your planned solver runs and cost per run to estimate budget impact."}
    E = measured["existing_at_start"]["runs"]
    A = measured["new_runs_completed"]
    pending = len(measured["pending_recommendations"])
    B = inputs.get("planned_runs")
    out = {"status": "scenario", "label": "Budget comparison against your stated plan (scenario, not verified savings)",
           "inputs": inputs, "E_existing_runs": E, "A_added_runs": A, "pending_recommended_runs": pending,
           "basis": inputs["basis"]}
    if B is None:
        out.update(status="incomplete", message="Add your planned solver runs to compare against your plan.")
        return out
    baseline_future = max(B - E, 0) if inputs["basis"] == "total_plan_including_existing" else B
    out["baseline_future_runs"] = baseline_future
    out["run_difference"] = baseline_future - A
    out["remaining_plan_forecast"] = {"runs_if_pending_completed": A + pending,
                                      "difference_if_pending_completed": baseline_future - A - pending}
    out["formulas"] = {
        "baseline_future_runs": "max(B - E, 0)" if inputs["basis"] == "total_plan_including_existing" else "B (future runs only)",
        "budget_difference": "(baseline_future_runs - A) * cost_per_run - known incremental charges",
    }
    cost, currency = inputs.get("cost_per_run"), inputs.get("currency")
    if cost is None or not currency:
        out.update(status="incomplete", message="Add your cost per run and currency to estimate budget impact. "
                                                "Run counts above are shown without a currency amount.")
    else:
        charges = measured["known_charges"]
        other = sorted(c for c in charges if c != currency)
        charge = charges.get(currency)
        difference = (baseline_future - A) * cost - (charge or 0.0)
        out["money"] = {"currency": currency, "cost_per_run": cost, "baseline_future_spend": baseline_future * cost,
                        "actual_new_run_spend": A * cost, "known_incremental_charges": charge,
                        "budget_difference": difference,
                        "direction": "unspent budget versus your plan" if difference >= 0 else "added cost versus your plan"}
        if other:
            out["money"]["incomplete"] = f"Charges recorded in {', '.join(other)} are not converted and are excluded."
        if not charges:
            out["money"]["incomplete"] = ("No service or compute charges recorded. Worker runtime is measured in seconds; "
                                          "its price is unknown and is not counted as zero.")
    hours = inputs.get("wall_hours_per_run")
    if hours is not None:
        out["time"] = {"serial_solver_hours_added": A * hours,
                       "serial_solver_hours_baseline_future": baseline_future * hours,
                       "note": "Serial solver-hours, not elapsed time."}
        if inputs.get("concurrency"):
            k = inputs["concurrency"]
            out["time"]["calendar_estimate_hours"] = {"added": math.ceil(A / k) * hours,
                                                      "baseline_future": math.ceil(baseline_future / k) * hours,
                                                      "assumption": f"{k} runs at a time, {hours} h each, no queueing"}
    return out


def build(view, ledger, scenario_inputs, queried_settings=0):
    measured = usage(view, ledger, queried_settings)
    model = _latest_ready(view["models"])
    last = view["versions"][-1]
    data_kind = view["data_kind"]
    return {
        "generated_at": time.time(),
        "study": {"study_id": view["study_id"], "name": view["name"], "data_kind": data_kind,
                  "data_label": {"generated_research_simulation": "Generated research simulation data",
                                 "customer_simulation": "Customer simulation data",
                                 "other_simulation": "Other simulation data"}[data_kind],
                  "example_id": view["example_id"], "capability": C.CAPABILITY_ID,
                  "versions": [{"number": v["number"], "version_id": v["version_id"], "origin": v["origin"], "runs": v["runs"],
                                "role_counts": v["role_counts"], "content_sha256": v["content_sha256"]} for v in view["versions"]],
                  "latest_version": last["number"]},
        "model": None if not model else {
            "model_id": model["model_id"], "version_id": model["version_id"], "profile": model["profile"],
            "profile_label": model["profile_label"], "measurement": "measured now on this study",
            "fits_latest_version": model["version_id"] == last["version_id"],
            "source_sha256": model["source_sha256"], "recipe_id": model["recipe"]["id"],
            "members": [{"seed": m["seed"], "weights_sha256": m["weights_sha256"], "best_step": m["best_step"],
                         "seconds": m["seconds"], "reused_after_interruption": m["reused_after_interruption"]} for m in model["members"]],
            "environment": model["environment"], "timing": model["timing"], "calibration": {
                k: v for k, v in model["calibration"].items() if k != "rule"},
            "readiness": model["readiness"], "verification": model["verification"],
            "readiness_wording": ("The model's interval-width criterion is met; further simulation may still be needed for "
                                  "independent verification.") if (model["readiness"] or {}).get("criterion_met") else
                                 ("The interval-width criterion is not met on the label-free probes." if model["readiness"] else
                                  "No calibrated intervals, so the interval-width criterion is not evaluated.")},
        "recommendations": view["recommendations"][-1] if view["recommendations"] else None,
        "usage": measured,
        "scenario": scenario(scenario_inputs, measured),
        "verified_cost_savings": "Verified cost savings: not yet established for this study.",
        "research_evidence": EVIDENCE,
        "limitations": LIMITATIONS,
    }


def _fmt(v, digits=4):
    if v is None:
        return "unknown"
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def markdown(report):
    s = report["study"]
    lines = [f"# Delta-V study report: {s['name']}", "",
             f"Data: **{s['data_label']}**. Capability `{s['capability']}`. Version {s['latest_version']}.", ""]
    m = report["model"]
    lines += ["## Model", ""]
    if not m:
        lines += ["No ready adapted model for this study yet.", ""]
    else:
        lines += [f"- Recipe: {PROFILE_DISPLAY.get(m['profile'], m['profile_label'])}, {m['measurement']}.",
                  f"- Source checkpoint SHA-256 `{m['source_sha256']}`; members " + ", ".join(
                      f"seed {x['seed']} (best step {x['best_step']}, weights `{x['weights_sha256'][:12]}…`)" for x in m["members"]) + ".",
                  f"- Hardware: {m['environment']['device']} ({m['environment']['device_name']}); total {m['timing']['total_seconds']:.1f} s.",
                  f"- Calibration: {m['calibration']['status']} (n={m['calibration']['n']}).",
                  f"- {m['readiness_wording']}"]
        v = m["verification"]
        if v:
            lines += [f"- Verification on {v['n']} held-out runs of this study: joint accuracy within tolerance "
                      f"{v['joint_accuracy_success']:.3f}; RMSE kinetic energy {v['rmse']['kinetic_energy']:.4g}, "
                      f"pressure RMS {v['rmse']['pressure_rms']:.4g}"
                      + (f"; joint interval coverage {v['joint_coverage']:.3f}." if "joint_coverage" in v else ".")]
        lines.append("")
    u = report["usage"]
    lines += ["## Measured usage", "",
              f"- Completed runs supplied (latest version): {u['supplied_total']} " + str(u["supplied_by_role"]),
              f"- Labels consumed by the latest model: {u['labels_consumed_by_latest_model']}",
              f"- New runs requested / completed during this workflow: {u['new_runs_requested']} / {u['new_runs_completed']}",
              f"- Untested settings predicted (predictions, not simulations eliminated): {u['untested_settings_predicted']}",
              f"- Worker runtime (seconds): {u['worker_runtime_seconds'] or 'none recorded'}",
              f"- Known charges: {u['known_charges'] or 'none recorded (unknown, not zero)'}", ""]
    sc = report["scenario"]
    lines += ["## Budget scenario", ""]
    if sc["status"] == "not_set":
        lines += [sc["message"], ""]
    else:
        lines += [f"{sc['label']}.", "", f"- Basis: {sc['basis']}; E={sc['E_existing_runs']}, A={sc['A_added_runs']}, "
                  f"B={sc['inputs'].get('planned_runs')}"]
        if "baseline_future_runs" in sc:
            lines += [f"- Baseline future runs: {sc['baseline_future_runs']} ({sc['formulas']['baseline_future_runs']})"]
        if "money" in sc:
            mo = sc["money"]
            lines += [f"- Budget difference: {mo['budget_difference']:.2f} {mo['currency']} ({mo['direction']}); "
                      f"formula {sc['formulas']['budget_difference']}"]
            if mo.get("incomplete"):
                lines.append(f"- Incomplete: {mo['incomplete']}")
        elif sc.get("message"):
            lines.append(f"- {sc['message']}")
        lines.append("")
    lines += [f"**{report['verified_cost_savings']}**", "", "## Limitations", ""] + [f"- {x}" for x in report["limitations"]]
    lines += ["", "## Research evidence (historical, separate from this study)", ""]
    for card in report["research_evidence"]["cards"]:
        lines += [f"- {card['claim']} _{card['qualification']}_ (source: {card['source']})"]
    return "\n".join(lines) + "\n"


def recommendation_csv(rec):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["candidate_id", *C.CONTROL_NAMES, "policy", "seed", "reason", "predicted_kinetic_energy",
                     "predicted_pressure_rms", "interval_status", "kinetic_energy_lower", "kinetic_energy_upper",
                     "pressure_rms_lower", "pressure_rms_upper", "label", *C.OUTCOME_NAMES])
    for row in rec["rows"]:
        interval = row.get("calibrated_interval")
        writer.writerow([row["candidate_id"], *[repr(float(v)) for v in row["controls"]], rec["policy"], rec.get("seed"),
                         row["reason"], *[f"{v:.6g}" for v in row["predicted"]],
                         "calibrated joint 90% (marginal over iid queries)" if interval else "uncalibrated: raw ensemble spread only",
                         *(sum(([f"{interval[0][j]:.6g}", f"{interval[1][j]:.6g}"] for j in range(2)), []) if interval else ["", "", "", ""]),
                         "experimental recommendation; no guaranteed savings", "", ""])
    return stream.getvalue()
