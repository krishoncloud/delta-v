# Delta-V v2: supported lane, new customer problem

Clarified with Krish on 16 October 2026. This records the original 14 release tasks plus Task 15, added by Krish
before the next feature launch, and makes the new-customer-problem test explicit.
It does not change historical experiment results or claim deployed functionality.

## Product contract

The customer chooses a supported domain lane, then supplies results from their
own problem within that lane. Lane 1 is fluid dynamics. The workbench is not
restricted to selecting one of Delta-V's demonstration datasets. A fluid lane
still needs an explicit capability description: dimensions, fields, geometry/grid,
boundaries, units, input/output representations, and supported query types.
Current demonstrated field support is bounded 2D fluid prediction; accepting a
fluid label must not imply support for every 2D/3D fluid solver or geometry.

The intended workflow is:

1. Choose the fluid lane and describe the study, input knobs, allowed ranges,
   outcome definitions/units, engineering goal and acceptable error.
2. Upload completed runs and map columns/files to the supported input contract.
3. Check compatibility, validate the data and reserve independent evaluation data.
4. Adapt the pretrained model to the customer's problem and evaluate predictions.
5. Predict outcomes for untested input settings, with evaluated uncertainty and
   explicit limits on interpolation and extrapolation.
6. Recommend useful next simulations. The customer executes their solver and
   returns results; measure whether decisions require less total simulation work.

The differentiating hypothesis is that prior fluid learning makes adaptation to
an unseen fluid problem more data-efficient or less costly than a fair GP and
simple baselines. This is a hypothesis to test, not a property assumed from the
FNO architecture. The product may retain GP as a baseline or fallback, but must
identify the actual method and must not label GP-only results as an FNO win.

## Status and revised acceptance scope of the 15 tasks

| # | Task | Status and acceptance scope |
|---|---|---|
| 1 | Lock experiment/data protocol | Completed for Euler Track A. Lock a separate protocol before the new product-transfer evaluation. |
| 2 | Prepare four-gamma Euler data | Completed for Track A; not automatically the dataset for the product-transfer test. |
| 3 | Euler scratch/full-fine-tune/adapter sweep | Completed: 54 jobs. Field-transfer evidence; not a knobs-to-outcome product result. |
| 4 | Scalar-outcome dataset and GP baseline | Research deliverable complete: three outcome tables, audits, matched mean/linear/GP comparisons, held-out predictions and measured example. Original aggregation provenance gaps remain recorded. The new MoLab run's HF save was not verified in the latest check; its downloaded ZIP integrity was verified. |
| 5 | Parameter-conditioned outcome prediction | Self-contained [MoLab development notebook](molab_task5.py) implemented: study contract, actual pretrained FNO conditioning, field-derived outcomes, matched scalar references and saved inference bundle. A controlled analytical study checks the mechanism; successful development gates and a harder Task 6 evaluation remain necessary. Not integrated into the workbench. |
| 6 | New-fluid-problem transfer benchmark; adapter versus full fine-tuning | Pending. Main scientific acceptance test for this release: does fluid pretraining help the actual customer prediction task, compared with scratch, GP, linear and mean? Detailed gates below. |
| 7 | Ensemble uncertainty | Pilot neural calibration was weak. Evaluate coverage and interval width on the new target problem; do not equate predicted SD with a guarantee. |
| 8 | Sobol/LHS and next-run selection | Pending. Compare complete acquisition workflows at equal engineering targets and count all required target simulations. Retrospective Task 4 budgets are not a validated stopping policy. |
| 9 | One OpenFOAM workflow end to end | Pending. Exercise a separate customer-style fluid study through export, compatibility, adaptation, query and validation. Preserve solver/mesh/BC/unit metadata and real runtime evidence. |
| 10 | Upload and validation API | Pending. Add lane/capability and study metadata, explicit scalar/field schema, supported-query validation, and real fit/predict/next contracts. Reject incompatible requests with actionable reasons. |
| 11 | Bundled worked example | A measured Task 4 research example exists. Product completion requires an example connected to the real backend on the accepted input contract, identifying the method actually used. |
| 12 | Bring-your-runs website | Built as a mock on `frontend/bring-your-runs`; integration pending. Represent lane selection and a new customer study, compatibility and any required field uploads. Keep unsupported lanes unavailable. |
| 13 | Model demo tab | Built on the frontend branch; integration pending. Keep recorded-field demonstrations distinguishable from the customer study workflow. |
| 14 | Test/deploy/publish evidence | Pending. Deployment requires Task 15 completion. Release only within demonstrated capabilities. Pretrained-adaptation and run-savings claims must pass the gates below; mocks and Task 4 GP results cannot substitute. |
| 15 | Separate public research from private product development | Pending; implementation paused at Krish's request. Mandatory before the next feature launch and before Task 14 deployment. Verify authorized development access, protect future product code/artifacts and customer data, and preserve the public demo. |

## Task 5: bridge customer inputs to the pretrained FNO

The existing FNO consumes four spatial frames and predicts the next field frame.
The frontend currently offers numeric input/outcome CSVs. Connecting that CSV
endpoint to the existing checkpoint is not sufficient to produce a parameter-
conditioned predictor for a new design.

Task 5 must deliver:

- A documented study schema: run ID, parameters, outcomes and units, ranges,
  initial/boundary conditions, geometry/grid, plus field trajectories when required.
  State exactly what is mandatory for the learned method and what a CSV-only
  baseline can support. Users name their own knobs; adapters map them to a
  documented supported representation, rather than assuming arbitrary names have
  pretrained semantic meaning.
- A declared prediction contract. For new-setting interpolation, list the inputs
  available at query time without obtaining the hidden outcome from the solver.
  If querying needs four true frames from that new run, identify it as continuation
  requiring a partial simulation and measure shortening separately from skipping
  a complete run. Never supply hidden future frames or hidden outcome labels.
- A real, trainable connection from the pretrained FNO to the required target
  outcomes, with parameter/condition encoding and outcome extraction/readout as
  needed. Save adapted weights, preprocessing, mappings and training provenance.
  An outcome MLP pretrained only on scalar tables is not the pretrained FNO.
- Validation and honest failure behavior for unsupported representations. A new
  problem may require an adapter; an incompatible study is not silently converted
  into an asserted neural-transfer success.

Do not select a particular new dataset or claim architecture compatibility until
its inputs, field definitions, training exposure and bounded data access are audited.

## Task 6: main release experiment

Question: on a new customer-style fluid problem, does our prior fluid learning
improve predictions or reduce the target simulation work needed for a fixed
engineering goal, compared with credible alternatives?

1. **New target problem.** Choose at least one fluid system or engineering setup
   excluded from source pretraining, not merely a withheld parameter value of a
   source problem. Record the checkpoint and exposure evidence; retain any
   unresolved provenance limitation. Existing Euler field-transfer scores alone
   cannot satisfy this product-flow test. Reused, previously inspected test data
   are exploratory; reserve a fresh untouched confirmatory evaluation.
2. **Actual customer task.** Predict the declared outcomes at unseen design
   settings with the Task 5 query contract. Training and testing four-to-fifth-frame
   forecasting alone does not establish knobs-to-outcome interpolation.
3. **Arms.** Compare mean, log-linear/appropriate declared linear model, fair GP,
   the same target neural architecture trained from scratch, full fine-tuning
   from the source FNO, and a frozen-backbone adapter. If an arm cannot consume
   the supported information, explain the mapping or limitation; do not hide it.
4. **Fair data and information.** Use identical nested target setting budgets,
   held-out settings and target observations for matched comparisons. Where FNO
   consumes fields while GP consumes summaries, report that extra information and
   its acquisition/export costs. Include an information-matched comparison where
   feasible, and avoid claiming the advantage comes from pretraining without the
   same-input scratch comparison.
5. **Leakage control.** Group repeated runs at the same design setting before
   setting-level holdout; keep whole trajectories together, including all windows
   and field-derived summaries. Fit preprocessing on training data only. Declare
   validation/model-selection data separately and count their simulation costs.
   Never choose architectures, tolerances or acquisition rules using test scores.
6. **Locked metrics and decision rule.** Predeclare engineering tolerances,
   primary outcomes, budgets, seeds, model-selection/tuning budgets, meaningful
   advantage criteria and confirmatory test before fitting. Report per-outcome
   errors, interpolation and boundary behavior, spread, failures and warnings.
   Task 4's scoring-only pool-range NRMSE is not a customer acceptance criterion.
7. **Measure the useful advantage.** Test lower target error at equal supplied
   data, fewer target settings to reach the same acceptable error, or lower total
   time/cost at that target. Report preprocessing, adaptation, inference and all
   target training/validation/verification simulation costs. Disclose reusable
   source-pretraining cost separately. A fast FNO forward pass alone is not proof
   of a cheaper customer workflow.
8. **Preserve evidence.** Save data/schema hashes, exact splits, checkpoint
   identities, configurations, adapted models, predictions, timings, warnings,
   uncertainty results and reports. Publish negative results alongside wins.

Task 6 completes when the locked comparison has been executed and audited,
regardless of which model wins. A successful pretrained-advantage claim requires
the predeclared advantage on the untouched target evaluation; a GP win is not
relabelled as success for the neural arm. Supported scope must match the tested
problem class; one target is not proof for all fluid problems.

## Task 14 release gates

Task 15 is a prerequisite for deployment, despite its later task number.

- Real lane -> study -> compatibility -> upload -> adaptation -> held-out
  validation -> prediction -> next-run -> report behavior, without mock compute.
- The input contract is achievable for a customer at prediction time. Additional
  field requirements and partial-simulation requirements are visible before fit.
- At least one measured new-fluid-problem adaptation example; unsupported cases
  are handled explicitly. Only publish a pretrained superiority claim if Task 6's
  predeclared comparison supports it against the applicable strong baseline.
- Publish a solver-run-savings claim only after Task 8/9 measures a complete
  workflow at an engineering target against GP-driven and fixed-design controls,
  accounting for validation and follow-up runs. Task 6 retrospective curves can
  support data-efficiency findings, but cannot alone establish prospective savings.
- Uncertainty is evaluated, result artifacts are durably verified, and published
  evidence identifies the method, data, scope and costs actually measured.

A functional GP-backed workflow may be useful, but releasing it does not satisfy
this release's intended demonstration of a pretrained FNO advantage.


## Task 15: public/private separation before the next feature launch

Status: pending, deliberately deferred. Krish added this task on 16 October 2026
and requires completion before the next feature launch. Adding the task does not
resume the paused migration or authorize an automatic visibility change.

Acceptance criteria:

1. Decide and document the publication boundary: selected research reports,
   reproducible benchmarks, demo notebooks/sample data and the public demo stay
   intentionally public; future production adaptation/orchestration code,
   commercial checkpoints, customer uploads, customer-adapted models and workflows
   use private development/storage locations.
2. Verify authorized GitHub/HF access for the owner and development tools before
   moving work. Private visibility must not strand development or require secrets
   to be pasted into source. Do not assume existing connector access covers a new
   private repository.
3. Create and verify the required private repositories/storage, with explicit
   destinations for new product artifacts. Audit export and upload paths so
   customer data and private models are not included in public research uploads.
4. Preserve the public demo: audit its artifact dependencies and verify authorized
   runtime access before changing any repository it uses. Test the running demo
   after the separation. Do not silently break unauthenticated artifact loading.
5. Record what was changed and confirm visibility/access at each destination.
   Existing public copies cannot be recalled; do not present this task as making
   already published code or weights secret retrospectively.

At the time of deferral, no private repository creation was submitted, no
visibility was changed, and no weights or customer data were migrated. Resume
implementation only when Krish explicitly revisits it. Task 14 deployment remains
blocked until these acceptance criteria are satisfied.

## Frontend inspection and integration notes

Inspected `frontend/bring-your-runs` at
`d9fc8ae57baf2956823b7999115026386340ca61` on 16 October 2026:

- `static/index.html` already describes adapting a fluid-pretrained model to the
  customer's runs, scopes the prototype to 2D fluids, and labels mock workbench
  and example behavior. The broad customer-owned-study direction matches.
- `static/workbench.js` has Describe/Upload/Fit/Predict/Next runs/Report stages.
  Its live-model wording is an intended contract; computation is currently mock.
- `static/flow.js` sets `FLOW_MODE = "mock"`, accepts numeric CSVs, and uses a
  browser GP stand-in. The listed fit contract has no explicit lane, field-data,
  geometry/BC or FNO compatibility payload. Those must be reconciled in Tasks
  5/10/12, rather than assuming the existing CSV automatically powers the FNO.
- No implemented lane selector was established by this inspection. The initial
  interface may explicitly fix Fluid as the sole available lane; adding more
  lanes is future work. Do not imply that other physics is already supported.

This document records the integration requirements; it does not modify, merge or
mark the frontend as deployed.
