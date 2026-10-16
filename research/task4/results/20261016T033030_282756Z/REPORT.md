# Delta-V task 4 — measured pilot report

Protocol SHA-256: `0e2c6e3829b0cbdcbf9943f475503fc9882b8310af1bb8a7f6d71b395c3cc84b`
HF revision: `e25651ff92cf1d8f8818164a0277b8613e83d8b3`

Completed 936 model fits and 9954 held-out scalar predictions in 5.2 seconds.
All 140 correctness checks passed.

## Data and scope

Each row is an aggregated parameter setting, not an individual solver run. Systems were fitted separately.
The mean and linear models are fair references for the GP; no pretrained neural model was tested.

- turbulent_radiative_layer_2D: 9 distinct settings, 1 inputs, two outcomes.
- rayleigh_benard: 35 distinct settings, 2 inputs, two outcomes.
- shear_flow: 28 distinct settings, 2 inputs, two outcomes.

## Predeclared worked example

Cooling-time system, 3 observed settings, 2 held-out settings, interpolation seed 42.
- mass_flux: RMSE 0.0171875 stored units; NRMSE 0.3287; 95% coverage 100.0%.
- turbulent_velocity: RMSE 0.00862206 stored units; NRMSE 0.2429; 95% coverage 100.0%.

## Paired comparisons

Counts below are descriptive field/budget/split comparisons with overlapping data, not independent statistical trials.
- rayleigh_benard / boundary / mass_flux: GP lower RMSE than mean in 89/96; than linear in 70/96.
- rayleigh_benard / boundary / turbulent_velocity: GP lower RMSE than mean in 91/96; than linear in 67/96.
- rayleigh_benard / interpolation / mass_flux: GP lower RMSE than mean in 21/21; than linear in 19/21.
- rayleigh_benard / interpolation / turbulent_velocity: GP lower RMSE than mean in 21/21; than linear in 18/21.
- rayleigh_benard / mixed / mass_flux: GP lower RMSE than mean in 23/24; than linear in 16/24.
- rayleigh_benard / mixed / turbulent_velocity: GP lower RMSE than mean in 23/24; than linear in 16/24.
- shear_flow / boundary / mass_flux: GP lower RMSE than mean in 27/84; than linear in 68/84.
- shear_flow / boundary / turbulent_velocity: GP lower RMSE than mean in 24/84; than linear in 74/84.
- shear_flow / interpolation / mass_flux: GP lower RMSE than mean in 13/18; than linear in 13/18.
- shear_flow / interpolation / turbulent_velocity: GP lower RMSE than mean in 7/18; than linear in 5/18.
- shear_flow / mixed / mass_flux: GP lower RMSE than mean in 8/21; than linear in 11/21.
- shear_flow / mixed / turbulent_velocity: GP lower RMSE than mean in 12/21; than linear in 17/21.
- turbulent_radiative_layer_2D / boundary / mass_flux: GP lower RMSE than mean in 22/24; than linear in 5/24.
- turbulent_radiative_layer_2D / boundary / turbulent_velocity: GP lower RMSE than mean in 24/24; than linear in 19/24.
- turbulent_radiative_layer_2D / interpolation / mass_flux: GP lower RMSE than mean in 4/12; than linear in 8/12.
- turbulent_radiative_layer_2D / interpolation / turbulent_velocity: GP lower RMSE than mean in 10/12; than linear in 10/12.
- turbulent_radiative_layer_2D / mixed / mass_flux: GP lower RMSE than mean in 9/12; than linear in 4/12.
- turbulent_radiative_layer_2D / mixed / turbulent_velocity: GP lower RMSE than mean in 12/12; than linear in 7/12.

## Uncertainty

Inspect coverage together with interval width, by system and outcome. No calibrated uncertainty guarantee is made.
171 fits produced optimizer warnings; all are recorded in fit_details.json.
GP intervals include fitted residual noise for aggregate table outcomes; raw-run variability and hyperparameter uncertainty are not fully represented.

## Source expansion audit

- turbulent_radiative_layer_2D: 9 distinct filename settings, 0 not in the table. No extra settings found; raw fields could improve repeat/aggregation audit.
- rayleigh_benard: 35 distinct filename settings, 0 not in the table. No extra settings found; raw fields could improve repeat/aggregation audit.
- shear_flow: 28 distinct filename settings, 0 not in the table. No extra settings found; raw fields could improve repeat/aggregation audit.

## What this completes

Versioned table exports, integrity/schema audit, grouped setting splits, GP and simple baselines, scarce-data curves, boundary tests, coverage/width measurements, a measured example, and reproducible artifacts.

## Evidence still needed

- Confirm the original extraction lineage, window counts, physical units, field mapping and aggregation variability from raw sources.
- Validate a real customer tolerance on a suitable engineering problem. No default acceptance threshold was invented.
- Compare a later Delta-V learned predictor against this GP before attributing an advantage to transfer learning.
- Test an end-to-end run-selection workflow and count all solver/validation costs before claiming savings.

## Reproduction

Run the notebook from a fresh kernel with its recorded controls. Tables and source inventories are embedded and hash-verified.
Exact split membership and all model settings are in protocol.json. Version differences and timing variation may affect numerical reproduction.
NRMSE uses the fixed non-test pool range for retrospective scoring only. No test or unacquired outcome labels enter fitting or training-order selection.
Test outcomes have now been inspected: changing the protocol creates further exploratory analysis, not a new untouched test set.
