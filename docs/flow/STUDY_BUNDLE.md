# Study bundle, schema `deltav-study-bundle/1`

A study bundle is one ZIP that the API validates on the server before anything
is stored. It describes completed simulations from the single supported
capability, `deltav-fluid-periodic2d-v1`:

- one generated 2D periodic forced incompressible flow family on the fixed
  periodic square `[0, 2π]²`, sampled on a 64×64 grid;
- controls `viscosity` ∈ [0.02, 0.08], `forcing` ∈ [0.2, 1.2],
  `initial_mix` ∈ [0.2, 0.8] (nondimensional);
- result at nondimensional time 2 from the analytic initial condition at time 0;
- outcomes read from the result field:
  `kinetic_energy = 0.5 · mean(u² + v²)` and
  `pressure_rms = sqrt(mean((p − mean p)²))`, both nondimensional.

It is not a general fluid uploader. Other grids, geometries, boundary
conditions, times, 3D data, meshes, customer CFD or OpenFOAM output are rejected
with the reason and the change required. Nothing is resampled or renamed into
compatibility.

`GET /flow/capabilities` returns the same contract as JSON.

## Files

| Path | Required | Content |
|---|---|---|
| `study.json` | yes | metadata below |
| `runs.csv` | yes | one row per completed run |
| `fields/*.npy` | yes | one initial and one result array per run |
| `checksums.json` | yes | `{"algorithm": "sha256", "files": {path: {"sha256", "bytes"}}}` for every other file |
| `candidates.csv` | no | feasible next-run settings: `candidate_id, viscosity, forcing, initial_mix` only |
| `probes.csv` | no | label-free readiness probe settings: `probe_id, viscosity, forcing, initial_mix` only |
| `README.md` | no | free text |

One wrapping folder is accepted. `__MACOSX/` and `.DS_Store` entries are
ignored with a warning. Any other file is an error: no Python, pickles, model
checkpoints or extra arrays are ever read.

## `study.json`

`deltav_flow.bundle.study_meta()` writes a compliant file. Required keys:

```json
{
  "schema": "deltav-study-bundle/1",
  "study_name": "My periodic flow study",
  "lane": "fluid",
  "capability": "deltav-fluid-periodic2d-v1",
  "grid": [64, 64],
  "geometry": "fixed periodic square [0,2pi]^2",
  "boundary_conditions": ["periodic", "periodic"],
  "units": "nondimensional",
  "time": {"initial": 0.0, "result": 2.0},
  "fields": {"channels": ["density", "pressure", "u", "v"], "layout": "channel,y,x",
             "coordinates": "x_j = 2*pi*j/64, y_i = 2*pi*i/64; field[c, i, j]",
             "dtype": "float32", "format": "npy"},
  "initial_condition": {"representation": "supplied_fields", "generator": "deltav-periodic-initial-v1"},
  "controls": [{"name": "viscosity", "range": [0.02, 0.08], "unit": "nondimensional"}, "..."],
  "outcomes": [{"name": "kinetic_energy", "unit": "nondimensional"}, {"name": "pressure_rms", "unit": "nondimensional"}],
  "provenance": {"data_kind": "customer_simulation", "description": "..."}
}
```

Optional: `seed` (integer; member seeds are `(1103|2207|3301 + seed·10007) mod 2³¹`),
`tolerance` (two absolute nondimensional outcome tolerances, default 0.015 each),
`roles` (informational), `acquisition` (`{"policy": "mapped-lhs-v1", "seed", "sequence_length"}`,
used by the worked example to reproduce the recorded Task 8 order; requires
`candidates.csv`).

Every compatibility key is compared exactly. Declared control ranges may be
narrower than the supported box, never wider. `provenance.data_kind` is one of
`generated_research_simulation`, `customer_simulation`, `other_simulation` and
is shown next to every result.

## `runs.csv`

Required columns: `run_id, viscosity, forcing, initial_mix, role, initial_file, result_file`.
Optional: `kinetic_energy, pressure_rms` (checked against the field-derived
values; relative tolerance 1e-4), `solver_seconds`, `solver_cost` and
`cost_currency` (three-letter code, required with a cost). Unknown columns are
ignored with a warning.

Roles:

| Role | Count | Use |
|---|---|---|
| `train` | 8 to 64 | adaptation |
| `development` | 12 to 64 | checkpoint selection only |
| `calibration` | 0 to 400; 39 or more for calibrated intervals | joint split-conformal calibration only |
| `verification` | 0 to 400 | benchmark evaluation after the fit; never used to fit, select, normalize or calibrate |

With fewer than 39 calibration runs the bundle is accepted with a warning. A
fit then requires an explicit opt-in to prediction-only output whose spread is
an uncalibrated ensemble standard deviation.

Run IDs are unique. Two runs with identical controls are rejected, whether in
the same role or across roles, because repeated settings must never cross data
roles. Every field file is referenced exactly once.

## Field arrays

Each `.npy` is little-endian float32, C order, shape `(4, 64, 64)`, channel order
density, pressure, u, v, layout `[c, y, x]` with `x_j = 2πj/64`, `y_i = 2πi/64`.
Version 1.0/2.0 headers are parsed with bounded size before reading and
`allow_pickle=False`. Arrays must be finite.

- The initial array must equal the analytic initial condition
  `deltav-periodic-initial-v1` at the run's controls (max abs difference 1e-5).
  Query-time inputs are generated the same way; no future field is ever needed.
- The result density channel must be 1 (incompressible family).

## Archive limits

64 MiB ZIP, 320 MiB unpacked, 2,200 entries, 1,000 runs, 4,096 candidates,
1,024 probes, one field per `.npy` (65,664 bytes), 4 MiB per text file and at
most 50× expansion for text members. Absolute, `..`, `.`, backslash and drive
paths, symlinks, encrypted members, duplicate names (also case-folded) and
non-deflate compression are rejected. Content is read with a hard per-member
byte bound.

## Addition bundles (`deltav-study-addition/1`)

Completed recommended runs are added with `addition.json`
(`{"schema": "deltav-study-addition/1", "capability": ..., "provenance": {...}}`),
`runs.csv`, `fields/` and `checksums.json`. The same run and field checks apply,
and new runs may not duplicate existing settings or IDs. The result is a new
immutable study version; earlier role assignments are preserved.

## Building bundles

```python
from deltav_flow import bundle as B
meta = B.study_meta("My study", "customer_simulation", "Solver X, mesh 64x64 periodic")
data = B.pack("study", meta, runs, initial, result, candidates=None)
B.validate_bundle(data)   # raises BundleRejected with actionable issues
```

`pack` is deterministic (sorted names, fixed timestamps). The worked example is
built from the restored Task 8 archive:

```bash
python3 research/evidence_archives/restore_archive.py task8 --output work/evidence/task8-original.zip
# extract with path checks to work/evidence/task8 (see docs/TASK10_11_PROGRESS.md)
.venv/bin/python scripts/pack_task8_example.py --manifest deltav_flow/examples/task8_r701.json
```

Its SHA-256 values are pinned in `deltav_flow/examples/task8_r701.json`.
