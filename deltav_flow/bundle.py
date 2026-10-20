"""Study bundle ZIP: safe reading, full validation and deterministic packing.

Bundle layout (schema deltav-study-bundle/1, documented in docs/flow/STUDY_BUNDLE.md):

    study.json        capability, grid/domain/BC/time/units, controls, outcomes, roles
    runs.csv          one row per completed run: id, controls, role, field files
    fields/*.npy      float32 [4, 64, 64] initial and result arrays, channel order
                      density, pressure, u, v; layout [c, y, x]
    candidates.csv    optional feasible next-run settings (controls only)
    probes.csv        optional label-free readiness probe settings (controls only)
    checksums.json    sha256/bytes of every other file

An addition bundle (schema deltav-study-addition/1) has addition.json, runs.csv,
fields/ and checksums.json, and adds completed runs to an existing study.

Nothing here executes uploaded code: arrays are parsed with allow_pickle=False
after bounded header checks, and no path from the archive touches the filesystem.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import stat
import zipfile
from dataclasses import dataclass, field

import numpy as np

from . import contract as C
from .physics import initial_fields, physical_outcomes

ADDITION_SCHEMA = "deltav-study-addition/1"
FIELD_SHAPE = (len(C.CHANNELS),) + tuple(C.GRID)
NPY_BYTES = 128 + int(np.prod(FIELD_SHAPE)) * 4
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
FIELD_PATH = re.compile(r"^fields/[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.npy$")
REQUIRED_RUN_COLUMNS = ["run_id", *C.CONTROL_NAMES, "role", "initial_file", "result_file"]
OPTIONAL_RUN_COLUMNS = [*C.OUTCOME_NAMES, "solver_seconds", "solver_cost", "cost_currency"]
POINT_COLUMNS = {"candidates.csv": "candidate_id", "probes.csv": "probe_id"}
DATA_KINDS = ["generated_research_simulation", "customer_simulation", "other_simulation"]
ACQUISITION_POLICIES = ["mapped-lhs-v1"]
IGNORED = re.compile(r"(^|/)(__MACOSX/.*|\.DS_Store)$")
# Analytic initial fields are reproduced in float32; anything larger is a different problem.
INITIAL_ATOL = 1e-5
OUTCOME_RTOL, OUTCOME_ATOL = 1e-4, 1e-6


def issue(level, code, message, fix=None, **where):
    entry = {"level": level, "code": code, "message": message}
    if fix:
        entry["fix"] = fix
    if where:
        entry["where"] = where
    return entry


class BundleRejected(ValueError):
    """Raised with every actionable issue found; nothing from the bundle is persisted."""

    def __init__(self, issues):
        self.issues = issues
        errors = [i for i in issues if i["level"] == "error"]
        super().__init__(f"{len(errors)} compatibility error(s): " + "; ".join(i["message"] for i in errors[:3]))


@dataclass
class Points:
    ids: list[str]
    x: np.ndarray


@dataclass
class ValidatedStudy:
    kind: str                         # "study" or "addition"
    meta: dict                        # study.json or addition.json, as supplied
    runs: list[dict]                  # parsed rows (controls as floats, optional costs)
    initial: np.ndarray               # [N, 4, 64, 64] float32
    result: np.ndarray                # [N, 4, 64, 64] float32
    outcomes: np.ndarray              # [N, 2] float32, derived from result fields
    candidates: Points | None
    probes: Points | None
    files: dict                       # path -> {sha256, bytes}
    content_sha256: str               # hash of the file inventory, independent of ZIP packaging
    archive_sha256: str
    warnings: list = field(default_factory=list)

    @property
    def ids(self):
        return [r["run_id"] for r in self.runs]

    def x(self, rows=None):
        rows = self.runs if rows is None else rows
        return np.array([[r[n] for n in C.CONTROL_NAMES] for r in rows], float)

    def role_counts(self):
        counts = {role: 0 for role in C.ROLES}
        for r in self.runs:
            counts[r["role"]] += 1
        return counts


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


# --------------------------------------------------------------------------- archive
def _safe_name(raw: str):
    if "\x00" in raw or "\\" in raw or raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        return None
    name = raw[:-1] if raw.endswith("/") else raw
    parts = name.split("/")
    if any(p in ("..", ".", "") for p in parts) or len(raw) > C.LIMITS["name_length"]:
        return None
    return name


def read_archive(data: bytes, limits=C.LIMITS):
    """Return ({name: bytes}, issues). Checks every archive-level abuse before reading content."""
    issues = []
    if len(data) > limits["zip_bytes"]:
        return {}, [issue("error", "archive_too_large", f"ZIP is {len(data)} bytes; the limit is {limits['zip_bytes']}.",
                          "Remove verification runs or split the study.")]
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        infos = archive.infolist()
    except (zipfile.BadZipFile, ValueError, OSError) as exc:
        return {}, [issue("error", "not_a_zip", f"The upload is not a readable ZIP archive ({exc}).",
                          "Create the bundle with the documented packer or a standard ZIP tool.")]
    if len(infos) > limits["files"]:
        return {}, [issue("error", "too_many_files", f"{len(infos)} entries; the limit is {limits['files']}.")]
    names, seen, folded, total = [], set(), set(), 0
    for info in infos:
        name = _safe_name(info.filename)
        if name is None:
            issues.append(issue("error", "unsafe_path", f"Unsafe archive path {info.filename!r}.",
                                "Use relative paths without '..', backslashes or drive letters."))
            continue
        if info.is_dir():
            continue
        if IGNORED.search(name):
            issues.append(issue("warning", "ignored_file", f"Ignored operating-system file {name}."))
            continue
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            issues.append(issue("error", "symlink", f"{name} is a symbolic link.", "Store the real file instead."))
            continue
        if info.flag_bits & 0x1:
            issues.append(issue("error", "encrypted", f"{name} is encrypted.", "Upload an unencrypted ZIP."))
            continue
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            issues.append(issue("error", "compression", f"{name} uses an unsupported compression method.",
                                "Use stored or deflate compression."))
            continue
        if name in seen or name.lower() in folded:
            issues.append(issue("error", "duplicate_entry", f"Duplicate archive entry {name}."))
            continue
        seen.add(name)
        folded.add(name.lower())
        cap = NPY_BYTES if name.endswith(".npy") else limits["text_bytes"]
        if info.file_size > cap:
            issues.append(issue("error", "member_too_large", f"{name} is {info.file_size} bytes when unpacked; the limit is {cap}.",
                                f"A field must be one float32 {FIELD_SHAPE} .npy ({NPY_BYTES} bytes); float64 or larger grids "
                                "are not accepted. Text files must stay under 4 MiB." if name.endswith(".npy") else
                                "Text files must stay under 4 MiB."))
            continue
        # .npy members are capped at one field's exact size; the ratio rule bounds text members.
        if not name.endswith(".npy") and info.compress_size and info.file_size / info.compress_size > limits["compression_ratio"]:
            issues.append(issue("error", "compression_ratio", f"{name} expands more than {limits['compression_ratio']:.0f}x.",
                                "This looks like a decompression bomb; upload ordinary data."))
            continue
        total += info.file_size
        names.append((name, info))
    if total > limits["unpacked_bytes"]:
        issues.append(issue("error", "unpacked_too_large", f"Unpacked size {total} exceeds {limits['unpacked_bytes']} bytes."))
    if any(i["level"] == "error" for i in issues):
        return {}, issues
    # Accept one wrapping folder (common with Finder/Explorer "compress folder").
    roots = {n.split("/", 1)[0] for n, _ in names}
    strip = ""
    if len(roots) == 1 and all("/" in n for n, _ in names):
        strip = next(iter(roots)) + "/"
    files = {}
    for name, info in names:
        with archive.open(info) as stream:
            content = stream.read(info.file_size + 1)
        if len(content) != info.file_size:
            issues.append(issue("error", "size_mismatch", f"{name} does not match its declared size."))
            continue
        files[name[len(strip):]] = content
    return files, issues


# --------------------------------------------------------------------------- parsing helpers
def parse_npy(content: bytes, name: str, issues: list):
    stream = io.BytesIO(content)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=1024)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=1024)
        else:
            raise ValueError(f"unsupported .npy version {version}")
    except (ValueError, OSError, SyntaxError) as exc:
        issues.append(issue("error", "bad_npy", f"{name} is not a readable .npy array ({exc}).",
                            "Save with numpy.save(path, array.astype(numpy.float32))."))
        return None
    if dtype.hasobject:
        issues.append(issue("error", "pickle", f"{name} contains Python objects.", "Upload plain float32 arrays."))
        return None
    if dtype != np.dtype("<f4") or fortran or tuple(shape) != FIELD_SHAPE:
        issues.append(issue("error", "array_contract",
                            f"{name} is {dtype.str}{' Fortran-order' if fortran else ''} with shape {tuple(shape)}; "
                            f"expected little-endian float32 C-order {FIELD_SHAPE} (density, pressure, u, v; [c, y, x]).",
                            "Do not resample or reorder silently: produce fields on the documented 64x64 periodic grid."))
        return None
    try:
        array = np.lib.format.read_array(io.BytesIO(content), allow_pickle=False)
    except (ValueError, OSError) as exc:
        issues.append(issue("error", "bad_npy", f"{name} could not be read ({exc})."))
        return None
    if not np.isfinite(array).all():
        issues.append(issue("error", "nonfinite", f"{name} contains NaN or infinite values."))
        return None
    return array


def parse_csv(content: bytes, name: str, issues: list):
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        issues.append(issue("error", "encoding", f"{name} is not UTF-8 text."))
        return None, []
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as exc:
        issues.append(issue("error", "csv", f"{name} is not valid CSV ({exc})."))
        return None, []
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        issues.append(issue("error", "csv_empty", f"{name} is empty."))
        return None, []
    header = [h.strip() for h in rows[0]]
    if len(set(header)) != len(header):
        issues.append(issue("error", "csv_header", f"{name} repeats a column name."))
        return None, []
    body = []
    for line, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            issues.append(issue("error", "csv_ragged", f"{name} line {line} has {len(row)} cells; header has {len(header)}."))
            return None, []
        body.append({h: cell.strip() for h, cell in zip(header, row)})
    return header, body


def parse_float(text: str):
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_json(content: bytes, name: str, issues: list):
    try:
        value = json.loads(content.decode("utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    except (UnicodeDecodeError, ValueError) as exc:
        issues.append(issue("error", "json", f"{name} is not valid JSON ({exc})."))
        return None
    if not isinstance(value, dict):
        issues.append(issue("error", "json", f"{name} must contain a JSON object."))
        return None
    return value


def setting_key(values):
    return tuple(float(f"{v:.12g}") for v in values)


# --------------------------------------------------------------------------- study.json
def check_study_meta(meta: dict, issues: list):
    """Exact compatibility check. Returns declared control ranges."""
    if meta.get("schema") != C.SCHEMA:
        issues.append(issue("error", "schema", f"study.json schema must be {C.SCHEMA!r}, got {meta.get('schema')!r}."))
    for key, expected in C.STUDY_REQUIREMENTS.items():
        if meta.get(key) != expected:
            issues.append(issue("error", "incompatible_" + key,
                                f"study.json {key} is {json.dumps(meta.get(key))}; this capability requires {json.dumps(expected)}.",
                                "Only the documented 64x64 periodic flow family is supported. Do not rename an incompatible problem."))
    name = meta.get("study_name")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80 or any(ord(ch) < 32 for ch in name):
        issues.append(issue("error", "study_name", "study_name must be 1 to 80 printable characters."))
    provenance = meta.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("data_kind") not in DATA_KINDS:
        issues.append(issue("error", "provenance", f"provenance.data_kind must be one of {DATA_KINDS}.",
                            "Label generated research data and customer data explicitly."))
    seed = meta.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        issues.append(issue("error", "seed", "seed must be an integer in [0, 2^31)."))
    tolerance = meta.get("tolerance", C.DEFAULT_TOLERANCE)
    if (not isinstance(tolerance, list) or len(tolerance) != 2
            or not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) and t > 0 for t in tolerance)):
        issues.append(issue("error", "tolerance", "tolerance must be two positive absolute outcome tolerances."))
    outcomes = meta.get("outcomes")
    expected_outcomes = [{"name": o["name"], "unit": o["unit"]} for o in C.OUTCOMES]
    if not isinstance(outcomes, list) or [{"name": o.get("name"), "unit": o.get("unit")} for o in outcomes if isinstance(o, dict)] != expected_outcomes:
        issues.append(issue("error", "outcomes", f"outcomes must be exactly {expected_outcomes}.",
                            "Both outcomes are derived from the result fields; other outcomes are not supported yet."))
    ranges = []
    controls = meta.get("controls")
    if not isinstance(controls, list) or [c.get("name") for c in controls if isinstance(c, dict)] != C.CONTROL_NAMES:
        issues.append(issue("error", "controls", f"controls must be {C.CONTROL_NAMES} in this order."))
        return [list(r) for r in C.RANGES]
    for spec, supported in zip(controls, C.CONTROLS):
        lo_hi = spec.get("range")
        ok = (isinstance(lo_hi, list) and len(lo_hi) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in lo_hi)
              and supported["range"][0] <= lo_hi[0] < lo_hi[1] <= supported["range"][1])
        if not ok or spec.get("unit") != "nondimensional":
            issues.append(issue("error", "control_range",
                                f"{spec.get('name')} range {lo_hi} must be nondimensional and inside the supported {supported['range']}.",
                                "Extrapolation outside the supported box is not offered."))
            ranges.append(list(supported["range"]))
        else:
            ranges.append([float(lo_hi[0]), float(lo_hi[1])])
    acquisition = meta.get("acquisition")
    if acquisition is not None:
        if (not isinstance(acquisition, dict) or acquisition.get("policy") not in ACQUISITION_POLICIES
                or not isinstance(acquisition.get("seed"), int) or not isinstance(acquisition.get("sequence_length"), int)
                or not 1 <= acquisition["sequence_length"] <= C.LIMITS["candidates"]):
            issues.append(issue("error", "acquisition", f"acquisition must name a policy in {ACQUISITION_POLICIES} with integer seed and sequence_length."))
    return ranges


# --------------------------------------------------------------------------- validation
def _check_checksums(files: dict, issues: list):
    content = files.get("checksums.json")
    if content is None:
        issues.append(issue("error", "missing_checksums", "checksums.json is missing.", "Pack with the documented packer."))
        return {}
    record = parse_json(content, "checksums.json", issues)
    if record is None:
        return {}
    listed = record.get("files")
    if record.get("algorithm") != "sha256" or not isinstance(listed, dict):
        issues.append(issue("error", "checksums", "checksums.json must have algorithm 'sha256' and a files object."))
        return {}
    actual = {name: {"sha256": sha256_bytes(data), "bytes": len(data)} for name, data in files.items() if name != "checksums.json"}
    for name in sorted(set(actual) | set(listed)):
        if name not in listed:
            issues.append(issue("error", "checksum_missing", f"{name} is not listed in checksums.json."))
        elif name not in actual:
            issues.append(issue("error", "checksum_extra", f"checksums.json lists {name}, which is not in the archive."))
        elif listed[name] != actual[name]:
            issues.append(issue("error", "checksum_mismatch", f"{name} does not match its checksum."))
    return actual


def _parse_points(files, name, ranges, issues):
    if name not in files:
        return None
    id_column = POINT_COLUMNS[name]
    header, rows = parse_csv(files[name], name, issues)
    if header is None:
        return None
    if header != [id_column, *C.CONTROL_NAMES]:
        issues.append(issue("error", name.split(".")[0] + "_columns",
                            f"{name} columns must be exactly {[id_column, *C.CONTROL_NAMES]}; got {header}.",
                            "Candidate and probe files may contain controls only, never outcomes or fields."))
        return None
    limit = C.LIMITS["candidates" if name == "candidates.csv" else "probes"]
    if len(rows) > limit:
        issues.append(issue("error", "too_many_points", f"{name} has {len(rows)} rows; the limit is {limit}."))
        return None
    ids, xs, keys = [], [], set()
    for line, row in enumerate(rows, start=2):
        x = [parse_float(row[n]) for n in C.CONTROL_NAMES]
        if not RUN_ID.match(row[id_column]) or None in x:
            issues.append(issue("error", "bad_point", f"{name} line {line} needs a valid id and finite controls."))
            continue
        if not all(lo <= v <= hi for v, (lo, hi) in zip(x, ranges)):
            issues.append(issue("error", "point_out_of_range", f"{name} line {line} lies outside the declared control ranges."))
            continue
        if row[id_column] in ids or setting_key(x) in keys:
            issues.append(issue("error", "duplicate_point", f"{name} line {line} repeats an id or setting."))
            continue
        ids.append(row[id_column])
        xs.append(x)
        keys.add(setting_key(x))
    return Points(ids, np.array(xs, float).reshape(-1, 3))


def _parse_runs(files, ranges, issues, warnings):
    if "runs.csv" not in files:
        issues.append(issue("error", "missing_runs", "runs.csv is missing."))
        return []
    header, rows = parse_csv(files["runs.csv"], "runs.csv", issues)
    if header is None:
        return []
    missing = [c for c in REQUIRED_RUN_COLUMNS if c not in header]
    if missing:
        issues.append(issue("error", "runs_columns", f"runs.csv is missing columns {missing}.",
                            f"Required columns: {REQUIRED_RUN_COLUMNS}."))
        return []
    extra = [c for c in header if c not in REQUIRED_RUN_COLUMNS + OPTIONAL_RUN_COLUMNS]
    if extra:
        warnings.append(issue("warning", "ignored_columns", f"runs.csv columns {extra} are ignored."))
    if len(rows) > C.LIMITS["runs_total"]:
        issues.append(issue("error", "too_many_runs", f"{len(rows)} runs; the limit is {C.LIMITS['runs_total']}."))
        return []
    parsed, ids, used_files, by_setting = [], set(), set(), {}
    for line, row in enumerate(rows, start=2):
        rid = row["run_id"]
        if not RUN_ID.match(rid):
            issues.append(issue("error", "run_id", f"runs.csv line {line}: run_id {rid!r} must be 1-64 letters, digits, '.', '_' or '-'."))
            continue
        if rid in ids:
            issues.append(issue("error", "duplicate_run_id", f"runs.csv line {line}: run_id {rid} repeats."))
            continue
        ids.add(rid)
        x = [parse_float(row[n]) for n in C.CONTROL_NAMES]
        if None in x:
            issues.append(issue("error", "nonfinite_control", f"Run {rid}: controls must be finite numbers."))
            continue
        bad = [n for n, v, (lo, hi) in zip(C.CONTROL_NAMES, x, ranges) if not lo <= v <= hi]
        if bad:
            issues.append(issue("error", "control_out_of_range", f"Run {rid}: {', '.join(bad)} outside the declared range.",
                                "Remove the run or widen the declared range within the supported box.", run_id=rid))
            continue
        role = row["role"]
        if role not in C.ROLES:
            issues.append(issue("error", "role", f"Run {rid}: role {role!r} must be one of {C.ROLES}."))
            continue
        entry = {"run_id": rid, "role": role, **dict(zip(C.CONTROL_NAMES, x))}
        for column in ("initial_file", "result_file"):
            path = row[column]
            if not FIELD_PATH.match(path):
                issues.append(issue("error", "field_path", f"Run {rid}: {column} must be fields/<name>.npy."))
                entry = None
                break
            if path in used_files:
                issues.append(issue("error", "shared_field", f"Run {rid}: {path} is referenced by more than one run or column."))
                entry = None
                break
            if path not in files:
                issues.append(issue("error", "missing_field", f"Run {rid}: {path} is not in the archive."))
                entry = None
                break
            used_files.add(path)
            entry[column] = path
        if entry is None:
            continue
        for name in C.OUTCOME_NAMES + ["solver_seconds", "solver_cost"]:
            if name in header and row[name] != "":
                value = parse_float(row[name])
                if value is None or (name.startswith("solver") and value < 0):
                    issues.append(issue("error", "bad_number", f"Run {rid}: {name} must be a finite"
                                        + (" non-negative" if name.startswith("solver") else "") + " number."))
                    value = None
                entry[name] = value
        currency = row.get("cost_currency", "")
        if currency and not re.match(r"^[A-Z]{3}$", currency):
            issues.append(issue("error", "currency", f"Run {rid}: cost_currency must be a three-letter code."))
        if entry.get("solver_cost") is not None and not currency:
            issues.append(issue("error", "currency", f"Run {rid}: solver_cost needs cost_currency."))
        entry["cost_currency"] = currency or None
        key = setting_key(x)
        if key in by_setting:
            other = by_setting[key]
            where = "the same role" if other["role"] == role else f"roles {other['role']} and {role}"
            issues.append(issue("error", "duplicate_setting",
                                f"Runs {other['run_id']} and {rid} share identical controls in {where}.",
                                "Keep one run per setting; repeated settings must never cross data roles.", run_id=rid))
            continue
        by_setting[key] = entry
        parsed.append(entry)
    unreferenced = sorted(n for n in files if n.startswith("fields/") and n not in used_files)
    if unreferenced:
        issues.append(issue("error", "unreferenced_fields", f"{len(unreferenced)} field files are not referenced by runs.csv, e.g. {unreferenced[0]}.",
                            "Remove unused arrays; hidden data is not accepted."))
    return parsed


def _check_fields(files, runs, issues):
    n = len(runs)
    initial = np.zeros((n,) + FIELD_SHAPE, np.float32)
    result = np.zeros((n,) + FIELD_SHAPE, np.float32)
    ok = True
    for i, run in enumerate(runs):
        a = parse_npy(files[run["initial_file"]], run["initial_file"], issues)
        b = parse_npy(files[run["result_file"]], run["result_file"], issues)
        if a is None or b is None:
            ok = False
            continue
        initial[i], result[i] = a, b
    if not ok or not n:
        return None, None, None
    expected = initial_fields(np.array([[r[c] for c in C.CONTROL_NAMES] for r in runs]))
    for i, run in enumerate(runs):
        rid = run["run_id"]
        gap = float(np.abs(initial[i] - expected[i]).max())
        if gap > INITIAL_ATOL:
            issues.append(issue("error", "initial_mismatch",
                                f"Run {rid}: initial field differs from {C.INITIAL_GENERATOR} at its controls (max abs {gap:.3g}).",
                                "Initial fields must be the documented analytic condition for the stated controls; "
                                "a different initial condition is a different, unsupported problem.", run_id=rid))
        if float(np.abs(result[i, 0] - 1).max()) > 1e-6:
            issues.append(issue("error", "density", f"Run {rid}: result density channel must be 1 (incompressible family).", run_id=rid))
        if float(np.abs(result[i, 1:]).max()) > 1e3:
            issues.append(issue("error", "magnitude", f"Run {rid}: result field magnitude exceeds 1e3; check units and scaling.", run_id=rid))
    # float32 readouts, exactly as Task 8 labels8 produced them; scientific code relies on this.
    outcomes = physical_outcomes(result)
    for i, run in enumerate(runs):
        for j, name in enumerate(C.OUTCOME_NAMES):
            stated = run.get(name)
            derived = float(outcomes[i, j])
            if stated is not None and abs(stated - derived) > OUTCOME_ATOL + OUTCOME_RTOL * abs(derived):
                issues.append(issue("error", "outcome_mismatch",
                                    f"Run {run['run_id']}: stated {name} {stated:.8g} differs from the field-derived {derived:.8g}.",
                                    "Outcomes are read from the result field; fix the scalar or the field.", run_id=run["run_id"]))
    return initial, result, outcomes


def _role_issues(runs, issues, warnings):
    counts = {role: 0 for role in C.ROLES}
    for r in runs:
        counts[r["role"]] += 1
    for role in ("train", "development", "calibration", "verification"):
        lo, hi = C.LIMITS[role]
        if counts[role] < lo:
            issues.append(issue("error", "role_count",
                                f"{counts[role]} {role} runs supplied; the {C.RECIPE['id']} recipe needs at least {lo}.",
                                f"Add {lo - counts[role]} more completed {role} runs."))
        if counts[role] > hi:
            issues.append(issue("error", "role_count", f"{counts[role]} {role} runs supplied; the limit is {hi}."))
    if counts["calibration"] < C.CALIBRATED_MINIMUM:
        warnings.append(issue("warning", "calibration_insufficient",
                              f"{counts['calibration']} calibration runs supplied; calibrated 90% joint intervals need {C.CALIBRATED_MINIMUM}.",
                              "Add calibration runs, or choose an explicitly uncalibrated prediction-only fit."))
    return counts


def validate_bundle(data: bytes, limits=C.LIMITS) -> ValidatedStudy:
    """Validate a complete study bundle. Raises BundleRejected with actionable issues."""
    files, issues = read_archive(data, limits)
    warnings = [i for i in issues if i["level"] == "warning"]
    issues = [i for i in issues if i["level"] == "error"]
    if issues:
        raise BundleRejected(issues + warnings)
    allowed = {"study.json", "runs.csv", "checksums.json", "candidates.csv", "probes.csv", "README.md"}
    for name in files:
        if name not in allowed and not FIELD_PATH.match(name):
            issues.append(issue("error", "unexpected_file", f"Unexpected file {name}.",
                                "Allowed: study.json, runs.csv, checksums.json, candidates.csv, probes.csv, README.md, fields/*.npy."))
    for name in ("study.json", "runs.csv"):
        if name not in files:
            issues.append(issue("error", "missing_" + name.split(".")[0], f"{name} is missing."))
    checks = _check_checksums(files, issues)
    meta = parse_json(files["study.json"], "study.json", issues) if "study.json" in files else None
    if meta is None or issues:
        raise BundleRejected(issues + warnings)
    ranges = check_study_meta(meta, issues)
    runs = _parse_runs(files, ranges, issues, warnings)
    candidates = _parse_points(files, "candidates.csv", ranges, issues)
    probes = _parse_points(files, "probes.csv", ranges, issues)
    if meta.get("acquisition") and candidates is None:
        issues.append(issue("error", "acquisition", "acquisition requires candidates.csv."))
    _role_issues(runs, issues, warnings)
    if issues:
        raise BundleRejected(issues + warnings)
    initial, result, outcomes = _check_fields(files, runs, issues)
    if issues or initial is None:
        raise BundleRejected(issues + warnings)
    if candidates is not None:
        supplied = {setting_key([r[c] for c in C.CONTROL_NAMES]): r["run_id"] for r in runs}
        for cid, x in zip(candidates.ids, candidates.x):
            rid = supplied.get(setting_key(x))
            if cid in {r["run_id"] for r in runs} and rid != cid:
                issues.append(issue("error", "candidate_id", f"Candidate {cid} reuses a run_id with different controls."))
        if issues:
            raise BundleRejected(issues + warnings)
    return ValidatedStudy("study", meta, runs, initial, result, outcomes, candidates, probes, checks,
                          canonical_hash(checks), sha256_bytes(data), warnings)


def validate_addition(data: bytes, ranges, limits=C.LIMITS) -> ValidatedStudy:
    """Validate completed runs being added to an existing study version."""
    files, issues = read_archive(data, limits)
    warnings = [i for i in issues if i["level"] == "warning"]
    issues = [i for i in issues if i["level"] == "error"]
    if issues:
        raise BundleRejected(issues + warnings)
    for name in files:
        if name not in {"addition.json", "runs.csv", "checksums.json", "README.md"} and not FIELD_PATH.match(name):
            issues.append(issue("error", "unexpected_file", f"Unexpected file {name} in an addition bundle."))
    checks = _check_checksums(files, issues)
    meta = parse_json(files["addition.json"], "addition.json", issues) if "addition.json" in files else None
    if meta is None:
        issues.append(issue("error", "missing_addition", "addition.json is missing or invalid."))
    elif meta.get("schema") != ADDITION_SCHEMA or meta.get("capability") != C.CAPABILITY_ID:
        issues.append(issue("error", "schema", f"addition.json must declare schema {ADDITION_SCHEMA!r} and capability {C.CAPABILITY_ID!r}."))
    elif meta.get("provenance", {}).get("data_kind") not in DATA_KINDS:
        issues.append(issue("error", "provenance", f"addition.json provenance.data_kind must be one of {DATA_KINDS}."))
    if issues:
        raise BundleRejected(issues + warnings)
    runs = _parse_runs(files, ranges, issues, warnings)
    if not runs and not issues:
        issues.append(issue("error", "empty_addition", "The addition contains no runs."))
    if issues:
        raise BundleRejected(issues + warnings)
    initial, result, outcomes = _check_fields(files, runs, issues)
    if issues or initial is None:
        raise BundleRejected(issues + warnings)
    return ValidatedStudy("addition", meta, runs, initial, result, outcomes, None, None, checks,
                          canonical_hash(checks), sha256_bytes(data), warnings)


# --------------------------------------------------------------------------- packing
def npy_bytes(array) -> bytes:
    array = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    if array.shape != FIELD_SHAPE:
        raise ValueError(f"field shape {array.shape} != {FIELD_SHAPE}")
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, version=(1, 0), allow_pickle=False)
    return stream.getvalue()


def csv_bytes(header, rows) -> bytes:
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(["" if v is None else (repr(float(v)) if isinstance(v, (float, np.floating)) else v) for v in row])
    return stream.getvalue().encode()


def zip_files(files: dict) -> bytes:
    """Deterministic ZIP: sorted names, fixed timestamps and permissions."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, files[name])
    return stream.getvalue()


def pack(kind: str, meta: dict, runs: list[dict], initial: dict, result: dict,
         candidates: Points | None = None, probes: Points | None = None, readme: str | None = None) -> bytes:
    """Build a study ("study") or addition ("addition") bundle from in-memory data.

    runs: dicts with run_id, controls, role, optional outcomes/costs. initial/result: run_id -> array.
    """
    files = {("study.json" if kind == "study" else "addition.json"): (json.dumps(meta, indent=2, allow_nan=False) + "\n").encode()}
    header = REQUIRED_RUN_COLUMNS + [c for c in OPTIONAL_RUN_COLUMNS if any(r.get(c) is not None for r in runs)]
    rows = []
    for r in runs:
        rid = r["run_id"]
        files[f"fields/{rid}_initial.npy"] = npy_bytes(initial[rid])
        files[f"fields/{rid}_result.npy"] = npy_bytes(result[rid])
        values = {**r, "initial_file": f"fields/{rid}_initial.npy", "result_file": f"fields/{rid}_result.npy"}
        rows.append([values.get(c) for c in header])
    files["runs.csv"] = csv_bytes(header, rows)
    for name, points in (("candidates.csv", candidates), ("probes.csv", probes)):
        if points is not None:
            files[name] = csv_bytes([POINT_COLUMNS[name], *C.CONTROL_NAMES], [[i, *map(float, x)] for i, x in zip(points.ids, points.x)])
    if readme:
        files["README.md"] = readme.encode()
    files["checksums.json"] = (json.dumps({"algorithm": "sha256", "files": {
        n: {"sha256": sha256_bytes(b), "bytes": len(b)} for n, b in sorted(files.items())}}, indent=2) + "\n").encode()
    return zip_files(files)


def study_meta(name: str, data_kind: str, description: str, seed: int = 0, tolerance=None,
               ranges=None, acquisition=None) -> dict:
    """study.json with every required compatibility statement for this capability."""
    meta = {"schema": C.SCHEMA, "study_name": name, **json.loads(json.dumps(C.STUDY_REQUIREMENTS)),
            "controls": [{"name": c["name"], "range": list(r), "unit": c["unit"]}
                         for c, r in zip(C.CONTROLS, ranges or C.RANGES)],
            "outcomes": [{"name": o["name"], "unit": o["unit"], "definition": o["definition"]} for o in C.OUTCOMES],
            "seed": int(seed), "tolerance": list(tolerance or C.DEFAULT_TOLERANCE),
            "roles": {"policy": "declared in runs.csv", "allowed": C.ROLES},
            "provenance": {"data_kind": data_kind, "description": description}}
    if acquisition:
        meta["acquisition"] = acquisition
    return meta


def combine(base: ValidatedStudy, additions: list) -> ValidatedStudy:
    """One immutable version: base bundle plus completed-run additions, in order.

    Earlier roles are preserved. New runs may not repeat an existing run ID or setting.
    """
    runs, initial, result, outcomes = list(base.runs), [base.initial], [base.result], [base.outcomes]
    ids = {r["run_id"] for r in runs}
    keys = {setting_key([r[c] for c in C.CONTROL_NAMES]): r["run_id"] for r in runs}
    issues = []
    for addition in additions:
        for r in addition.runs:
            key = setting_key([r[c] for c in C.CONTROL_NAMES])
            if r["run_id"] in ids:
                issues.append(issue("error", "duplicate_run_id", f"Added run {r['run_id']} already exists in this study."))
            elif key in keys:
                issues.append(issue("error", "duplicate_setting",
                                    f"Added run {r['run_id']} repeats the controls of existing run {keys[key]}.",
                                    "Each setting appears once; repeated settings must never cross data roles."))
            ids.add(r["run_id"])
            keys[key] = r["run_id"]
        runs += addition.runs
        initial.append(addition.initial)
        result.append(addition.result)
        outcomes.append(addition.outcomes)
    if issues:
        raise BundleRejected(issues)
    content = canonical_hash([base.content_sha256] + [a.content_sha256 for a in additions])
    combined = ValidatedStudy("version", base.meta, runs, np.concatenate(initial), np.concatenate(result),
                              np.concatenate(outcomes), base.candidates, base.probes, base.files, content,
                              base.archive_sha256, list(base.warnings))
    counts = combined.role_counts()
    over = [role for role in C.ROLES if counts[role] > C.LIMITS[role][1]]
    if over or len(runs) > C.LIMITS["runs_total"]:
        raise BundleRejected([issue("error", "role_count", f"Adding these runs exceeds the limit for {over or 'total runs'}.")])
    return combined
