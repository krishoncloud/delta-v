"""Install and fetch the worked example's pre-adapted model (adapted once on a GPU, e.g. with the MoLab notebook).

Every file is checked against model_manifest.json, the model must be the
task8-fno-lhs-v1 recipe on the pinned example version, and the ensemble is
reloaded with weight hashes verified before it is marked ready.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath

from . import adapt as A
from . import bundle as B
from . import engine as E

class InstallError(ValueError):
    pass


EXAMPLE = json.loads((Path(__file__).resolve().parent / "examples" / "task8_r701.json").read_text())
EQUIVALENCE_BASIS = ("weights identical to the recorded research model; training runs identical; CPU spot check "
                     "reproduces stored predictions; verification metrics recomputed from this site's example copy")


def example_content_sha(example_dir: Path) -> str:
    path = example_dir / EXAMPLE["files"]["study"]["name"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXAMPLE["files"]["study"]["sha256"]:
        raise InstallError(f"{path} is not the pinned example bundle; run scripts/pack_task8_example.py first")
    return B.validate_bundle(data).content_sha256


def install(files: Path, dest_root: Path, example_dir: Path, allow_test_profile=False) -> Path:
    manifest_bytes = (files / "model_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("kind") != "deltav-adapted-example-model/1":
        raise InstallError("model_manifest.json is not a Delta-V example model manifest")
    if manifest["profile"] != "task8-fno-lhs-v1" and not allow_test_profile:
        raise InstallError(f"Profile {manifest['profile']!r} is not the task8-fno-lhs-v1 recipe")
    expected_key = example_content_sha(example_dir)
    keyed_elsewhere = manifest["version_key"] != expected_key
    stage = dest_root / (EXAMPLE["example_id"] + ".incoming")
    if stage.exists():
        shutil.rmtree(stage)
    model = stage / "model"
    model.mkdir(parents=True)
    with zipfile.ZipFile(files / "deltav-example-model-metadata.zip") as archive:
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if info.is_dir() or name.name == "model_manifest.json":
                continue
            if name.is_absolute() or ".." in name.parts or str(name) not in manifest["files"]:
                raise InstallError(f"Unexpected file in the metadata ZIP: {info.filename}")
            target = model / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    for name in manifest["files"]:
        if name.endswith("weights.pt"):
            source = files / (PurePosixPath(name).parent.name + "__weights.pt")
            if not source.is_file():
                raise InstallError(f"Missing weight file {source.name}")
            shutil.copyfile(source, model / name)
    for name, entry in manifest["files"].items():
        path = model / name
        if not path.is_file() or path.stat().st_size != entry["bytes"] or E.digest(path) != entry["sha256"]:
            raise InstallError(f"{name} does not match model_manifest.json")
    equivalence = None
    if keyed_elsewhere:
        equivalence = equivalent_to_recorded(model, example_dir)
        meta = json.loads((model / "model.json").read_text())
        (model / "model.original.json").write_text(json.dumps(meta, indent=2))
        meta["version_key"] = expected_key
        meta["provenance"] = {"adapted_version_key": manifest["version_key"], "version_key": expected_key, **equivalence}
        (model / "model.json").write_text(json.dumps(meta, indent=2))
    A.Ensemble(model)  # reload every member with its weight hash checked
    result = A.model_result(model)
    meta = result["model"]
    info = {"example_id": EXAMPLE["example_id"], "version_key": meta["version_key"], "profile": meta["profile"],
            "equivalence": equivalence,
            "device_name": meta["environment"]["device_name"], "device": meta["environment"]["device"],
            "adapted_at": manifest.get("adapted_at"), "imported_at": time.time(),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "label": f"Adapted in advance with the standard fine-tuning recipe on {meta['environment']['device_name']}; "
                     "not adapted live on this site."}
    if meta["profile"] != "task8-fno-lhs-v1":
        info["label"] = "TEST STAND-IN: " + meta["profile_label"]
    (stage / "result.json").write_text(json.dumps(result))
    (stage / "preadapted.json").write_text(json.dumps(info, indent=2))
    (stage / "model_manifest.json").write_bytes(manifest_bytes)
    final = dest_root / EXAMPLE["example_id"]
    if final.exists():
        shutil.rmtree(final)
    os.replace(stage, final)
    return final



def equivalent_to_recorded(model: Path, example_dir: Path):
    """Accept a model whose recorded study key differs only if it is provably the recorded Task 8 model.

    Every member's weights must equal the weights recorded in the Task 8 archive, its training runs must be the
    example's, a CPU spot check must reproduce the stored predictions, and the stored verification metrics must
    follow from this site's own copy of the example's verification results.
    """
    import numpy as np

    recorded = EXAMPLE["recorded_member_weights"]["members"]
    meta = json.loads((model / "model.json").read_text())
    if meta["profile"] != "task8-fno-lhs-v1" or sorted(str(m["seed"]) for m in meta["members"]) != sorted(recorded):
        raise InstallError("The model was adapted on a different study version than the pinned example")
    for m in meta["members"]:
        if m["weights_sha256"] != recorded[str(m["seed"])]["sha256"]:
            raise InstallError("The model was adapted on a different study version, and its weights are not the recorded Task 8 weights")
        train = json.loads((model / m["dir"] / "contract.json").read_text())["normalization"]["train_ids"]
        if train != EXAMPLE["train_ids"]:
            raise InstallError("The model's training runs differ from the worked example's")
    study = B.validate_bundle((example_dir / EXAMPLE["files"]["study"]["name"]).read_bytes())
    stored = E.load_npz(model / "aggregate_predictions.npz")
    roles = stored["roles"]
    x_by_id = {r["run_id"]: [r[c] for c in ("viscosity", "forcing", "initial_mix")] for r in study.runs}
    y_by_id = {r["run_id"]: o for r, o in zip(study.runs, study.outcomes)}
    ver = np.flatnonzero(roles == "verification")
    ids = [str(i) for i in stored["ids"][ver]]
    check = A.Ensemble(model).predict([x_by_id[i] for i in ids[:4]])
    if not np.allclose(check["mean"], stored["mean"][ver[:4]], atol=1e-5, rtol=1e-4):
        raise InstallError("A CPU spot check did not reproduce the model's stored predictions")
    metrics = E.verification_metrics(np.array([y_by_id[i] for i in ids]), stored["mean"][ver], stored["sd"][ver],
                                     np.array(meta["floor"]), meta["calibration"].get("rule"), meta["tolerance"], meta["recipe"]["level"])
    for key in ("joint_accuracy_success", "joint_coverage"):
        if abs(metrics[key] - meta["verification"][key]) > 1e-9:
            raise InstallError("Stored verification metrics do not follow from the example's verification runs")
    return {"basis": EQUIVALENCE_BASIS,
            "spot_check_max_abs_diff": float(np.abs(check["mean"] - stored["mean"][ver[:4]]).max())}


def fetch_from_hub(repo, revision, path_in_repo, dest_root: Path, example_dir: Path, manifest_sha256=None):
    """Download the MoLab files from a Hugging Face dataset repo (pinned revision) and install them."""
    from huggingface_hub import hf_hub_download

    def get(name):
        return Path(hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision, filename=f"{path_in_repo.rstrip('/')}/{name}",
                                    token=os.environ.get("HF_TOKEN") or False))
    manifest = get("model_manifest.json")
    if manifest_sha256 and hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_sha256:
        raise InstallError("Pre-adapted model manifest does not match FLOW_PREADAPTED_MANIFEST_SHA256")
    files = manifest.parent
    get("deltav-example-model-metadata.zip")
    for name in json.loads(manifest.read_text())["files"]:
        if name.endswith("weights.pt"):
            get(PurePosixPath(name).parent.name + "__weights.pt")
    return install(files, dest_root, example_dir)
