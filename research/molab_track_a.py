"""Delta-V Task 1: Euler transfer and design-run reduction on MoLab."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Delta-V · Task 1 · Transfer and design-run evidence

    This is one notebook with two independent studies. **Track A** asks whether
    pretraining transfers to Euler fields. **Track B** asks whether a model can
    predict engineering outcomes with fewer measured cooling-time settings.
    Neither track is a production claim until its results are inspected.

    ### Track A — Euler transfer

    This notebook reuses four Euler gamma settings saved under v2, then compares **scratch FNO**,
    **full fine-tuning of epoch 59**, and **a frozen FNO with learned adapters**.
    The nested training budgets are 1, 2, 4, 8, 16, and 32 trajectories.
    Pilot runs use validation data. Final runs use three seeds and a held-out
    trajectory test set after the training protocol is locked.

    Before starting: attach MoLab's GPU using notebook specs and add `HF_TOKEN`
    in the Secrets panel. It must have read/write access to
    `KrishMalik/deltav-fluid`. In MoLab's package panel, pin
    `torch==2.11.0`, `neuraloperator==2.0.0`, `tensorly-torch==0.5.0`,
    `tensorly==0.10.0`; also install `h5py`, `httpx`, `huggingface_hub`,
    `numpy`, `scipy`, `scikit-image`, and `matplotlib`.

    Every expensive action below has its own button. The original training
    notebook is **not executed**. The 92 GB `slices/` directory and complete
    21 GB Euler source files are **never downloaded**. Existing gamma windows
    are read from v2; new training jobs are saved under v3. MoLab's
    local files are temporary; the HF copies are the durable record.
    The four existing Euler artifacts that all contain gamma 1.13 are left
    untouched; Track A reuses the verified v2 gamma groups.
    """)
    return


@app.cell
def _():
    import csv
    import hashlib
    import io
    import json
    import os
    import statistics
    import subprocess
    import sys
    import time
    from importlib.metadata import version as package_version
    from pathlib import Path

    import marimo as mo
    import torch
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    return (
        HfApi,
        Path,
        csv,
        hashlib,
        hf_hub_download,
        io,
        json,
        mo,
        os,
        package_version,
        snapshot_download,
        statistics,
        subprocess,
        sys,
        time,
        torch,
    )


@app.cell
def _(Path):
    DATASET = "KrishMalik/deltav-fluid"
    GITHUB = "https://github.com/krishoncloud/delta-v.git"
    BRANCH = "research/euler-transfer"
    TRACK_B_CODE_SHA = "9682e94100deb294f4bc2a66a2cef12acce3639e"
    PREFIX = "research/euler-track-a/v3"
    DATA_PREFIX = "research/euler-track-a/v2"
    SOURCE_REVISION = "84e3eb65faf286533714295515caf9b99b4717a7"
    NOTEBOOK_SHA256 = "e5204f1326e9abf2f16f18cd15ffc6b53425f4d3ac49774c22ed06c318871caf"
    GAMMAS = ("1.13", "1.22", "1.33", "1.365")
    BUDGETS = (1, 2, 4, 8, 16, 32)
    SEEDS = (42, 123, 2026)
    METHODS = ("scratch", "full_finetune", "adapter_only")
    ROOT = Path("/tmp/deltav-track-a-v3")
    CODE = ROOT / "code"
    DATA = ROOT / DATA_PREFIX / "data"
    return (
        BRANCH,
        BUDGETS,
        CODE,
        DATA,
        DATA_PREFIX,
        DATASET,
        GAMMAS,
        GITHUB,
        METHODS,
        NOTEBOOK_SHA256,
        PREFIX,
        ROOT,
        SEEDS,
        SOURCE_REVISION,
        TRACK_B_CODE_SHA,
    )


@app.cell
def _(Path, hashlib, json):
    def file_sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def json_bytes(value):
        return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()

    return file_sha256, json_bytes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Set up the pinned code and checkpoint

    This clones the research branch, pins its commit in a protocol file on HF,
    and downloads only the epoch-59 checkpoint. Later sessions reuse that
    commit and checkpoint revision even if GitHub or HF changes.
    """)
    return


@app.cell
def _(mo):
    setup_button = mo.ui.run_button(label="1 · Set up code, GPU and checkpoint")
    setup_button
    return (setup_button,)


@app.cell
def _(
    BRANCH,
    CODE,
    DATASET,
    GITHUB,
    HfApi,
    NOTEBOOK_SHA256,
    PREFIX,
    Path,
    SOURCE_REVISION,
    file_sha256,
    hf_hub_download,
    io,
    json,
    json_bytes,
    mo,
    os,
    package_version,
    setup_button,
    subprocess,
    torch,
):
    mo.stop(not setup_button.value, mo.md("Click **Set up** after attaching the GPU and setting `HF_TOKEN`."))
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in MoLab Secrets, then rerun this cell")
    if not torch.cuda.is_available():
        raise RuntimeError("Attach the GPU from MoLab notebook specs first")
    for _package, _expected in (("torch", "2.11.0"), ("neuraloperator", "2.0.0"),
                                ("tensorly-torch", "0.5.0"), ("tensorly", "0.10.0")):
        if package_version(_package).split("+")[0] != _expected:
            raise RuntimeError(f"Install {_package}=={_expected} in MoLab's package panel")
    api = HfApi(token=token)
    if not CODE.exists():
        CODE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--branch", BRANCH, GITHUB, str(CODE)], check=True)
    else:
        _checkout_status = subprocess.check_output(
            ["git", "-C", str(CODE), "status", "--porcelain"], text=True).strip()
        if _checkout_status:
            raise ValueError("MoLab research-code checkout has local edits; inspect before continuing")
        subprocess.run(["git", "-C", str(CODE), "fetch", "origin", BRANCH], check=True)
    protocol_path = f"{PREFIX}/protocol.json"
    remote_files = set(api.list_repo_files(DATASET, repo_type="dataset"))
    if protocol_path in remote_files:
        saved_path = hf_hub_download(DATASET, protocol_path, repo_type="dataset", token=token)
        protocol = json.loads(Path(saved_path).read_text())
        if protocol["source_revision"] != SOURCE_REVISION or protocol["notebook_sha256"] != NOTEBOOK_SHA256:
            raise ValueError("Saved Track A protocol differs from this notebook; do not mix runs")
        subprocess.run(["git", "-C", str(CODE), "checkout", protocol["code_sha"]], check=True)
    else:
        if any(path.startswith(f"{PREFIX}/") for path in remote_files):
            raise ValueError("Research files already exist without protocol.json; choose another prefix")
        subprocess.run(["git", "-C", str(CODE), "checkout", "--detach", "origin/" + BRANCH], check=True)
        code_sha = subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
        checkpoint_revision = api.dataset_info(DATASET).sha
        protocol = {"kind": "Delta-V Euler Track A", "code_sha": code_sha,
                    "checkpoint_revision": checkpoint_revision,
                    "source_revision": SOURCE_REVISION,
                    "notebook_sha256": NOTEBOOK_SHA256,
                    "budgets": [1, 2, 4, 8, 16, 32],
                    "seeds": [42, 123, 2026],
                    "arms": ["scratch", "full_finetune", "adapter_only"]}
    checkpoint = hf_hub_download(
        repo_id=DATASET, repo_type="dataset", filename="checkpoints/domain_fluid.pt",
        revision=protocol["checkpoint_revision"], token=token,
    )
    checkpoint_hash = file_sha256(checkpoint)
    if "checkpoint_sha256" in protocol and checkpoint_hash != protocol["checkpoint_sha256"]:
        raise ValueError("Epoch-59 checkpoint hash changed from the saved protocol")
    if "checkpoint_sha256" not in protocol:
        protocol["checkpoint_sha256"] = checkpoint_hash
        api.upload_file(path_or_fileobj=io.BytesIO(json_bytes(protocol)),
                        path_in_repo=protocol_path, repo_id=DATASET,
                        repo_type="dataset", commit_message="Record Euler Track A protocol")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Code:", protocol["code_sha"])
    print("Checkpoint SHA-256:", checkpoint_hash)
    print("Research path:", PREFIX)
    return api, checkpoint, protocol, token


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Verify the four Euler gamma settings already saved under v2

    This checks the v2 manifests and remote window inventory without reading
    The Well or downloading any source HDF5. Each gamma has 16 trajectories:
    8 train, 4 validation and 4 test, with four five-frame windows per trajectory.
    If a gamma is missing, stop and finish v2 preparation before using v3.
    """)
    return


@app.cell
def _(mo):
    prepare_button = mo.ui.run_button(label="2 · Verify saved v2 gamma windows")
    prepare_button
    return (prepare_button,)


@app.cell
def _(
    DATA_PREFIX,
    DATASET,
    GAMMAS,
    Path,
    SOURCE_REVISION,
    api,
    hf_hub_download,
    json,
    mo,
    prepare_button,
    protocol,
    token,
):
    mo.stop(not prepare_button.value, mo.md("Click **Verify** to check the saved v2 gamma windows."))
    _remote_files = set(api.list_repo_files(DATASET, repo_type="dataset"))
    for gamma in GAMMAS:
        _gamma_prefix = f"{DATA_PREFIX}/data/gamma-{gamma}"
        _gamma_manifest = f"{_gamma_prefix}/manifest.json"
        if _gamma_manifest not in _remote_files:
            raise ValueError(f"Gamma {gamma} is missing from v2; finish v2 preparation first")
        remote = hf_hub_download(DATASET, _gamma_manifest, repo_type="dataset", token=token)
        part = json.loads(Path(remote).read_text())
        expected = {f"{_gamma_prefix}/{r['path']}" for r in part["windows"]}
        if (part["selection"]["gammas"] != [gamma]
                or part["source_revision"] != SOURCE_REVISION
                or part["checkpoint"]["sha256"] != protocol["checkpoint_sha256"]
                or any(r["gamma"] != float(gamma) for r in part["windows"])
                or len(expected) != 64
                or not expected.issubset(_remote_files)):
            raise ValueError(f"Gamma {gamma} remote manifest/windows are incomplete; stop and inspect HF")
        print("VERIFIED", gamma, "64 windows", flush=True)
    print("All four v2 gamma groups are saved and verified. Continue to step 3.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Restore the verified v2 dataset

    Press after all four gammas are verified. This downloads **only** the v2
    prepared windows from your HF repo into the fresh v3 MoLab scratch directory,
    checks their hashes, and verifies the fixed 32/16/16 trajectory split. Repeat this
    restore step in every new MoLab session before training.
    """)
    return


@app.cell
def _(mo):
    restore_button = mo.ui.run_button(label="3 · Restore verified v2 windows")
    restore_button
    return (restore_button,)


@app.cell
def _(
    CODE,
    DATA,
    DATA_PREFIX,
    DATASET,
    GAMMAS,
    Path,
    ROOT,
    api,
    checkpoint,
    file_sha256,
    hf_hub_download,
    json,
    json_bytes,
    mo,
    protocol,
    restore_button,
    snapshot_download,
    subprocess,
    sys,
    token,
):
    mo.stop(not restore_button.value, mo.md("Click **Restore** after verifying all four v2 gammas."))
    _remote_files = set(api.list_repo_files(DATASET, repo_type="dataset"))
    required = {f"{DATA_PREFIX}/data/gamma-{g}/manifest.json" for g in GAMMAS}
    if not required.issubset(_remote_files):
        raise ValueError(f"Prepare these gammas first: {sorted(required - _remote_files)}")
    snapshot_download(repo_id=DATASET, repo_type="dataset", token=token,
                      local_dir=str(ROOT),
                      allow_patterns=[f"{DATA_PREFIX}/data/manifest.json",
                                      f"{DATA_PREFIX}/data/gamma-*/manifest.json",
                                      f"{DATA_PREFIX}/data/gamma-*/windows/*.npy"])
    parts = [json.loads((DATA / f"gamma-{g}" / "manifest.json").read_text()) for g in GAMMAS]
    if any(p["source_revision"] != protocol["source_revision"] or
           p["checkpoint"]["sha256"] != protocol["checkpoint_sha256"] for p in parts):
        raise ValueError("Gamma manifests disagree about source or checkpoint")
    combined = {key: parts[0][key] for key in ("kind", "source_repo", "source_revision", "checkpoint")}
    combined["selection"] = {**parts[0]["selection"], "gammas": list(GAMMAS)}
    combined["windows"] = [{**r, "path": f"gamma-{g}/{r['path']}"}
                           for g, p in zip(GAMMAS, parts) for r in p["windows"]]
    counts = {split: len({(r["source_file"], r["trajectory"])
                          for r in combined["windows"] if r["split"] == split})
              for split in ("train", "validation", "test")}
    if counts != {"train": 32, "validation": 16, "test": 16} or len(combined["windows"]) != 256:
        raise ValueError(f"Unexpected Euler selection: {counts}, {len(combined['windows'])} windows")
    manifest_path = DATA / "manifest.json"
    payload = json_bytes(combined)
    if not manifest_path.exists() or manifest_path.read_bytes() != payload:
        raise ValueError("Saved v2 combined manifest differs from the gamma files")
    subprocess.run([sys.executable, "-c",
                    "import sys; sys.path.insert(0, 'research'); from common import read_manifest, core; read_manifest(sys.argv[1], sys.argv[2], required_checkpoint_exclusion=core.EULER, require_statistics=False)",
                    str(manifest_path), checkpoint], cwd=CODE, check=True)
    print("Ready:", manifest_path, counts, "manifest SHA-256:", file_sha256(manifest_path))
    return (manifest_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Pilot on validation trajectories

    The pilot runs all three arms at budgets 1 and 4 with seed 42. It checks
    runtime, finite loss, checkpoint saving and whether the validation curve
    is still improving at the last step. Pilot results are **not** the held-out
    test claim. The preset is 100 updates; you can use the observed timing to
    choose a longer fixed update budget for the final protocol.
    """)
    return


@app.cell
def _(mo):
    pilot_button = mo.ui.run_button(label="4 · Run validation-only pilot")
    pilot_button
    return (pilot_button,)


@app.cell
def _(
    CODE,
    DATASET,
    PREFIX,
    ROOT,
    api,
    checkpoint,
    hf_hub_download,
    json,
    manifest_path,
    mo,
    pilot_button,
    subprocess,
    sys,
    time,
    token,
):
    mo.stop(not pilot_button.value, mo.md("Click **Run pilot** after restoring the data."))
    if not manifest_path.is_file():
        raise RuntimeError("Run step 3 first in this MoLab session")
    remote_prefix = f"{PREFIX}/runs/pilot-100-steps-seed-42"
    remote_result = f"{remote_prefix}/results.json"
    if remote_result in set(api.list_repo_files(DATASET, repo_type="dataset")):
        result_path = hf_hub_download(DATASET, remote_result, repo_type="dataset", token=token)
        print("Pilot already saved:", result_path)
    else:
        output = ROOT / "local-runs" / f"pilot-{time.time_ns()}"
        subprocess.run([sys.executable, "research/experiment.py",
                        "--manifest", str(manifest_path), "--checkpoint", checkpoint,
                        "--output-dir", str(output), "--device", "cuda",
                        "--trajectory-budgets", "1", "4", "--seeds", "42",
                        "--steps", "100", "--validation-interval", "20",
                        "--evaluate-split", "validation"], cwd=CODE, check=True)
        report = json.loads((output / "results.json").read_text())
        assert len(report["results"]) == 6
        api.upload_folder(folder_path=str(output), path_in_repo=remote_prefix,
                          repo_id=DATASET, repo_type="dataset",
                          commit_message="Save Euler Track A validation pilot")
        print("Saved pilot. Seconds per arm:",
              [(r["method"], r["requested_trajectories"], round(r["training_seconds"]))
               for r in report["results"]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Lock the final protocol

    Decide the number of optimizer updates and the validation-check interval
    using the pilot's timing and validation traces. Use the **same settings for
    every arm, budget and seed**. Once saved, the settings cannot be changed
    under this research path. Choose a new path/version for a different study.
    The test set has not been used by the pilot.
    """)
    return


@app.cell
def _(mo):
    final_steps = mo.ui.number(value=1000, start=1, label="Optimizer updates per job")
    check_every = mo.ui.number(value=100, start=1, label="Validation check every N updates")
    lock_button = mo.ui.run_button(label="5 · Lock final protocol")
    mo.vstack([final_steps, check_every, lock_button])
    return check_every, final_steps, lock_button


@app.cell
def _(
    BUDGETS,
    DATASET,
    METHODS,
    PREFIX,
    Path,
    SEEDS,
    api,
    check_every,
    file_sha256,
    final_steps,
    hf_hub_download,
    io,
    json,
    json_bytes,
    lock_button,
    manifest_path,
    mo,
    protocol,
    token,
):
    mo.stop(not lock_button.value, mo.md("Inspect the pilot first, then lock the final settings."))
    if not manifest_path.is_file():
        raise RuntimeError("Restore the Euler dataset before locking the protocol")
    if f"{PREFIX}/runs/pilot-100-steps-seed-42/results.json" not in set(api.list_repo_files(DATASET, repo_type="dataset")):
        raise RuntimeError("Complete and save the validation pilot in step 4 before locking final settings")
    final_protocol = {"kind": "Delta-V Euler Track A final evaluation",
                      "code_sha": protocol["code_sha"],
                      "checkpoint_sha256": protocol["checkpoint_sha256"],
                      "manifest_sha256": file_sha256(manifest_path),
                      "training_steps": int(final_steps.value),
                      "validation_interval": int(check_every.value),
                      "learning_rate": 1e-4,
                      "budgets": list(BUDGETS), "seeds": list(SEEDS),
                      "methods": list(METHODS),
                      "selection": "best validation MSE; test evaluated once per completed job"}
    if final_protocol["validation_interval"] > final_protocol["training_steps"]:
        raise ValueError("Validation interval cannot exceed training steps")
    path = f"{PREFIX}/final_protocol.json"
    if path in set(api.list_repo_files(DATASET, repo_type="dataset")):
        saved = json.loads(Path(hf_hub_download(DATASET, path, repo_type="dataset", token=token)).read_text())
        if saved != final_protocol:
            raise ValueError("A different final protocol is already locked under this research path")
        print("Final protocol already locked:", saved)
    else:
        api.upload_file(path_or_fileobj=io.BytesIO(json_bytes(final_protocol)),
                        path_in_repo=path, repo_id=DATASET, repo_type="dataset",
                        commit_message="Lock Euler Track A final evaluation protocol")
        print("Locked final protocol:", final_protocol)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Run the final experiment, with session-safe checkpoints

    There are 54 jobs: six budgets × three seeds × three arms. The button runs
    the next unfinished jobs, uploads each job's checkpoint, statistics and
    result to HF, and skips completed jobs on future sessions. Start with one
    job to measure runtime. Choose a larger batch only if it fits within the
    remaining MoLab session. Repeat steps 1 and 3 when you open a new session.
    Each final job uses the same fixed settings and evaluates the test set once.
    """)
    return


@app.cell
def _(mo):
    batch_size = mo.ui.number(value=1, start=1, stop=54, label="Jobs to run this click")
    final_button = mo.ui.run_button(label="6 · Run next unfinished final jobs")
    mo.vstack([batch_size, final_button])
    return batch_size, final_button


@app.cell
def _(
    BUDGETS,
    CODE,
    DATASET,
    METHODS,
    PREFIX,
    Path,
    ROOT,
    SEEDS,
    api,
    batch_size,
    checkpoint,
    file_sha256,
    final_button,
    hf_hub_download,
    json,
    manifest_path,
    mo,
    subprocess,
    sys,
    time,
    token,
):
    mo.stop(not final_button.value, mo.md("Click **Run next jobs** after locking the final protocol."))
    _protocol_path = f"{PREFIX}/final_protocol.json"
    if _protocol_path not in set(api.list_repo_files(DATASET, repo_type="dataset")):
        raise RuntimeError("Lock the final protocol in step 5 first")
    settings = json.loads(Path(hf_hub_download(DATASET, _protocol_path, repo_type="dataset", token=token)).read_text())
    if not manifest_path.is_file() or file_sha256(manifest_path) != settings["manifest_sha256"]:
        raise RuntimeError("Restore the pinned Euler data in step 3 first")
    _actual_code_sha = subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
    if _actual_code_sha != settings["code_sha"]:
        raise RuntimeError("MoLab code checkout differs from the locked final protocol; rerun step 1")
    _existing = set(api.list_repo_files(DATASET, repo_type="dataset"))
    jobs = [(budget, seed, method) for budget in BUDGETS for seed in SEEDS for method in METHODS]
    pending = [(budget, seed, method) for budget, seed, method in jobs
               if f"{PREFIX}/runs/final/budget-{budget}/seed-{seed}/{method}/results.json" not in _existing]
    print(f"Completed: {len(jobs) - len(pending)}/54. Running: {min(int(batch_size.value), len(pending))}")
    for _budget, _seed, _method in pending[:int(batch_size.value)]:
        _remote_prefix = f"{PREFIX}/runs/final/budget-{_budget}/seed-{_seed}/{_method}"
        _output = ROOT / "local-runs" / f"{_method}-{_budget}-{_seed}-{time.time_ns()}"
        print("START", _budget, _seed, _method, flush=True)
        subprocess.run([sys.executable, "research/experiment.py",
                        "--manifest", str(manifest_path), "--checkpoint", checkpoint,
                        "--output-dir", str(_output), "--device", "cuda",
                        "--trajectory-budgets", str(_budget), "--seeds", str(_seed),
                        "--methods", _method,
                        "--steps", str(settings["training_steps"]),
                        "--validation-interval", str(settings["validation_interval"]),
                        "--learning-rate", str(settings["learning_rate"]),
                        "--evaluate-split", "test"], cwd=CODE, check=True)
        _report = json.loads((_output / "results.json").read_text())
        if len(_report["results"]) != 1 or _report["results"][0]["evaluation_split"] != "test":
            raise ValueError("Final job did not produce one held-out result")
        api.upload_folder(folder_path=str(_output), path_in_repo=_remote_prefix,
                          repo_id=DATASET, repo_type="dataset",
                          commit_message=f"Euler Track A {_method} budget {_budget} seed {_seed}")
        _remote = hf_hub_download(DATASET, f"{_remote_prefix}/results.json",
                                 repo_type="dataset", token=token)
        if json.loads(Path(_remote).read_text()) != _report:
            raise ValueError("Uploaded job report did not match local output")
        print("SAVED", _budget, _seed, _method, "seconds", round(_report["results"][0]["training_seconds"]), flush=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Summarize and save the evidence

    Once 54 jobs are saved, press the button. It computes three-seed mean and
    sample standard deviation for Rel. L2, RMSE and SSIM, by field, budget and
    arm. The output is a CSV, a JSON record and a four-panel Rel. L2 chart in
    the private HF research path. It reports measured performance; it does not
    infer a transfer win when the numbers do not support one.
    """)
    return


@app.cell
def _(mo):
    summary_button = mo.ui.run_button(label="7 · Build and save final summary")
    summary_button
    return (summary_button,)


@app.cell
def _(
    BUDGETS,
    DATASET,
    METHODS,
    PREFIX,
    Path,
    ROOT,
    SEEDS,
    api,
    csv,
    file_sha256,
    hf_hub_download,
    json,
    mo,
    statistics,
    summary_button,
    token,
):
    mo.stop(not summary_button.value, mo.md("Summarize after all 54 final jobs are complete."))
    paths = [(budget, seed, method,
              f"{PREFIX}/runs/final/budget-{budget}/seed-{seed}/{method}/results.json")
             for budget in BUDGETS for seed in SEEDS for method in METHODS]
    _remote_files = set(api.list_repo_files(DATASET, repo_type="dataset"))
    missing = [(b, s, m) for b, s, m, path in paths if path not in _remote_files]
    if missing:
        raise ValueError(f"{len(missing)} final jobs remain. First missing: {missing[:5]}")
    rows = []
    for budget, seed, method, _path in paths:
        _report = json.loads(Path(hf_hub_download(DATASET, _path, repo_type="dataset", token=token)).read_text())
        item = _report["results"][0]
        if (item["requested_trajectories"], item["seed"], item["method"], item["evaluation_split"]) != (budget, seed, method, "test"):
            raise ValueError(f"Wrong result identity at {_path}")
        fields = item["metrics"]["equal_trajectory_mean_by_system"]["euler_multi_quadrants_openBC"]
        for field, metrics in fields.items():
            rows.append({"budget": budget, "seed": seed, "method": method,
                         "field": field, "rel_l2": metrics["rel_l2"],
                         "rmse": metrics["rmse"], "ssim": metrics["ssim"],
                         "training_seconds": item["training_seconds"],
                         "trainable_real_parameters": item["trainable_real_parameters"]})
    summary = []
    for budget in BUDGETS:
        for method in METHODS:
            for field in ("scalar", "pressure", "velocity_x", "velocity_y"):
                selected = [r for r in rows if r["budget"] == budget and r["method"] == method and r["field"] == field]
                if len(selected) != 3:
                    raise ValueError("Missing seed in summary")
                entry = {"budget": budget, "method": method, "field": field}
                for metric in ("rel_l2", "rmse", "ssim"):
                    values = [r[metric] for r in selected if r[metric] is not None]
                    entry[f"{metric}_mean"] = statistics.mean(values) if values else None
                    entry[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else None
                entry["training_seconds_mean"] = statistics.mean(r["training_seconds"] for r in selected)
                entry["trainable_real_parameters"] = selected[0]["trainable_real_parameters"]
                summary.append(entry)
    _output = ROOT / "summary"
    _output.mkdir(parents=True, exist_ok=True)
    (_output / "all_seed_results.json").write_text(json.dumps(rows, indent=2))
    (_output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (_output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, field in zip(axes.flat, ("scalar", "pressure", "velocity_x", "velocity_y")):
        for method in METHODS:
            selected = [r for r in summary if r["method"] == method and r["field"] == field]
            axis.errorbar([r["budget"] for r in selected], [r["rel_l2_mean"] for r in selected],
                          yerr=[r["rel_l2_sd"] or 0 for r in selected], marker="o", label=method)
        axis.set_xscale("log", base=2)
        axis.set_title(field)
        axis.set_ylabel("Held-out relative L2")
        axis.set_xticks(BUDGETS, labels=[str(b) for b in BUDGETS])
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.supxlabel("Euler training trajectories")
    figure.suptitle("Delta-V Euler transfer · three seeds · trajectory-held-out test")
    figure.tight_layout()
    figure.savefig(_output / "rel_l2_by_budget.png", dpi=180)
    plt.close(figure)
    _remote_prefix = f"{PREFIX}/summary"
    if f"{_remote_prefix}/summary.json" in _remote_files:
        _existing = hf_hub_download(DATASET, f"{_remote_prefix}/summary.json", repo_type="dataset", token=token)
        if file_sha256(_existing) != file_sha256(_output / "summary.json"):
            raise ValueError("A different summary already exists remotely")
        print("Summary already saved:", _remote_prefix)
    else:
        api.upload_folder(folder_path=str(_output), path_in_repo=_remote_prefix,
                          repo_id=DATASET, repo_type="dataset",
                          commit_message="Publish Euler Track A held-out summary")
        print("Saved summary:", _remote_prefix)
    mo.vstack([mo.md(f"**54 final jobs complete.** Results: `{_remote_prefix}`"), figure])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Track B · Does Delta-V reduce design runs?

    This is a **separate, retrospective outcome-table experiment**. The current
    radiative-layer artifact has only **nine distinct `t_cool` settings** and
    two measured outcomes, `mass_flux` and `turbulent_velocity`. We hold out
    whole cooling-time settings in three predefined folds. Starting from three
    observed settings, a fixed Gaussian-process uncertainty rule selects the
    next setting. All three arms receive the **same observed settings**:

    1. A Gaussian process/kriging outcome predictor.
    2. A small outcome network trained from scratch.
    3. The same network warm-started on the **other two fluid systems' outcome
       tables**, then fitted on the observed radiative settings. This is an
       outcome-table warm start, **not** the epoch-59 FNO's field features.

    Set a target normalized RMSE below. It applies to **both** outcomes and is
    normalized by the non-test design-range within each fold. The output shows
    first observed-run count reaching the target, predicted outcomes, one- and
    two-sigma coverage, and settings avoided relative to measuring all nine.
    Failure to meet the target is reported as failure, not zero saved runs.

    **Limitations:** These nine values are previously published aggregates of
    sampled training frames, not a fresh trajectory-held-out dataset. Avoided
    settings are a retrospective cost proxy, not measured solver savings. The
    warm start does not use FNO features because those would require partial
    simulation fields at every queried design and must be costed separately.
    Do not present this as proven prospective design-run reduction.
    """)
    return


@app.cell
def _(mo):
    b_target = mo.ui.number(value=0.2, start=0.01, step=0.01,
                            label="Maximum normalized RMSE per outcome")
    b_button = mo.ui.run_button(label="8 · Run Track B and save design-run benchmark")
    mo.vstack([b_target, b_button])
    return b_button, b_target


@app.cell
def _(
    GITHUB,
    DATASET,
    Path,
    ROOT,
    TRACK_B_CODE_SHA,
    api,
    b_button,
    b_target,
    file_sha256,
    hf_hub_download,
    json,
    mo,
    subprocess,
    sys,
    time,
    token,
):
    mo.stop(not b_button.value, mo.md("Click **Run Track B** when ready."))
    target = float(b_target.value)
    if not 0 < target < 1:
        raise ValueError("Select a normalized target strictly between 0 and 1")
    remote_path = "research/design-runs-track-b/v1/results.json"
    revision = api.dataset_info(DATASET).sha
    _b_names = ("turbulent_radiative_layer_2D", "rayleigh_benard", "shear_flow")
    _b_paths = [hf_hub_download(DATASET, f"checkpoints/parametric_{name}.npz",
                                repo_type="dataset", revision=revision, token=token)
                for name in _b_names]
    _b_hashes = {name: file_sha256(path) for name, path in zip(_b_names, _b_paths)}
    existing = set(api.list_repo_files(DATASET, repo_type="dataset"))
    if remote_path in existing:
        _b_saved = hf_hub_download(DATASET, remote_path, repo_type="dataset", token=token)
        _b_report = json.loads(Path(_b_saved).read_text())
        if _b_report["protocol"]["target_normalized_rmse_each_outcome"] != target:
            raise ValueError("Track B has a different locked target. Use a new research version for a new study")
        if _b_report["protocol"]["input_sha256"] != _b_hashes:
            raise ValueError("Outcome tables changed since the saved Track B run")
        if _b_report["protocol"].get("code_sha") != TRACK_B_CODE_SHA:
            raise ValueError("Track B result was generated by different code")
        print("Using already saved Track B result:", remote_path)
    else:
        _b_code = ROOT / "track-b-code"
        if not _b_code.exists():
            subprocess.run(["git", "clone", GITHUB, str(_b_code)], check=True)
        subprocess.run(["git", "-C", str(_b_code), "checkout", TRACK_B_CODE_SHA], check=True)
        _b_head = subprocess.check_output(["git", "-C", str(_b_code), "rev-parse", "HEAD"], text=True).strip()
        if _b_head != TRACK_B_CODE_SHA:
            raise ValueError("Track B code checkout differs from the pinned commit")
        _b_output = ROOT / f"track-b-{time.time_ns()}"
        subprocess.run([sys.executable, "research/track_b.py",
                        "--radiative", _b_paths[0], "--rayleigh", _b_paths[1],
                        "--shear", _b_paths[2], "--output-dir", str(_b_output),
                        "--target", str(target)], cwd=_b_code, check=True)
        _b_report = json.loads((_b_output / "results.json").read_text())
        _b_report["protocol"]["artifact_revision"] = revision
        _b_report["protocol"]["code_sha"] = TRACK_B_CODE_SHA
        _b_payload = json.dumps(_b_report, indent=2, allow_nan=False) + "\n"
        (_b_output / "results.json").write_text(_b_payload)
        if remote_path in set(api.list_repo_files(DATASET, repo_type="dataset")):
            raise RuntimeError("Track B result appeared remotely during this run; inspect before uploading")
        api.upload_file(path_or_fileobj=(_b_output / "results.json"),
                        path_in_repo=remote_path, repo_id=DATASET,
                        repo_type="dataset", commit_message="Save Track B design-run benchmark")
        print("Saved Track B:", remote_path)
    lines = ["| Arm | Folds meeting target | First runs by fold | Saved from nine-setting sweep | 68% coverage | 95% coverage |",
             "|---|---:|---|---|---:|---:|"]
    for row in _b_report["summary"]:
        lines.append("| {} | {}/3 | {} | {} | {:.1%} | {:.1%} |".format(
            row["method"], row["folds_meeting_target"],
            row["first_runs_to_target_by_fold"],
            row["runs_saved_from_nine_setting_sweep_by_fold"],
            row["final_68pct_coverage"], row["final_95pct_coverage"]))
    mo.md("\n".join(lines))
    return


if __name__ == "__main__":
    app.run()
