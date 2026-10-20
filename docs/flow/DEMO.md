# Public demo (no GPU cost)

What visitors can do on the Hugging Face Space:

- Read about the supported study type and upload their own study ZIP (it is checked).
- See Adapt as a step. It is switched off (`FLOW_ADAPT_ENABLED=0`) because it needs a GPU.
- Open the worked example, which comes with a model adapted in advance, and use
  Explore, Next runs, add recorded results, and the Report. Predictions run on the
  Space's CPU (`FLOW_EMBEDDED_WORKER=1`).

## 1. Adapt the example once on MoLab

1. Import `research/task10_11/DeltaV_ExampleAdapt_MoLab.py` and pick a GPU runtime.
2. Click **Rebuild and verify the example bundle**, then **Adapt the FNO**.
3. Download the 5 files: `deltav-example-model-metadata.zip`, `model_manifest.json`
   and three `member_<seed>__weights.pt`.

## 2. Check them locally

```bash
.venv/bin/python scripts/import_example_model.py --files <folder with the 5 files>
```

This verifies every file and reloads the model. Run the site locally with
`FLOW_ADAPT_ENABLED=0 FLOW_EMBEDDED_WORKER=1` to rehearse.

## 3. Host the files (free)

Upload the 5 files to your existing dataset repo, for example:

```bash
huggingface-cli upload KrishMalik/deltav-fluid <folder> preadapted/task8-r701-n16-v1 --repo-type dataset
```

Note the commit revision it prints and the SHA-256 of `model_manifest.json`.

## 4. Space settings

| Variable | Value |
|---|---|
| `FLOW_PREADAPTED_REPO` | `KrishMalik/deltav-fluid` |
| `FLOW_PREADAPTED_PATH` | `preadapted/task8-r701-n16-v1` |
| `FLOW_PREADAPTED_REVISION` | the upload's commit hash |
| `FLOW_PREADAPTED_MANIFEST_SHA256` | SHA-256 of `model_manifest.json` |

The Dockerfile already sets demo mode and builds the example bundles from the
archive parts in the repo. At startup the Space downloads the model (about 860 MB),
checks it, and attaches it to every opened worked example.

## Notes

- The free Space's disk is temporary: visitors' studies vanish when it restarts.
- Pushing to the Space publishes this code. Task 15 (public/private split) is your call.
- Not yet verified: the Docker build itself (Docker was not running locally).
- The MoLab model is installed locally and verified (see VALIDATION.md). Its
  `model_manifest.json` SHA-256 is `b4ecbaf8a68e0ae085e0c2d76a1f1c46a068f989eb17a679b7a4bb01faf19f8d`.
