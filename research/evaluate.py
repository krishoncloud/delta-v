"""Evaluate a checkpoint only after trajectory exposure has been documented."""
import argparse
import json
from pathlib import Path
from common import read_manifest, evaluate, core, sha256

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest, args.checkpoint)
    records = [r for r in manifest["windows"] if r["split"] == "test"]
    if set(r["system"] for r in records) != set(core.FLUID_FAMILY):
        raise ValueError("Held-out evaluation requires test trajectories for all three served systems")
    model, epoch = core.load_fno_checkpoint(core.build_fno(args.device), args.checkpoint, args.device)
    result = evaluate(model, records, manifest, args.device)
    result.update(checkpoint_sha256=sha256(args.checkpoint), epoch=epoch, manifest_sha256=sha256(args.manifest))
    with Path(args.output).open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)

if __name__ == "__main__":
    main()
