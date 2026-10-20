"""Download the website's data files from the Hugging Face dataset, pinned and checksummed.

    python scripts/fetch_site_files.py --examples DIR --downloads DIR

The worked-example bundles are fetched byte for byte instead of being rebuilt in the
image: rebuilding regenerates starting fields with whatever numpy/scipy and CPU the
build machine has, which can change the bytes and break the pinned SHA-256. The sample
study is fetched the same way. Every file is checked before it is written.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "KrishMalik/deltav-fluid"
REVISION = "18fe11d88050f40a556ab4925cb669c18527dac5"
FILES = {
    "deltav-task8-r701-n16-v1.zip": ("examples", "629f86a4c2f0235596e1adc4f3c591ce856432865279e8483e1296b33aa1ef3b"),
    "deltav-task8-r701-recorded-pool-v1.zip": ("examples", "f41128419a89de536559b1e815478fa0e620da21eb2dd9a35fc515ce9235868a"),
    "delta-v-sample-study.zip": ("downloads", "8fd53267fd20534aafe81c626b11af57238f8ac021750aff73427ac73b2885bf"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--examples", type=Path, required=True)
    ap.add_argument("--downloads", type=Path, required=True)
    args = ap.parse_args()
    dest = {"examples": args.examples, "downloads": args.downloads}
    for name, (kind, sha) in FILES.items():
        path = Path(hf_hub_download(REPO, f"website/{name}", repo_type="dataset", revision=REVISION, token=False))
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != sha:
            raise SystemExit(f"{name}: SHA-256 {got} does not match the pinned {sha}")
        dest[kind].mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest[kind] / name)
        print(f"{name}: verified")


if __name__ == "__main__":
    main()
