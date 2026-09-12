"""Create deterministic trajectory-grouped splits for a NEW training run."""
import argparse
import hashlib
import json
from pathlib import Path
from common import trajectory_key

def split_records(records, seed=42):
    systems = {}
    for r in records:
        systems.setdefault(r["system"],set()).add(trajectory_key(r))
    assignment = {}
    for system, keys in systems.items():
        if len(keys)<3: raise ValueError("Need at least three trajectories per system")
        ordered=sorted(keys,key=lambda k:hashlib.sha256((str(seed)+k).encode()).hexdigest())
        n_test=max(1,int(len(keys)*.15)); n_val=max(1,int(len(keys)*.15))
        for i,key in enumerate(ordered):
            assignment[key]="test" if i<n_test else "validation" if i<n_test+n_val else "train"
    return [{**r,"split":assignment[trajectory_key(r)]} for r in records]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("inventory",help="JSON object with windows containing source/trajectory provenance")
    p.add_argument("output"); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); document=json.loads(Path(a.inventory).read_text())
    document["windows"]=split_records(document["windows"],a.seed)
    document["split_policy"]={"unit":"source trajectory","seed":a.seed,"target_fractions":[.7,.15,.15],"scope":"new training run; cannot retroactively make the existing checkpoint unseen"}
    with Path(a.output).open("x") as f: json.dump(document,f,indent=2)

if __name__=="__main__": main()
