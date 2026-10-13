"""Offline Euler transfer comparison. Requires explicit prepared data and provenance."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from common import read_manifest, trajectory_key, tensors, fit_training_stats, evaluate, core, sha256

class AdaptedFNO(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.input = torch.nn.Conv2d(16, 16, 1)
        self.output = torch.nn.Conv2d(4, 4, 1)
        for layer in (self.input, self.output):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            with torch.no_grad():
                for i in range(layer.in_channels):
                    layer.weight[i, i, 0, 0] = 1

    def forward(self, x):
        return self.output(self.backbone(self.input(x)))

def count_parameters(model, trainable=False):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if not trainable or p.requires_grad)

def nested_trajectory_subsets(records, budgets):
    groups = {}
    for record in records:
        groups.setdefault(record["source_file"], set()).add(trajectory_key(record))
    queues = [sorted(group, key=lambda k: hashlib.sha256(k.encode()).hexdigest()) for _, group in sorted(groups.items())]
    keys = [queue[i] for i in range(max(map(len, queues))) for queue in queues if i < len(queue)]
    if not budgets or any(b < 1 for b in budgets) or budgets != sorted(set(budgets)):
        raise ValueError("Trajectory budgets must be unique positive integers in ascending order")
    if budgets[-1] > len(keys):
        raise ValueError(f"Largest trajectory budget {budgets[-1]} exceeds {len(keys)} available training trajectories")
    return keys, {budget: set(keys[:budget]) for budget in budgets}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--validation-interval", type=int, default=100)
    p.add_argument("--evaluate-split", choices=("validation", "test"), default="validation")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--trajectory-budgets", type=int, nargs="+", default=[1,2,4,8,16,32])
    args = p.parse_args()
    if args.steps < 1 or args.validation_interval < 1 or args.learning_rate <= 0 or len(set(args.seeds)) != len(args.seeds):
        p.error("Use positive steps, validation interval and learning rate, plus unique seeds")
    manifest = read_manifest(args.manifest, args.checkpoint, required_checkpoint_exclusion=core.EULER, require_statistics=False)
    if any(json.loads(k)[0] == core.EULER for k in manifest["checkpoint"]["exposed_trajectories"]):
        raise ValueError("The backbone has seen Euler; cannot label this an unseen-system experiment")
    records = [r for r in manifest["windows"] if r["system"] == core.EULER]
    splits = {s:[r for r in records if r["split"] == s] for s in ("train","validation","test")}
    if not all(splits.values()):
        raise ValueError("Euler requires nonempty, disjoint trajectory splits")
    keys, subsets = nested_trajectory_subsets(splits["train"], args.trajectory_budgets)
    root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=False)
    report = []
    for budget, selected in subsets.items():
        training = [r for r in splits["train"] if trajectory_key(r) in selected]
        stats = fit_training_stats(training)
        stats_path = root / f"statistics-{budget}-trajectories.npz"
        np.savez(stats_path, **stats)
        for seed in args.seeds:
            for method in ("scratch","full_finetune","adapter_only"):
                torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
                model = core.build_fno(args.device)
                if method != "scratch":
                    model, _ = core.load_fno_checkpoint(model, args.checkpoint, args.device)
                if method == "adapter_only":
                    model = AdaptedFNO(model).to(args.device)
                optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
                best = float("inf"); best_state = None; best_step = None
                started = time.perf_counter()
                order = list(training); random.shuffle(order)
                for step in range(1, args.steps + 1):
                    model.train()
                    record = order[(step - 1) % len(order)]
                    x,y = tensors(record, stats); optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(model(x.unsqueeze(0).to(args.device))[0],y.to(args.device))
                    loss.backward(); optimizer.step()
                    if step % len(order) == 0: random.shuffle(order)
                    if step % args.validation_interval and step != args.steps: continue
                    model.eval(); validation = []
                    with torch.no_grad():
                        for record in splits["validation"]:
                            x,y=tensors(record,stats)
                            validation.append(float(torch.nn.functional.mse_loss(model(x.unsqueeze(0).to(args.device))[0],y.to(args.device))))
                    score=float(np.mean(validation))
                    if score < best:
                        best=score; best_step=step; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if args.device.startswith("cuda"): torch.cuda.synchronize()
                elapsed=time.perf_counter()-started
                if best_state is None: raise ValueError("No finite validation checkpoint")
                model.load_state_dict(best_state)
                result={"method":method,"requested_trajectories":budget,"actual_trajectories":len(selected),"total_training_trajectories":len(keys),"selected_trajectories":sorted(selected),"statistics_sha256":sha256(stats_path),"normalization":"fitted on this budget's training windows only; identical across arms and seeds","seed":seed,"training_steps":args.steps,"best_validation_step":best_step,"best_validation_mse":best,"evaluation_split":args.evaluate_split,"learning_rate":args.learning_rate,"training_seconds":elapsed,"trainable_real_parameters":count_parameters(model,True),"total_real_parameters":count_parameters(model),"metrics":evaluate(model,splits[args.evaluate_split],manifest,args.device,stats)}
                report.append(result)
                torch.save({"model":best_state,"method":method,"step":best_step,"seed":seed,"training_trajectories":sorted(selected)},root/f"{method}-{budget}-trajectories-seed-{seed}.pt")
                (root/"results.json").write_text(json.dumps({"checkpoint_sha256":sha256(args.checkpoint),"manifest_sha256":sha256(args.manifest),"results":report},indent=2,allow_nan=False))
                del model, optimizer, best_state
    print("Completed",len(report),"experiment cells in",root)

if __name__ == "__main__": main()
