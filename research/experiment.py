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

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--fractions", type=float, nargs="+", default=[.01,.05,.1,.25,.5,1.0])
    args = p.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0 or any(f <= 0 or f > 1 for f in args.fractions):
        p.error("Use positive epochs/rate and fractions in (0,1]")
    manifest = read_manifest(args.manifest, args.checkpoint)
    if any(json.loads(k)[0] == core.EULER for k in manifest["checkpoint"]["exposed_trajectories"]):
        raise ValueError("The backbone has seen Euler; cannot label this an unseen-system experiment")
    records = [r for r in manifest["windows"] if r["system"] == core.EULER]
    splits = {s:[r for r in records if r["split"] == s] for s in ("train","validation","test")}
    if not all(splits.values()):
        raise ValueError("Euler requires nonempty, disjoint trajectory splits")
    keys = sorted({trajectory_key(r) for r in splits["train"]}, key=lambda k:hashlib.sha256((str(args.seed)+k).encode()).hexdigest())
    root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=False)
    report = []
    for fraction in args.fractions:
        selected = set(keys[:max(1, int(len(keys)*fraction))])
        training = [r for r in splits["train"] if trajectory_key(r) in selected]
        stats = fit_training_stats(training)
        stats_path = root / f"statistics-{fraction}.npz"
        np.savez(stats_path, **stats)
        for method in ("scratch","full_finetune","adapter_only"):
            torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
            model = core.build_fno(args.device)
            if method != "scratch":
                model, _ = core.load_fno_checkpoint(model, args.checkpoint, args.device)
            if method == "adapter_only":
                model = AdaptedFNO(model).to(args.device)
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
            best = float("inf"); best_state = None; best_epoch = None
            started = time.perf_counter()
            for epoch in range(args.epochs):
                model.train(); order = list(training); random.shuffle(order)
                for record in order:
                    x,y = tensors(record, stats); optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(model(x.unsqueeze(0).to(args.device))[0],y.to(args.device))
                    loss.backward(); optimizer.step()
                model.eval(); validation = []
                with torch.no_grad():
                    for record in splits["validation"]:
                        x,y=tensors(record,stats)
                        validation.append(float(torch.nn.functional.mse_loss(model(x.unsqueeze(0).to(args.device))[0],y.to(args.device))))
                score=float(np.mean(validation))
                if score < best:
                    best=score; best_epoch=epoch+1; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if args.device.startswith("cuda"): torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            if best_state is None: raise ValueError("No finite validation checkpoint")
            model.load_state_dict(best_state)
            result={"method":method,"requested_fraction":fraction,"actual_trajectories":len(selected),"total_training_trajectories":len(keys),"selected_trajectories":sorted(selected),"statistics_sha256":sha256(stats_path),"normalization":"fitted on this fraction's training windows only; identical across arms","seed":args.seed,"epochs":args.epochs,"best_validation_epoch":best_epoch,"learning_rate":args.learning_rate,"training_seconds":elapsed,"trainable_real_parameters":count_parameters(model,True),"total_real_parameters":count_parameters(model),"metrics":evaluate(model,splits["test"],manifest,args.device,stats)}
            report.append(result)
            torch.save({"model":best_state,"method":method,"epoch":best_epoch},root/f"{method}-{fraction}.pt")
            (root/"results.json").write_text(json.dumps({"checkpoint_sha256":sha256(args.checkpoint),"manifest_sha256":sha256(args.manifest),"results":report},indent=2,allow_nan=False))
            del model, optimizer, best_state
    print("Completed",len(report),"experiment cells in",root)

if __name__ == "__main__": main()
