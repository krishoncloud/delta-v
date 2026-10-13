import sys
from pathlib import Path
import pytest
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import common
from split import split_records
from common import euler_trajectory_plan, trajectory_key
from experiment import AdaptedFNO, count_parameters, nested_trajectory_subsets

def test_split_keeps_every_window_of_a_trajectory_together():
    records=[dict(system="shear_flow",source_revision="fixed",source_file="train.h5",trajectory=i,start_index=t) for i in range(20) for t in (0,1,2)]
    out=split_records(records)
    memberships={}
    for r in out: memberships.setdefault(trajectory_key(r),set()).add(r["split"])
    assert all(len(s)==1 for s in memberships.values())
    assert {r["split"] for r in out}=={"train","validation","test"}
    assert {trajectory_key(r):r["split"] for r in out}=={trajectory_key(r):r["split"] for r in split_records(list(reversed(records)))}
    with pytest.raises(ValueError): split_records(records[:3])

def test_adapter_starts_identical_and_freezes_backbone_but_receives_gradients():
    base=torch.nn.Conv2d(16,4,1)
    model=AdaptedFNO(base)
    x=torch.randn(1,16,8,8)
    torch.testing.assert_close(model(x),base(x))
    assert count_parameters(model,True)==292
    model(x).square().mean().backward()
    assert all(p.grad is None for p in base.parameters())
    assert model.input.weight.grad is not None and model.output.weight.grad is not None

def test_fraction_statistics_use_training_only(monkeypatch):
    monkeypatch.setattr(common, "physical_frames", lambda r: [torch.full((4,2,2),r["value"])])
    stats=common.fit_training_stats([{"split":"train","value":1.0},{"split":"train","value":3.0}])
    np.testing.assert_allclose(stats["mean"],2.0)
    np.testing.assert_allclose(stats["std"],1.0)
    with pytest.raises(ValueError,match="training windows only"):
        common.fit_training_stats([{"split":"test","value":10.0}])

def test_trajectory_budgets_are_nested_and_seed_independent():
    records = [dict(system="euler_multi_quadrants_openBC", source_revision="fixed", source_file=f"gamma-{gamma}.h5", trajectory=i) for gamma in range(4) for i in range(10)]
    keys, subsets = nested_trajectory_subsets(records, [1,2,4,8,16,32])
    assert len(keys) == 40
    assert [len(subsets[b]) for b in subsets] == [1,2,4,8,16,32]
    assert all(subsets[a] < subsets[b] for a,b in zip([1,2,4,8,16],[2,4,8,16,32]))
    assert nested_trajectory_subsets(list(reversed(records)), [1,2,4,8,16,32])[1] == subsets
    with pytest.raises(ValueError, match="exceeds"):
        nested_trajectory_subsets(records, [41])
    with pytest.raises(ValueError, match="unique positive"):
        nested_trajectory_subsets(records, [2,1])

def test_euler_plan_is_reproducible_and_trajectory_separated():
    first = euler_trajectory_plan("gamma.h5", 16)
    assert first == euler_trajectory_plan("gamma.h5", 16)
    assert len(first) == len({trajectory for trajectory, _ in first}) == 16
    assert {split: sum(s == split for _, s in first) for split in ("train", "validation", "test")} == {"train": 8, "validation": 4, "test": 4}
