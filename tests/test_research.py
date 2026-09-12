import sys
from pathlib import Path
import pytest
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import common
from split import split_records
from common import trajectory_key
from experiment import AdaptedFNO, count_parameters

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
