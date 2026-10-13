"""Small leakage and shape check for the retrospective design benchmark."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from track_b import SYSTEMS, run


def test_track_b_keeps_test_settings_out_of_acquisition():
    x = np.geomspace(0.03, 3.16, 9)
    tables = {
        SYSTEMS[0]: (x[:, None], np.column_stack((np.log(x), np.sqrt(x)))),
        SYSTEMS[1]: (np.column_stack((x, np.ones(9))), np.column_stack((np.log(x), np.sqrt(x)))),
        SYSTEMS[2]: (np.column_stack((x, np.ones(9))), np.column_stack((np.log(x), np.sqrt(x)))),
    }
    result = run(tables, target=0.2, steps=2, source_steps=2)
    assert len(result["records"]) == 45
    for record in result["records"]:
        assert set(record["heldout_indices"]).isdisjoint(record["acquired_indices"])
        assert len(record["acquired_indices"]) == record["runs"]
        assert np.isfinite(record["predicted"]).all()
        assert np.isfinite(record["uncertainty_1sigma"]).all()
    assert len(result["summary"]) == 3
