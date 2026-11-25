import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "source"))
from run_experiment import GatedLoRALinear, ROLES, load_series, role_data


def test_identity_coordinates_preserve_native_gated_lora_function():
    torch.manual_seed(10)
    x = torch.randn(3, 5, 8)
    for view in ("shared", "mirror", "diag"):
        base = nn.Linear(8, 6)
        m = GatedLoRALinear(base, n_roles=4, view=view)
        m.active_role = 2
        actual = m(x)
        expected = base(x) + m.gate * m.b(m.a(x))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if view != "shared":
            assert m.code.shape == (4, 2)
            assert torch.count_nonzero(m.code).item() == 0


def test_train_validation_audit_forecast_targets_respect_time_boundaries():
    series, split = load_series()
    for ds in ("ETTh1", "ETTh2"):
        for horizon in (192, 720):
            role = (ds, horizon)
            _, _, _, train_origins = role_data(series, split, role, "train")
            _, _, _, dev_origins = role_data(series, split, role, "dev")
            _, _, _, audit_origins = role_data(series, split, role, "audit")
            assert train_origins[-1] + 512 + horizon <= split[ds]["train"]
            assert dev_origins[0] >= split[ds]["train"]
            assert dev_origins[-1] + horizon <= split[ds]["val"]
            assert audit_origins[0] >= split[ds]["val"]
            assert audit_origins[-1] + horizon <= split[ds]["test"]
            assert len(dev_origins) == (8 if horizon == 192 else 7)
            if horizon == 720:
                assert len(audit_origins) == 4


def test_frozen_protocol_pins_native_trace_controls_and_bytes():
    protocol = json.loads((HERE / "PROTOCOL.json").read_text())
    assert protocol["experiment_id"] == "MA-1052"
    assert protocol["hypothesis"]
    assert protocol["model"]["selection"].startswith("TRACE Gated DSIC")
    assert any("Native TRACE" in c for c in protocol["controls"])
    assert protocol["storage"]["authority"] == "actual serialized inference payload bytes"
    assert len(ROLES) == 4
