import importlib.util
from pathlib import Path

import numpy as np
import torch

MODULE = Path(__file__).parents[1] / "source" / "run_experiment.py"
spec = importlib.util.spec_from_file_location("ma545", MODULE)
ma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ma)


def test_support_calibration_and_audit_splits_are_disjoint():
    for tid in range(ma.N_TASKS):
        support, calibration, audit = ma.split_task(tid, 54501)
        sx = {x for x, _ in support}; cx = {x for x, _ in calibration}; ax = {x for x, _ in audit}
        assert len(sx) == 8 and len(cx) == 4 and len(ax) == 4
        assert sx.isdisjoint(cx) and sx.isdisjoint(ax) and cx.isdisjoint(ax)
        assert len(sx | cx | ax) == 16


def test_top_two_router_mixes_only_selected_codes_and_normalizes():
    vectors = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    protos = np.eye(4, dtype=np.float32)[:, :3]
    tid, a, b, weights, mixed, _ = ma.route(np.array([1, 0, 0], np.float32), protos, vectors)
    assert tid == 0 and (a, b) == (0, 1)
    assert np.isclose(float(weights.sum()), 1.0)
    assert np.allclose(mixed, weights[0] * vectors[a] + weights[1] * vectors[b])


def test_position_gated_bias_delta_equals_activation_addition():
    torch.manual_seed(5)
    x = torch.randn(2, 4, 6)
    w = torch.randn(6, 6)
    b = torch.randn(6)
    delta = torch.randn(6)
    pos = 2
    mlp = torch.nn.Linear(6, 6)
    with torch.no_grad():
        mlp.weight.copy_(w); mlp.bias.copy_(b)
    normal = x + mlp(x)
    act = normal.clone(); act[:, pos, :] += delta
    # Equivalent to a per-task output-bias expert whose router gates only the
    # selected query position in the layer output projection.
    def add_at_pos(module, args, out):
        y = out.clone(); y[:, pos, :] += delta; return y
    handle = mlp.register_forward_hook(add_at_pos)
    try:
        bias_expert = x + mlp(x)
    finally:
        handle.remove()
    assert torch.allclose(act, bias_expert, atol=1e-6, rtol=1e-6)
