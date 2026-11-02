import importlib.util
from pathlib import Path

import numpy as np
import torch

MODULE = Path(__file__).parents[1] / "source" / "run_experiment.py"
spec = importlib.util.spec_from_file_location("ma540", MODULE)
ma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ma)


def test_world_has_noncommutative_heldout_pairs_and_no_pair_leakage():
    maps, _, _, pair_data, held = ma.make_world(54001)
    seen_pairs = set(map(tuple, pair_data[0].tolist()))
    assert not seen_pairs.intersection(ma.HELDOUT)
    assert {tuple(row[:2]) for row in held} == set(ma.HELDOUT)
    assert any(not np.array_equal(maps[b, maps[a]], maps[a, maps[b]]) for a, b in ((0, 1), (2, 3)))


def test_internal_hard_unroll_matches_external_two_call():
    _, support, _, _, held = ma.make_world(54002)
    model = ma.Executor("fv_tied")
    ops = torch.as_tensor(held[:, :2])
    states = torch.as_tensor(held[:, 2])
    sup = torch.as_tensor(support)
    internal, _ = model(ops, states, sup, hard_feedback=True)
    external, _ = model.external_two_call(ops[:, 0], ops[:, 1], states, sup)
    assert torch.equal(internal.argmax(-1), external.argmax(-1))
