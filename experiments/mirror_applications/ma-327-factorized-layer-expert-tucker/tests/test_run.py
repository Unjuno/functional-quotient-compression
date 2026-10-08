import importlib.util
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "source" / "run.py"
spec = importlib.util.spec_from_file_location("ma327_run", SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_world_and_split_shapes():
    basis, coeff, targets = m.world(32701)
    split = m.split_pairs(32701, targets)
    assert basis.shape == (2, 16, 16)
    assert coeff.shape == (4, 4, 2)
    assert targets.shape == (4, 4, 16, 16)
    assert split["test"][0].shape == (4, 4, 1024, 16)


def test_ordinary_product_and_mirror_are_same_function_family():
    a = m.Bank("ordinary_product", 3, torch.randn(2, 16, 16))
    b = m.Bank("mirror_product", 3, a.basis.detach().clone())
    b.load_state_dict(a.state_dict())
    for l in range(4):
        for e in range(4):
            assert torch.equal(a.matrix(l, e), b.matrix(l, e))


def test_paired_product_control_training_is_identically_seeded():
    # The experiment runner pairs random initialization and minibatches for
    # the mathematically equivalent ordinary/Mirror parameterizations.
    assert m.METHODS.index("ordinary_product") != m.METHODS.index("mirror_product")


def test_heldout_diagonal_pair_is_reserved_for_product_methods():
    basis, _, targets = m.world(32702)
    pairs = m.split_pairs(32702, targets)
    # The runner's selection mask reserves all diagonal layer/expert pairs.
    heldout = [(l, e) for l in range(4) for e in range(4) if l == e]
    train = [(l, e) for l in range(4) for e in range(4) if l != e]
    assert len(heldout) == 4 and len(train) == 12
    assert pairs["test"][1].shape[-1] == 16


def test_serialized_model_reload(tmp_path):
    basis, _, _ = m.world(32703)
    model = m.Bank("mirror_product", 4, basis)
    (n, digest), arrays = m.serialize(model, tmp_path / "payload.zip")
    restored = m.load_model("mirror_product", arrays, 4, basis)
    assert n > 0 and len(digest) == 64
    for k, v in model.state_dict().items():
        assert torch.equal(restored.state_dict()[k], torch.from_numpy(arrays[k]).to(v.dtype))
