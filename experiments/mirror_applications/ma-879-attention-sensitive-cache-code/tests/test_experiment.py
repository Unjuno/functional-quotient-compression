import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SOURCE = Path(__file__).parents[1] / "source" / "experiment.py"
spec = importlib.util.spec_from_file_location("ma879_experiment", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_torch_target_rope_matches_numpy_transform():
    cfg = SimpleNamespace(rope_theta=100000.0, model_type="llama")
    x = np.random.default_rng(3).normal(size=(7, 64)).astype(np.float32)
    actual = module.torch_rope_target(torch.tensor(x), cfg).detach().numpy()
    expected = module.rope_transform(x, cfg).astype(np.float32)
    assert np.max(np.abs(actual - expected)) < 2e-6


def test_differentiable_fixed_support_map_shapes_and_gradient():
    rng = np.random.default_rng(5)
    source = []
    for _ in range(2):
        source.append((rng.normal(size=(8, 5, 64)), rng.normal(size=(8, 5, 64))))
    refs, mats = [], []
    for th, sh in enumerate((0, 4, 7)):
        for kv in (0, 1):
            refs.append((0, th, kv, 1, sh))
            mat = np.zeros((65, 64), dtype=np.float32)
            mat[:64] = np.eye(64, dtype=np.float32)
            mats.append(mat)
    codes = torch.nn.Parameter(torch.ones((6, 4), dtype=torch.float32))
    mean = torch.zeros(65 * 64)
    basis = torch.zeros((4, 65 * 64))
    for i in range(4):
        basis[i, i] = 1
    mapped = module.maps_from_codes(codes, mean, basis)
    cfg_s = SimpleNamespace(model_type="gpt_neox", rotary_pct=1.0, hidden_size=64,
                            num_attention_heads=1, rope_theta=10000.0)
    cfg_t = SimpleNamespace(model_type="llama", rope_theta=100000.0,
                            num_hidden_layers=1, num_key_value_heads=3)
    past = module.apply_maps_torch(source, refs, mapped, cfg_s, cfg_t, 4)
    assert past[0][0].shape == (1, 3, 4, 64)
    assert past[0][1].shape == (1, 3, 4, 64)
    sum(k.sum() + v.sum() for k, v in past).backward()
    assert codes.grad is not None and torch.isfinite(codes.grad).all()
    assert float(codes.grad.abs().sum()) > 0


def test_payload_serializer_accounts_for_all_code_variants():
    fixed = {"maps": np.zeros((180, 65, 64), dtype=np.float64),
             "refs": [(layer, h, kv, layer % 2, (0, 4, 7)[h]) for layer in range(30) for h in range(3) for kv in (0, 1)]}
    mean = np.zeros(65 * 64)
    basis = np.zeros((32, 65 * 64))
    codes = np.zeros((180, 32))
    sizes = module.serialize_ma879(fixed, mean, basis, codes, codes.copy())
    assert sizes["pca_rank32"] == sizes["nll_optimized_rank32"]
    assert sizes["full_weighted_ridge"] > sizes["pca_rank32"]
