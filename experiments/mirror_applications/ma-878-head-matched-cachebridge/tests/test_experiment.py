import importlib.util
from pathlib import Path
import numpy as np

path = Path(__file__).parents[1] / "source" / "experiment.py"
spec = importlib.util.spec_from_file_location("ma878_experiment", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _stat(error):
    sxx = np.eye(mod.D + 1)
    sxy = np.zeros((mod.D + 1, mod.D))
    sxy[:mod.D, :] = np.eye(mod.D)
    syy = np.ones(mod.D) * (1.0 + error)
    return [sxx, sxy, syy, np.array(float(mod.D))]


def test_attention_importance_is_positive_and_normalized():
    attn = np.zeros((12, 5, 5), dtype=np.float64)
    for q in range(5):
        attn[:, q, :q+1] = 1.0 / (q + 1)
    w = mod.target_key_importance(attn, kv_head=1, q_per_kv=4)
    assert w.shape == (5,)
    assert np.all(w > 0)
    assert abs(float(w.mean()) - 1.0) < 1e-12


def test_attention_head_assignment_is_injective_and_selects_low_error():
    stats = {}
    for th, best in ((0, 2), (1, 0)):
        for sh in range(3):
            err = 0.0 if sh == best else 0.4 + 0.1 * sh
            for kv in (0, 1):
                stats[(0, th, sh, kv)] = _stat(err)
    assignment = mod.choose_matched_heads(stats, target_layers=1, target_heads=2, source_heads=3)
    assert assignment == [[2, 0]]
    assert len(set(assignment[0])) == 2


def test_rope_strip_and_target_reapply_roundtrip():
    from types import SimpleNamespace
    source = SimpleNamespace(model_type="gpt_neox", hidden_size=64, num_attention_heads=1, rotary_pct=.25, rope_theta=10000)
    target = SimpleNamespace(model_type="llama", hidden_size=64, num_attention_heads=1, rope_theta=100000)
    x = np.random.default_rng(878).normal(size=(21, 64))
    source_key = mod.rope_transform(x, source)
    position_free = mod.rope_transform(source_key, source, inverse=True)
    target_key = mod.rope_transform(position_free, target)
    restored = mod.rope_transform(target_key, target, inverse=True)
    assert np.max(np.abs(restored - x)) < 1e-12
