import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_ma1141", ROOT / "source" / "run_ma1141.py")
ma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ma)


def test_draw_replay_and_pair_partition():
    draw = json.loads((ROOT / "source" / "draw27_exclusions.json").read_text())
    pool_bytes = "\n".join(draw["pool_ids"]).encode()
    assert len(draw["pool_ids"]) == 523
    assert hashlib.sha256(pool_bytes).hexdigest() == draw["pool_sha256"]
    digest = hashlib.sha256(bytes.fromhex(draw["seed_hex"]) + bytes.fromhex(draw["pool_sha256"])).digest()
    index = int.from_bytes(digest, "big") % len(draw["pool_ids"])
    assert index == 512 and draw["pool_ids"][index] == "MA-1141"
    assert len(ma.TRAIN_PAIRS) == len(ma.AUDIT_PAIRS) == 8
    assert set(ma.TRAIN_PAIRS).isdisjoint(ma.AUDIT_PAIRS)
    assert {m for m, _ in ma.TRAIN_PAIRS} == {0, 1, 2, 3}
    assert {s for _, s in ma.TRAIN_PAIRS} == {0, 1, 2, 3}


def test_teacher_action_is_finite_bounded_and_has_four_patterns():
    actions = [ma.teacher_action(0, s, .7) for s in range(4)]
    assert all(a.shape == (8,) and np.isfinite(a).all() for a in actions)
    assert all(float(np.max(np.abs(a))) <= 1 for a in actions)
    assert len({tuple(np.round(a, 7)) for a in actions}) == 4


def test_policy_shapes_and_condition_code_gradients():
    dim = 31
    x = torch.randn(12, dim)
    morph, skill = torch.arange(12) % 4, torch.arange(12).flip(0) % 4
    for method in ma.METHODS:
        ma.set_seed(31)
        model = ma.Policy(method, dim)
        out = model(x, morph, skill)
        assert out.shape == (12, 8)
        assert torch.isfinite(out).all()
        if method in ("film", "mirror"):
            out.square().mean().backward()
            assert model.morph_codes.grad is not None
            assert model.morph_codes.grad.abs().sum().item() > 0


def test_all_pinned_morphology_models_load():
    for m in range(4):
        env = ma.make_env(m)
        obs, _ = env.reset(seed=123 + m)
        assert env.observation_space.contains(obs)
        assert env.action_space.shape == (8,)
        env.close()


def test_serialized_payload_accounting_is_actual_file_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "ART", tmp_path)
    model = ma.Policy("mirror", 31)
    server, codes, pair, total = ma.model_state_bytes(model, "mirror", "fixture")
    assert server == (tmp_path / "fixture_server.safetensors").stat().st_size + (tmp_path / "fixture_metadata.json").stat().st_size
    assert codes == (tmp_path / "fixture_codes.safetensors").stat().st_size
    assert pair == (tmp_path / "fixture_pair.safetensors").stat().st_size
    assert total == server + codes
