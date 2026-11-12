import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SOURCE = Path(__file__).parents[1] / 'source' / 'experiment.py'
spec = importlib.util.spec_from_file_location('ma880_experiment', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _configs():
    return {name: SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2, num_attention_heads=2,
                                  num_key_value_heads=2, hidden_size=128))
            for name in module.MODELS}


def _maps(pair, rng, models):
    refs = module.map_refs(models[pair[0]].config, models[pair[1]].config)
    return {'pair': pair, 'refs': refs, 'maps': rng.normal(size=(len(refs), 65, 64)), 'samples': 100}


def test_role_address_is_bounded_and_kv_distinct():
    models = _configs()
    ref = (1, 1, 0, 0, 0)
    key = module.role_index(ref, models[module.MODELS[1]].config)
    value = module.role_index((1, 1, 1, 0, 0), models[module.MODELS[1]].config)
    assert 0 <= key < module.ROLE_COUNT
    assert value == key + 1


def test_rope_inverse_roundtrip_and_absolute_position_model_noop():
    cfg = SimpleNamespace(model_type='llama', rope_theta=10000.0, hidden_size=64,
                          num_attention_heads=1)
    x = np.random.default_rng(31).normal(size=(9, 64))
    np.testing.assert_allclose(module.rope_transform(module.rope_transform(x, cfg), cfg, inverse=True),
                               x, rtol=1e-12, atol=1e-12)
    gpt2 = SimpleNamespace(model_type='gpt2')
    np.testing.assert_array_equal(module.rope_transform(x, gpt2), x)


def test_factorized_pair_maps_reconstruct_deterministically():
    rng = np.random.default_rng(19)
    models = _configs()
    train = [_maps(pair, rng, models) for pair in module.TRAIN_PAIRS]
    mean, basis, _ = module.fit_basis(train)
    factor = module.fit_pair_factor(train, mean, basis, 4, 2, 8, models, 13)
    pair = module.FRESH_PAIRS[0]
    refs = module.map_refs(models[pair[0]].config, models[pair[1]].config)
    pred1 = module.predict_pair_maps(pair, factor, mean, basis, refs, models)
    pred2 = module.predict_pair_maps(pair, factor, mean, basis, refs, models)
    assert pred1.shape == (len(refs), 65, 64)
    assert factor['code_mean'].shape == (4,)
    assert factor['code_scale'].shape == (4,)
    assert (factor['code_scale'] > 0).all()
    assert np.isfinite(pred1).all()
    np.testing.assert_array_equal(pred1, pred2)


def test_serializers_charge_state_and_all_full_map_refs():
    rng = np.random.default_rng(23)
    models = _configs()
    train = [_maps(pair, rng, models) for pair in module.TRAIN_PAIRS]
    mean, basis, _ = module.fit_basis(train)
    factor = module.fit_pair_factor(train, mean, basis, 4, 2, 4, models, 7)
    outputs = []
    for pair in module.FRESH_PAIRS:
        refs = module.map_refs(models[pair[0]].config, models[pair[1]].config)
        outputs.append({'pair': pair, 'refs': refs,
                        'maps': rng.normal(size=(len(refs), 65, 64))})
    a = module.serialize_factor_payload(factor, mean, basis, outputs)
    b = module.serialize_factor_payload(factor, mean, basis, outputs)
    full = module.full_payload_bytes(outputs)
    assert a == b
    assert len(a) == module.factor_payload_bytes(factor, mean, basis, outputs)
    assert full > 0
