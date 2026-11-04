import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def test_development_payload_sizes_and_alias():
    rows = json.loads((ROOT / 'source/development.json').read_text())
    assert len(rows) == 36
    by_key = {(r['seed'], r['residual_rms'], r['method']): r for r in rows}
    for r in rows:
        payload = ROOT / 'source' / f"{r['seed']}_rho{r['residual_rms']:g}_{r['method']}.npz"
        assert payload.stat().st_size == r['serialized_bytes']
    for seed in (67101, 67102):
        for rho in (0.0, 0.05, 0.2):
            for a, b in (('mirror_code', 'native_shared_basis'), ('mirror_code_private', 'native_shared_basis_private')):
                ra, rb = by_key[(seed, rho, a)], by_key[(seed, rho, b)]
                assert ra['output_nrmse'] == rb['output_nrmse']
                with np.load(ROOT / 'source' / f'{seed}_rho{rho:g}_{a}.npz') as x, np.load(ROOT / 'source' / f'{seed}_rho{rho:g}_{b}.npz') as y:
                    assert np.array_equal(x['base'], y['base'])
                    assert np.array_equal(x['router'], y['router'])
                    assert np.array_equal(x['atom_left'], y['atom_left'])
                    assert np.array_equal(x['atom_right'], y['atom_right'])
                    assert np.array_equal(x['expert_codes' if a == 'mirror_code' else 'expert_codes'], y['expert_scales'])
                    assert ('private_left' in x) == ('private_left' in y)
