import hashlib, json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
def test_protocol_freeze_and_fresh_sealed():
    f=json.loads((P/'freeze.json').read_text())
    digest=hashlib.sha256((P/'PROTOCOL.json').read_bytes()).hexdigest()
    assert f['protocol_sha256']==digest and f['fresh_accessed'] is False
def test_pca_native_control_aliases_mirror():
    rows=json.loads((P/'source'/'development.json').read_text())
    for a in rows:
        if a['method']!='mirror_pca_residual': continue
        b=next(x for x in rows if (x['world'],x['seed'],x['rank'],x['method'])==(a['world'],a['seed'],a['rank'],'native_pca_residual'))
        assert a['serialized_bytes']==b['serialized_bytes']
        assert a['heldout_nrmse']==b['heldout_nrmse']
        pa=P/'source'/f"{a['world']}_{a['seed']}_r{a['rank']}_mirror_pca_residual.npz"
        pb=P/'source'/f"{b['world']}_{b['seed']}_r{b['rank']}_native_pca_residual.npz"
        with np.load(pa) as x, np.load(pb) as y:
            assert x.files==y.files
            assert all(np.array_equal(x[k],y[k]) for k in x.files)
def test_no_private_rank_meets_joint_gate_on_both_seeds():
    rows=json.loads((P/'source'/'development.json').read_text())
    for seed in [62301,62302]:
        viable=[x for x in rows if x['seed']==seed and x['world']=='private_interaction' and x['method']=='mirror_pca_residual' and x['heldout_nrmse']<=.05 and x['serialized_bytes']<=.75*x['independent_payload_bytes']]
        assert not viable
