import importlib.util, json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma622',P/'source'/'run_experiment.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def test_rotation_codes_roundtrip_and_payload():
    q,a=m.givens(62201,0); q2=np.eye(8)
    for k,x in enumerate(a):
        i,j=2*k,2*k+1; c,s=np.cos(x),np.sin(x); g=np.eye(8); g[i,i]=g[j,j]=c; g[i,j]=-s; g[j,i]=s; q2=q2@g
    assert np.allclose(q,q2)
def test_fresh_seeds_sealed():
    f=json.loads((P/'freeze.json').read_text()); assert f['fresh_accessed'] is False
    assert f['fresh_seeds']==[62211,62212,62213]
def test_inference_payload_loads_with_charged_factor_indices():
    m.run(62201)
    p=P/'source'/'aligned_62201_mirror_rotation.npz'
    with np.load(p) as z:
        assert set(z.files)=={'w','basis','c0','factor_angles','factor_pairs'}
        assert z['factor_pairs'].shape==(16,2)
        assert z['factor_angles'].size==32
