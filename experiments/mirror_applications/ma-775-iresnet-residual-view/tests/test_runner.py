import csv,importlib.util,io
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma775',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_fixed_point_inverse_contract():
    L=np.eye(4,dtype=np.float32)*.2;x=np.asarray([[1.,-.5,.2,.3]],np.float32);y=x+x@L.T
    z,n,ok=m.inverse(y,L)
    assert ok and n<100
    np.testing.assert_allclose(z,x,atol=1e-6)

def test_payload_roundtrip():
    b=m.pack(x=np.arange(5,dtype=np.float32))
    with np.load(io.BytesIO(b)) as z:np.testing.assert_array_equal(z['x'],np.arange(5,dtype=np.float32))

def test_native_basis_and_mirror_alias():
    m.run()
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    a=next(r for r in rows if r['method']=='native_rank2_residual_basis')
    b=next(r for r in rows if r['method']=='mirror_invertible_residual_views')
    for k in ('serialized_bytes','payload_sha256','heldout_output_nrmse','max_inverse_error','inverse_converged_fraction','active_macs'):
        assert a[k]==b[k]
