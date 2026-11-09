import importlib.util
import csv
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma734',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_givens_orthogonal():
    r=m.rotation(np.asarray([.2,-.3,.4,.1,-.2,.3]))
    np.testing.assert_allclose(r.T@r,np.eye(4),atol=1e-12)

def test_serialized_roundtrip():
    b=m.npz_bytes(x=np.arange(9,dtype=np.float32))
    import io
    with np.load(io.BytesIO(b)) as z: np.testing.assert_array_equal(z['x'],np.arange(9,dtype=np.float32))

def test_saved_direct_givens_control_alias():
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    native=next(x for x in rows if x['method']=='native_direct_givens')
    mirror=next(x for x in rows if x['method']=='mirror_role_view')
    for k in ('serialized_bytes','payload_sha256','branch_trunk_factor_nrmse','composed_operator_nrmse','active_multiplies'):
        assert native[k]==mirror[k]
