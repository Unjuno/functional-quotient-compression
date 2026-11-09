import csv,importlib.util,io
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma751',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_rotation_orthogonal():
    r=m.rotation(np.asarray([.1,-.2,.3,.4,-.3,.2],np.float32))
    np.testing.assert_allclose(r.T@r,np.eye(m.HID),atol=1e-6)

def test_serialized_payload_roundtrip():
    b=m.pack(x=np.arange(8,dtype=np.float32))
    with np.load(io.BytesIO(b)) as z:np.testing.assert_array_equal(z['x'],np.arange(8,dtype=np.float32))

def test_mirror_and_native_givens_alias():
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    a=next(x for x in rows if x['method']=='native_direct_givens')
    b=next(x for x in rows if x['method']=='mirror_givens_view')
    for k in ('serialized_bytes','payload_sha256','spatial_nrmse','active_macs'):
        assert a[k]==b[k]
