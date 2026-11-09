import csv
import importlib.util
import io
from pathlib import Path

import numpy as np

SOURCE=Path(__file__).parents[1]/'source'/'run_experiment.py'
spec=importlib.util.spec_from_file_location('ma736',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_green_operator_and_payload_roundtrip():
    k,_,_=m.make_green(.3,7)
    assert k.shape==(24,24)
    assert np.all(np.isfinite(k))
    b=m.payload(x=np.arange(11,dtype=np.float32))
    with np.load(io.BytesIO(b)) as z:np.testing.assert_array_equal(z['x'],np.arange(11,dtype=np.float32))

def test_sparse_residual_budget():
    x=np.arange(24*24,dtype=np.float32).reshape(24,24)
    idx,val,out=m.topk_sparse(x,8)
    assert idx.shape==(8,2) and val.shape==(8,) and np.count_nonzero(out)==8

def test_mirror_equals_native_pca_and_private_controls():
    p=SOURCE.parents[1]/'RESULTS_CORE.csv'
    rows=list(csv.DictReader(p.open(newline='')))
    pairs=[('native_pca','mirror_view',0)]+[('native_pca_private','mirror_private',k) for k in (0,4,8,16,32,64)]
    for a,b,k in pairs:
      for group in ('fit','heldout'):
       x=next(z for z in rows if z['method']==a and z['private_topk']==str(k) and z['contrast_group']==group)
       y=next(z for z in rows if z['method']==b and z['private_topk']==str(k) and z['contrast_group']==group)
       for field in ('serialized_bytes','payload_sha256','global_solution_nrmse','interface_solution_nrmse','active_multiplies'):
        assert x[field]==y[field]
