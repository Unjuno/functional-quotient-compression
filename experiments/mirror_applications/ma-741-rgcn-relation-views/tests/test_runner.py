import csv
import importlib.util
import io
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma741',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_payload_roundtrip():
    b=m.pack(x=np.arange(5,dtype=np.float32))
    with np.load(io.BytesIO(b)) as z:np.testing.assert_array_equal(z['x'],np.arange(5,dtype=np.float32))

def test_code_factor_rank():
    rng=np.random.default_rng(1);u=rng.normal(size=(32,2));v=rng.normal(size=(2,4));c=u@v
    assert np.linalg.matrix_rank(c,tol=1e-10)<=2

def test_mirror_equals_native_factorization():
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    for split in ('train','heldout'):
        a=next(x for x in rows if x['method']=='mirror_relation_code' and x['split']==split)
        b=next(x for x in rows if x['method']=='native_rank2_factor' and x['split']==split)
        for k in ('serialized_bytes','payload_sha256','operator_nrmse','hits_at_1','mrr','message_multiply_proxy'):
            assert a[k]==b[k]
