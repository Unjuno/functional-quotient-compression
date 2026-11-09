import csv,importlib.util,io
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma772',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_flow_cycle_and_split():
    x=np.asarray([[.2,-.3]],np.float32);y,s=m.transform(x,.4)
    inv=np.stack([y[:,0],(y[:,1]-(m.FLOW[2]*y[:,0]+m.FLOW[3]*.4))*np.exp(-s)],axis=1)
    np.testing.assert_allclose(inv,x,atol=1e-6)
    assert len(m.TRAIN)==12 and len(m.HELD)==4

def test_payload_roundtrip():
    blob=m.pack(x=np.arange(6,dtype=np.float32))
    with np.load(io.BytesIO(blob)) as z:np.testing.assert_array_equal(z['x'],np.arange(6,dtype=np.float32))

def test_native_factor_and_mirror_alias():
    m.run()
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    a=next(r for r in rows if r['method']=='native_product_condition_behavior')
    b=next(r for r in rows if r['method']=='mirror_product_condition_behavior')
    for k in ('serialized_bytes','payload_sha256','heldout_transform_nrmse','max_cycle_error','heldout_code_rmse'):
        assert a[k]==b[k]
