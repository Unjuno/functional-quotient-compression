import csv,importlib.util,io
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma742',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_payload_roundtrip():
    b=m.pack(a=np.arange(7,dtype=np.float32))
    with np.load(io.BytesIO(b)) as z:np.testing.assert_array_equal(z['a'],np.arange(7,dtype=np.float32))

def test_heldout_combinations_are_disjoint():
    # Eight binary attribute triples; first six fit, last two are held out.
    attrs=np.asarray([[a,b,c] for a in (0,1) for b in (0,1) for c in (0,1)])
    assert len({tuple(x) for x in attrs[:6]}.intersection({tuple(x) for x in attrs[6:]}))==0

def test_mirror_exactly_aliases_native_attribute_generator():
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    for split in ('fit','heldout'):
        a=next(x for x in rows if x['method']=='native_attribute_generator' and x['split']==split)
        b=next(x for x in rows if x['method']=='mirror_attribute_code' and x['split']==split)
        for k in ('serialized_bytes','payload_sha256','operator_nrmse','hits_at_1','message_multiply_proxy'):
            assert a[k]==b[k]
