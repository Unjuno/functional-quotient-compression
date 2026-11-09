import csv, importlib.util, io
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma752',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_payload_roundtrip():
    b=m.pack(a=np.arange(5,dtype=np.float32))
    with np.load(io.BytesIO(b)) as z: np.testing.assert_array_equal(z['a'],np.arange(5,dtype=np.float32))

def test_heldout_pairs_are_disjoint():
    assert len(m.HELDOUT)==4
    train={(c,d) for c in range(m.C) for d in range(m.D)}-m.HELDOUT
    assert train.isdisjoint(m.HELDOUT)
    assert len(train)==12

def test_mirror_is_exact_native_factorized_alias_after_run():
    m.run()
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    native=next(x for x in rows if x['method']=='native_factorized_condition_x_depth')
    mirror=next(x for x in rows if x['method']=='mirror_factorized_condition_x_depth')
    for k in ('serialized_bytes','payload_sha256','mean_heldout_operator_nrmse','active_macs'):
        assert native[k]==mirror[k]
