import csv,importlib.util,io
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma755',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_serialization_roundtrip():
    b=m.pack(x=np.arange(3,dtype=np.int8))
    with np.load(io.BytesIO(b)) as z:np.testing.assert_array_equal(z['x'],np.arange(3,dtype=np.int8))

def test_split_has_connected_training_graph():
    pairs={(s,b) for s in range(m.S) for b in range(m.B)}
    train=pairs-m.HELDOUT
    assert len(train)==9 and len(m.HELDOUT)==3
    assert all(any(s0==s for s0,_ in train) for s in range(m.S))
    assert all(any(b0==b for _,b0 in train) for b in range(m.B))

def test_mirror_equals_native_tower_after_run():
    m.run()
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    a=next(r for r in rows if r['method']=='native_two_tower_bridge')
    b=next(r for r in rows if r['method']=='mirror_source_x_backbone_views')
    for k in ('serialized_bytes','payload_sha256','heldout_bridge_nrmse','active_macs'):
        assert a[k]==b[k]
