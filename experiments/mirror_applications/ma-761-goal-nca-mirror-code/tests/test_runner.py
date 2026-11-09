import csv,importlib.util,io
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma761',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_payload_roundtrip():
    b=m.pack(a=np.arange(4,dtype=np.int8))
    with np.load(io.BytesIO(b)) as z:np.testing.assert_array_equal(z['a'],np.arange(4,dtype=np.int8))

def test_goal_split_disjoint():
    assert set(m.TRAIN_GOALS).isdisjoint(m.HELDOUT)
    assert len(m.TRAIN_GOALS)==20 and len(m.HELDOUT)==5

def test_coordinate_view_exactly_aliases_native_after_run():
    m.run()
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    a=next(r for r in rows if r['method']=='native_coordinate_goal_rule')
    b=next(r for r in rows if r['method']=='mirror_coordinate_goal_view')
    for k in ('serialized_bytes','payload_sha256','rollout_success','mean_steps','active_macs'):
        assert a[k]==b[k]
