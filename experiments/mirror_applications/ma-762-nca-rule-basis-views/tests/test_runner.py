import csv,importlib.util,io
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma762',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_serialized_payload_roundtrip():
    b=m.pack(x=np.arange(5,dtype=np.float32))
    with np.load(io.BytesIO(b)) as z:np.testing.assert_array_equal(z['x'],np.arange(5,dtype=np.float32))

def test_training_and_heldout_rule_counts():
    assert m.TRAIN==64 and m.HELD==32 and m.TASKS==m.TRAIN+m.HELD

def test_native_and_mirror_basis_alias_after_run():
    m.run()
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    a=next(r for r in rows if r['method']=='native_rank3_rule_basis')
    b=next(r for r in rows if r['method']=='mirror_rule_views')
    for k in ('serialized_bytes','payload_sha256','heldout_update_nrmse','rollout_nrmse','active_macs'):
        assert a[k]==b[k]
