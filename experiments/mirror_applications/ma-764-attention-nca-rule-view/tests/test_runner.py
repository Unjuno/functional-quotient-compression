import csv,importlib.util,io
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma764',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_payload_roundtrip():
    b=m.pack(a=np.arange(7,dtype=np.float32))
    with np.load(io.BytesIO(b)) as z:np.testing.assert_array_equal(z['a'],np.arange(7,dtype=np.float32))

def test_context_split():
    assert m.TRAIN==20 and m.HELD==12 and m.TASKS==m.TRAIN+m.HELD

def test_native_and_mirror_hypernetwork_alias():
    m.run()
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    a=next(r for r in rows if r['method']=='native_attention_hypernetwork')
    b=next(r for r in rows if r['method']=='mirror_attention_rule_view')
    for k in ('serialized_bytes','payload_sha256','heldout_update_nrmse','rollout_nrmse','active_macs'):
        assert a[k]==b[k]
