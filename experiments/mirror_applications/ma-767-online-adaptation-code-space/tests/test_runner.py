import csv,importlib.util,io
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma767',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_payload_roundtrip():
    b=m.pack(x=np.arange(8,dtype=np.float32))
    with np.load(io.BytesIO(b)) as z:np.testing.assert_array_equal(z['x'],np.arange(8,dtype=np.float32))

def test_rank_and_support_contract():
    assert m.R==2 and m.D==16 and m.SUPPORTS==(4,8,16)

def test_native_lowrank_aliases_mirror_after_run():
    m.run()
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    for k in m.SUPPORTS:
      a=next(r for r in rows if int(r['support_examples_per_task'])==k and r['method']=='native_lowrank_adapter_bank')
      b=next(r for r in rows if int(r['support_examples_per_task'])==k and r['method']=='mirror_functional_code_bank')
      for key in ('serialized_bytes','payload_sha256','mean_test_mse','relative_rmse','adaptation_ops'):
        assert a[key]==b[key]
