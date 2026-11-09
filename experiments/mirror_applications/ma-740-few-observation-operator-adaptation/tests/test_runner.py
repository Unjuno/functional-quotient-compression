import csv,importlib.util,io
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma740',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_fourier_basis_dimensions_and_payload_roundtrip():
    assert m.basis(np.asarray([0.,1.])).shape==(2,8)
    b=m.pack(x=np.arange(4,dtype=np.float32))
    with np.load(io.BytesIO(b)) as z:np.testing.assert_array_equal(z['x'],np.arange(4,dtype=np.float32))

def test_phase_and_native_control_exact_alias():
    r=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    for n in ('4','8','16'):
      a=next(x for x in r if x['observations_per_task']==n and x['method']=='mirror_phase_code')
      b=next(x for x in r if x['observations_per_task']==n and x['method']=='native_phase_coordinate')
      for k in ('serialized_bytes','payload_sha256','task_state_bytes','mean_test_nrmse','mean_function_evaluations','adaptation_multiply_proxy'):
       assert a[k]==b[k]

def test_initial_unregularized_run_preserved():
    p=P.parents[1]/'RESULTS_INITIAL_UNREGULARIZED.csv'
    rows=list(csv.DictReader(p.open(newline='')))
    assert len(rows)==15
    assert any(x['method']=='cavia_linear_context' and x['observations_per_task']=='8' and float(x['mean_test_nrmse'])>1 for x in rows)
