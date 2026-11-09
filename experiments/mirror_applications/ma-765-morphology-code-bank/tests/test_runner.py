import csv,importlib.util,io
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma765',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_target_generator_shape_and_range():
    y=m.generate(np.asarray([.4,.2,2.,0.],np.float32))
    assert y.shape==(m.SIZE,m.SIZE) and np.isfinite(y).all() and (y.min()>=0) and (y.max()<=1)

def test_payload_roundtrip():
    b=m.pack(x=np.arange(5,dtype=np.float32))
    with np.load(io.BytesIO(b)) as z:np.testing.assert_array_equal(z['x'],np.arange(5,dtype=np.float32))

def test_native_code_and_mirror_alias_after_run():
    m.run()
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    a=next(r for r in rows if r['method']=='native_analytic_morphology_code')
    b=next(r for r in rows if r['method']=='mirror_morphology_views')
    for k in ('serialized_bytes','payload_sha256','final_target_nrmse','recovery_fraction','active_macs'):
        assert a[k]==b[k]
