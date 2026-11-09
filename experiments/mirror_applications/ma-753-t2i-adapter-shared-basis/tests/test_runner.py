import csv,importlib.util,io
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma753',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_npz_payload_roundtrip():
    blob=m.pack(x=np.arange(6,dtype=np.float32))
    with np.load(io.BytesIO(blob)) as z: np.testing.assert_array_equal(z['x'],np.arange(6,dtype=np.float32))

def test_calibration_split_sizes():
    assert m.TRAIN==8 and m.N-m.TRAIN==4 and m.CAL==8

def test_native_and_mirror_alias_after_run():
    m.run()
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    a=next(r for r in rows if r['method']=='native_shared_PCA_basis')
    b=next(r for r in rows if r['method']=='mirror_shared_adapter_views')
    for k in ('serialized_bytes','payload_sha256','heldout_condition_nrmse','composition_nrmse','active_macs'):
        assert a[k]==b[k]
