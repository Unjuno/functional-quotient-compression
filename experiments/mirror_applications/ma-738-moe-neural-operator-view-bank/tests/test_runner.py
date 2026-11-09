import csv
import importlib.util
import io
from pathlib import Path
import numpy as np

P=Path(__file__).parents[1]/'source'/'run_experiment.py'
spec=importlib.util.spec_from_file_location('ma738',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_spectral_atoms_cover_all_bands():
    assert m.BANDS[-1][1]==m.M
    assert len(m.BANDS)==m.R

def test_payload_roundtrip():
    b=m.pack(a=np.arange(6,dtype=np.float32),r=np.asarray([1,2],np.int8))
    with np.load(io.BytesIO(b)) as z:
        np.testing.assert_array_equal(z['a'],np.arange(6,dtype=np.float32))
        np.testing.assert_array_equal(z['r'],[1,2])

def test_mirror_rows_exactly_alias_native_controls():
    rows=list(csv.DictReader((P.parents[1]/'RESULTS_CORE.csv').open(newline='')))
    for top in ('1','2'):
        name='native_pca_top'+top
        if top=='1':name='native_pca_top1'
        a=next(x for x in rows if x['method']==name)
        b=next(x for x in rows if x['method']=='mirror_view_top'+top)
        for key in ('serialized_bytes','payload_sha256','total_solution_nrmse','high_band_solution_nrmse','active_spectral_multiplies'):
            assert a[key]==b[key]
