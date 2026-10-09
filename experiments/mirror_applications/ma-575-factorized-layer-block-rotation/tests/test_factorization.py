import numpy as np
from pathlib import Path
import importlib.util
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma575',P); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_signed_permutation_composition_matches_sequential_application():
    rng=np.random.default_rng(575); a=m.new_code(rng); b=m.new_code(rng); x=rng.normal(size=(3,32)).astype(np.float32)
    sequential=(x[:,a[1]]*a[0][None,:])[:,b[1]]*b[0][None,:]
    c=m.compose_codes(a,b); combined=x[:,c[1]]*c[0][None,:]
    np.testing.assert_array_equal(sequential,combined)
def test_factor_code_count_is_small():
    assert 16*33+6+16 < 6*16*33
