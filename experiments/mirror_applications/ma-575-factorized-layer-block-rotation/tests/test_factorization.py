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

def test_int4_unpack_restores_original_width():
    x=np.random.default_rng(3).normal(size=(5,512)).astype(np.float32)
    packed,scales=m.quant_int4(x)
    decoded=m.unpack(packed,scales)
    assert packed.shape==(5,256)
    assert decoded.shape==x.shape

def test_identity_int4_reconstruction_is_close():
    x=np.random.default_rng(5).normal(size=(7,512)).astype(np.float32)
    packed,scales=m.quant_int4(x)
    decoded=m.unpack(packed,scales)
    assert np.linalg.norm(decoded-x)/np.linalg.norm(x)<0.11
