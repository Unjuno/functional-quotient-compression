import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py'
spec=importlib.util.spec_from_file_location('ma573',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_fwht_squared_is_identity_up_to_scale():
    x=np.random.default_rng(1).normal(size=(3,32)).astype(np.float32)
    np.testing.assert_allclose(m.fwht(m.fwht(x))/32,x,atol=2e-6)

def test_block_rotation_and_inverse_preserve_function():
    rng=np.random.default_rng(4);w=rng.normal(size=(7,m.D)).astype(np.float32);x=rng.normal(size=(5,m.D)).astype(np.float32);s,p=m.sample_code(rng)
    wp=m.apply_weight_view(w,s,p);xp=m.apply_input_view(x,s,p)
    np.testing.assert_allclose(x@w.T,xp@wp.T,atol=3e-5)

def test_int4_roundtrip_shapes_and_error():
    w=np.random.default_rng(8).normal(size=(9,m.D)).astype(np.float32)
    packed,scales,dq=m.quantize_int4(w)
    assert packed.shape==(9,m.D//2) and scales.shape==(9,m.BLOCKS)
    assert dq.shape==w.shape and np.linalg.norm(w-dq)/np.linalg.norm(w)<.25
    np.testing.assert_array_equal(m.dequantize_int4(packed,scales),dq)
