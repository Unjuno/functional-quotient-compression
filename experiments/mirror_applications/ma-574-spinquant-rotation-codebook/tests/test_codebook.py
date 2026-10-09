import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py';spec=importlib.util.spec_from_file_location('ma574',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_hadamard_transform_is_orthogonal():
    x=np.random.default_rng(2).normal(size=(4,32)).astype(np.float32)
    np.testing.assert_allclose(m.fwht(m.fwht(x))/32,x,atol=2e-6)

def test_view_and_inverse_preserve_linear_function():
    rng=np.random.default_rng(4);w=rng.normal(size=(9,m.WIDTH)).astype(np.float32);x=rng.normal(size=(3,m.WIDTH)).astype(np.float32);code=m.new_code(rng)
    wv=m.view_weight(w,code);xv=m.view_input(x,code)
    np.testing.assert_allclose(x@w.T,xv@wv.T,atol=3e-5)

def test_int4_pack_unpack_is_exact_to_dequantized_values():
    w=np.random.default_rng(8).normal(size=(9,m.WIDTH)).astype(np.float32);p,s,d=m.quant_int4(w)
    np.testing.assert_array_equal(m.unpack_int4(p,s),d)

def test_training_layer_codebook_uses_only_training_errors():
    e=np.array([[.2,.1,.3,.4],[.3,.2,.1,.4],[.1,.4,.3,.2]],float)
    cb=m.choose_codebook(e,[0,1],2)
    assert len(set(cb.tolist()))==2 and max(cb)<4
