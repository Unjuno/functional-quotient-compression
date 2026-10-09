import importlib.util,zipfile
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parents[1]/'source'/'run_compiler.py'
spec=importlib.util.spec_from_file_location('ma521_compiler',SRC);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_demo_code_is_unit_normalized():
    x=np.random.default_rng(4).normal(size=(5,512)).astype(np.float32)
    a=np.random.default_rng(7).normal(size=(512,3)).astype(np.float32);b=np.zeros(3,np.float32)
    z=mod.encode_features(x,a,b,True)
    assert z.shape==(5,3) and np.isfinite(z).all() and np.max(np.abs(z))<=1

def test_vector_audit_identity():
    x=np.random.default_rng(3).normal(size=(16,512)).astype(np.float32)
    a=mod.vector_audit(x,x)
    assert a['heldout_relative_rmse']==0 and a['heldout_cosine_min']>0.999999

def test_shared_harness_direct_demos_prefix():
    class Tok:
        pad_token_id=0
        def encode(self,s,add_special_tokens=False):return [ord(c)%97 for c in s]
    query='France';demos=[('Italy','Rome')]
    p=mod.h.prompt(query,demos)
    assert 'Input: Italy\nOutput: Rome' in p and p.endswith('Input: France\nOutput:')
