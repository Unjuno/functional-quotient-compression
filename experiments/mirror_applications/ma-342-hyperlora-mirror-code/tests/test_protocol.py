import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma342',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_heldout_phases_are_unseen_and_shape_matches():
 a=m.world(34221);assert not set(a['phtr']).intersection(set(a['phte']))
 assert a['xt'].shape==(16,64,8) and a['xe'].shape==(8,256,8)
def test_effective_lora_delta_is_gauge_invariant():
 r=np.random.default_rng(1);aa=r.normal(size=(8,2));bb=r.normal(size=(2,8));g=np.array([[2.,.5],[0.,.75]])
 d=aa@bb;dp=(aa@g)@np.linalg.solve(g,bb)
 assert np.allclose(d,dp,atol=1e-10)
def test_client_code_payload_charged():
 a=m.world(34222);desc=np.stack([np.cos(a['phte']),np.sin(a['phte'])],-1).astype(np.float32);phase=a['phte'][:,None].astype(np.float32)
 x=m.pack('mirror_phase',a['base'],[a['u'],a['v']],phase);y=m.pack('generic_coefficients',a['base'],[a['u'],a['v']],desc)
 assert len(x)<len(y)
