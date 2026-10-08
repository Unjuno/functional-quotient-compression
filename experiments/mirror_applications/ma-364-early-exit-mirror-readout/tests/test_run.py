import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('ma364',P);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def test_fixed_exit_fractions():
    w=m.world(36401);exit_id=w[5];assert np.allclose([(exit_id==i).mean() for i in range(3)],m.FRAC,atol=1/m.N)

def test_mirror_direct_predictions_equal():
    w=m.world(36401);x=w[4];a,ma=m.states(w)['direct_coefficients'];b,mb=m.states(w)['mirror_depth_views']
    for i in range(3):assert np.array_equal(m.logits('direct_coefficients',a,x,i),m.logits('mirror_depth_views',b,x,i))

def test_pack_roundtrip():
    w=m.world(36402)
    for name,(a,meta) in m.states(w).items():
        p,s,_=m.pack(a,meta);assert p
        x=w[4]
        assert np.array_equal(m.logits(name,a,x,1),m.logits(name,s,x,1))
