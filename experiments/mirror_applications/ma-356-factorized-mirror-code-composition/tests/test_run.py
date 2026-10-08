import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]/'source'/'run.py';spec=importlib.util.spec_from_file_location('ma356',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_factorized_heldout_pairs_exact():
    w=m.world(35601);arr,meta=m.states(w)['mirror_factorized'];_,s,_=m.pack(arr,meta)
    assert max(np.max(np.abs(m.decode('mirror_factorized',s,i,j)-w[6][i,j])) for i,j in w[9])==0

def test_direct_control_matches_factorized():
    w=m.world(35602);a,_=m.states(w)['mirror_factorized'];b,_=m.states(w)['direct_coefficients']
    for i,j in [(0,0),(2,5),(7,7)]:assert np.array_equal(m.decode('mirror_factorized',a,i,j),m.decode('direct_coefficients',b,i,j))

def test_pack_reload_exact():
    w=m.world(35601)
    for name,(a,meta) in m.states(w).items():
        p,s,_=m.pack(a,meta);assert p
        for i,j in [(0,0),(7,7)]:assert np.array_equal(m.decode(name,a,i,j),m.decode(name,s,i,j))
