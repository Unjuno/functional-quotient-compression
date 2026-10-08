import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/"source"/"run.py"
spec=importlib.util.spec_from_file_location("ma337",P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_position_role_actions_commute_and_reconstruct():
    W,t,off,x,Ps,Ss=m.world(33711)
    for p,r in m.ALL_PAIRS:
        A=np.kron(Ps[p],Ss[r])
        assert np.max(np.abs(A@W@A.T-t[p,r]))<1e-6
        assert np.max(np.abs(np.kron(Ps[p],Ss[r])@W-W@np.kron(Ps[p],Ss[r])))>1e-5 or p==0 and r==0

def test_held_compositions_not_seen_ids():
    assert set(m.SEEN).isdisjoint(m.HELD)
    assert len(m.SEEN)==6 and len(m.HELD)==2
    assert set(m.SEEN+m.HELD)==set(m.ALL_PAIRS)

def test_orbit_has_multiple_functions_and_offorbit_private_is_distinct():
    W,t,off,x,Ps,Ss=m.world(33712)
    outputs=[x@t[p,r].T for p,r in m.ALL_PAIRS]
    assert any(np.max(np.abs(outputs[0]-q))>1e-4 for q in outputs[1:])
    assert np.mean((x@W.T-x@off.T)**2)>1e-4
