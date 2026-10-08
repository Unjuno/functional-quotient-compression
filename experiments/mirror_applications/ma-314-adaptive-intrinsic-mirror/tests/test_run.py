import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run import basis,make_world,pack_arrays,load_arrays,decode,encode_state

def test_projection_basis_and_complexity_partition():
 p=basis(31401);assert p.shape==(32,16);assert np.max(np.abs(p.T@p-np.eye(16)))<1e-6
 _,_,z,_,_,groups=make_world(31401)
 assert len(groups)==160 and groups.count('aligned_d2')==32 and groups.count('aligned_d16')==32 and groups.count('unrelated')==32
 assert np.allclose(z[:32,2:],0)

def test_payload_roundtrip_adaptive_methods(tmp_path):
 P,b,z,w,sets,groups=make_world(31402)
 for method in ['adaptive_direct','adaptive_mirror']:
  arrays,events,_=encode_state(method,P,b,w,sets,groups,1e-4)
  path=tmp_path/(method+'.npz');n,h=pack_arrays(arrays,path);loaded=load_arrays(path)
  assert n==path.stat().st_size and len(h)==64 and len(events)==160
  for i,task in enumerate(sets):
   pred=task[2][0]@decode(method,loaded,i);assert np.isfinite(pred).all()

def test_deterministic_payload_bytes(tmp_path):
 a={'x':np.arange(9,dtype=np.float32),'y':np.array([1,2],dtype=np.uint8)}
 x=tmp_path/'a.npz';y=tmp_path/'b.npz';assert pack_arrays(a,x)==pack_arrays(a,y)
