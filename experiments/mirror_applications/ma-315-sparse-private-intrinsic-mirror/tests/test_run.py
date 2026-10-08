import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run import basis,world,encode,save,load,decode

def test_basis_and_private_partition():
 p=basis(31501);assert p.shape==(32,16);assert np.max(np.abs(p.T@p-np.eye(16)))<1e-6
 _,_,_,_,_,g=world(31501);assert len(g)==128 and sum(x=='private_0' for x in g)==48 and sum(x=='private_8' for x in g)==16

def test_sparse_payload_roundtrip(tmp_path):
 P,b,z,t,sets,g=world(31502)
 for method in ('shared_sparse_direct','mirror_sparse'):
  arrays,events,ops=encode(method,P,b,t,sets,g,1e-4,tmp_path,31502,'development');path=tmp_path/(method+'.npz');n,h=save(arrays,path);s=load(path)
  assert n==path.stat().st_size and len(h)==64 and len(events)==128 and ops>0
  for i,task in enumerate(sets):assert np.isfinite(task[2][0]@decode(method,s,i)).all()

def test_independent_control_recovers_teacher(tmp_path):
 P,b,z,t,sets,g=world(31503);a,_,_=encode('independent',P,b,t,sets,g,1e-4,tmp_path,31503,'development');p=tmp_path/'ind.npz';save(a,p);s=load(p)
 for i,task in enumerate(sets):assert np.max(np.abs(decode('independent',s,i)-t[i]))<1e-7
