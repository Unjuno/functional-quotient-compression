import importlib.util
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parents[1]/'source'/'run.py';spec=importlib.util.spec_from_file_location('ma320_run',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_harmonic_teacher_is_recovered_by_dense_tucker():
 b,base,w,sets,g=m.world(32001);c=m.fit_coeff(sets[0][0][0],sets[0][0][1],base,b,16);rec=base+np.einsum('k,kod->od',c,b)
 assert m.nrmse(sets[0][2][0],sets[0][2][1],rec)<1e-8

def test_mirror_phase_search_recovers_aligned_expert():
 b,base,w,sets,g=m.world(32001);a,c,ops=m.mirror_angle(*sets[0][0],base,b);rec=base+np.einsum('k,kod->od',c,b)
 assert ops>0 and m.nrmse(sets[0][2][0],sets[0][2][1],rec)<1e-5

def test_unrelated_expert_is_off_shared_bank():
 b,base,w,sets,g=m.world(32001);c=m.fit_coeff(sets[-1][0][0],sets[-1][0][1],base,b,16);rec=base+np.einsum('k,kod->od',c,b)
 assert m.nrmse(sets[-1][2][0],sets[-1][2][1],rec)>m.THRESHOLD

def test_deterministic_archive_bytes(tmp_path):
 b,base,w,sets,g=m.world(32001);arr,ev,ops,wall=m.encode('tucker16_private',b,base,w,sets,g);p1=tmp_path/'a.npz';p2=tmp_path/'b.npz';m.pack(arr,p1);m.pack(arr,p2)
 assert p1.read_bytes()==p2.read_bytes()

def test_mirror_archive_reload_matches_decoded_functions(tmp_path):
 b,base,w,sets,g=m.world(32001);arr,_,_,_=m.encode('mirror',b,base,w,sets,g);path=tmp_path/'mirror.npz';m.pack(arr,path);s=m.unpack(path)
 for i in [0,17,47,48,63]:
  x,y=sets[i][2];assert m.nrmse(x,y,m.decode('mirror',s,i))<1e-4
