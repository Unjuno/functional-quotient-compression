import importlib.util
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parents[1]/'source'/'run.py';spec=importlib.util.spec_from_file_location('ma322_run',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_tt_reconstructs_64_by_64_matrix():
 c=m.random_cores(np.random.default_rng(2));w=m.tt_matrix(*c);assert w.shape==(64,64) and np.isfinite(w).all()

def test_direct_pair_recovers_aligned_task_function():
 w=m.world(32201,'aligned');x,y=w['sets'][0][0];a,b=m.fit_pair(x,y,w['basew'],w['uw'],w['vw']);pred=w['basew']+a*w['uw']+b*w['vw'];assert m.nrmse(w['sets'][0][2][0],w['sets'][0][2][1],pred)<1e-8

def test_phase_search_recovers_aligned_task_function():
 w=m.world(32201,'aligned');x,y=w['sets'][0][0];a,ops=m.fit_phase(x,y,w['basew'],w['uw'],w['vw']);pred=w['basew']+m.RHO*(np.cos(a)*w['uw']+np.sin(a)*w['vw']);assert ops>0 and m.nrmse(w['sets'][0][2][0],w['sets'][0][2][1],pred)<1e-4

def test_unrelated_tasks_select_private_loretta_fallback():
 w=m.world(32201,'mixed');a,events,_,_=m.encode('mirror_phase_private',w);assert len(a['private_ids'])==32 and len(a['angle_ids'])==96

def test_payload_is_deterministic_and_roundtrips(tmp_path):
 w=m.world(32201,'mixed');a,_,_,_=m.encode('mirror_phase_private',w);p1=tmp_path/'a.npz';p2=tmp_path/'b.npz';m.pack(a,p1);m.pack(a,p2);s=m.unpack(p1);assert p1.read_bytes()==p2.read_bytes()
 for i in [0,95,96,127]:
  x,y=w['sets'][i][2];assert m.nrmse(x,y,m.decode('mirror_phase_private',s,i))<1e-4
