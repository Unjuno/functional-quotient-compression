import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run import make_world,rotations,transformed,pack,load,simulate,E,D,NSEQ,LENGTH

def test_givens_codes_form_orthogonal_views():
 base,theta,x=make_world(43601);q=rotations(theta)
 assert q.shape==(E,D,D) and x.shape==(E,NSEQ,LENGTH)
 assert torch.max(torch.abs(q.transpose(1,2)@q-torch.eye(D)[None]))<1e-6

def test_native_generated_control_matches_view_payload_and_outputs():
 base,theta,x=make_world(43601);a=pack(base,theta,'mirror');b=pack(base,theta,'native')
 assert a==b
 ya=simulate(a,x,'mirror');yb=simulate(b,x,'native')
 assert torch.max(torch.abs(ya-yb))<1e-6 and not ya.requires_grad

def test_full_expert_copies_load_all_recurrence_matrices():
 base,theta,x=make_world(43601);v=load(pack(base,theta,'independent'),'independent')
 assert v['A'].shape==(E,D,D) and v['B'].shape==v['C'].shape==(E,D)
