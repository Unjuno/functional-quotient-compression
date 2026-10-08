import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run import make_world,pack,load,simulate,ROLES,NSEQ,LENGTH

def test_recurrence_world_shapes_and_no_training_state():
 base,codes,x=make_world(43401)
 assert len(codes)==ROLES and x.shape==(ROLES,NSEQ,LENGTH)
 assert base['A'].shape==base['B'].shape==base['C'].shape==(16,)

def test_native_decay_bias_is_exact_payload_and_output_alias():
 base,codes,x=make_world(43401);a=pack(base,codes,'mirror');b=pack(base,codes,'native')
 assert a==b
 ya=simulate(a,x,'mirror');yb=simulate(b,x,'native')
 assert torch.max(torch.abs(ya-yb))==0 and not ya.requires_grad

def test_independent_full_copy_payload_expands_every_role():
 base,codes,x=make_world(43401);raw=pack(base,codes,'independent');v=load(raw,'independent')
 assert v['A'].shape==v['B'].shape==v['C'].shape==(ROLES,16)
 assert v['wd'].shape==v['bd'].shape==(ROLES,)
 assert simulate(raw,x,'independent').shape==x.shape
