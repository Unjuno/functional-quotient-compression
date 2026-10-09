import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from run import PeriodicView,NativeHarmonic,make_world,sample_x,pack_model,infer

def test_seeded_world_partition_and_shapes():
 codes,amp,freq,phase,train,held=make_world(41901)
 assert codes.shape==(320,8) and amp.shape==freq.shape==phase.shape==(320,4)
 assert len(train)==256 and len(held)==64 and len(set(train.tolist())&set(held.tolist()))==0

def test_native_harmonic_is_exact_periodic_view_alias():
 torch.manual_seed(5);a=PeriodicView();torch.manual_seed(5);b=NativeHarmonic();x=torch.linspace(-1,1,13).repeat(3,1);z=torch.randn(3,8)
 raw_a=pack_model(a,z,'mirror');raw_b=pack_model(b,z,'native');pa=infer(raw_a,x,'mirror');pb=infer(raw_b,x,'native')
 assert torch.max(torch.abs(pa-pb))==0
