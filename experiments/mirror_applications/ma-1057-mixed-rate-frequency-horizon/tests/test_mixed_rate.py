import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from collect_samples import causal_block_mean,make_origins
from fit_controls import decode

def test_physical_context_and_horizon_are_equal_across_rates():
    assert 288*5==96*15==24*60
    assert 12*5==4*15==60
    assert 288*5==96*15==24*60

def test_three_point_box_mean_attenuates_alias_frequency():
    x=np.tile(np.array([1.0,-.5,-.5],dtype=np.float32)[:,None],(400,2))
    y=causal_block_mean(x)
    assert np.max(np.abs(y))<1e-7

def test_fit_dev_audit_targets_stay_in_their_chronological_bounds():
    for f,n,ctx,hs in [(0,52116,288,(12,288)),(1,17372,96,(4,96))]:
        for split in ('train','dev','audit'):
            for h in hs:
                origins=make_origins(n,ctx,h,split,f)
                n15=52116//3; tr=int(n15*.6); dv=int(n15*.8)
                if f==0: tr,dv=tr*3,dv*3
                lo,hi=((ctx,tr) if split=='train' else (tr,dv) if split=='dev' else (dv,n))
                assert all(o>=lo and o+h<=hi for o in origins)

def test_factorized_scale_additive_offset_is_native_additive_log_alias():
    z=np.array([.3,-.2,.4,.1,2.,3.,-.5,.8])
    ls=[]; offsets=[]
    for f,h in ((0,0),(0,1),(1,0),(1,1)):
        s,b=decode('mirror_factorized',z,[f],[h]); ls.append(np.log(s[0])); offsets.append(b[0])
    additive=np.array([ls[0],ls[2]-ls[0],ls[1]-ls[0],offsets[0],offsets[2]-offsets[0],offsets[1]-offsets[0]])
    for f,h in ((0,0),(0,1),(1,0),(1,1)):
        mirror=decode('mirror_factorized',z,[f],[h]); simple=decode('additive_log',additive,[f],[h])
        assert np.allclose(mirror,simple,rtol=1e-12,atol=1e-12)
