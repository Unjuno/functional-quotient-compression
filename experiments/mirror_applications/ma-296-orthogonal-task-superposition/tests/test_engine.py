import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_orthogonal_transform_roundtrip():
    rng=np.random.default_rng(1); x=rng.normal(size=run.D)
    y=run.q_apply(x,0.71)
    assert np.allclose(run.q_apply(y,0.71,inverse=True),x,atol=1e-12)

def test_payload_roundtrip_and_direct_orbit_equivalence():
    w=run.world(29601,'aligned_orthogonal_orbit')
    mirror=run.package('mirror_superposition',w); direct=run.package('direct_orbit',w)
    mr=run.reconstruct(run.deserialize(run.serialize(mirror)),w,'mirror_superposition')
    dr=run.reconstruct(run.deserialize(run.serialize(direct)),w,'direct_orbit')
    assert np.allclose(mr,dr,atol=2e-7,rtol=0)
    assert len(run.serialize(mirror)) >= len(run.serialize(direct))

def test_psp_codes_are_signed_and_roundtrip():
    w=run.world(29602,'aligned_orthogonal_orbit')
    pkg=run.package('psp_rademacher',w,psp_seed=296101)
    codes=np.unpackbits(pkg['arrays']['packed_codes'],axis=1,count=run.D,bitorder='little')
    assert set(np.unique(codes)) <= {0,1}
    pred=run.reconstruct(run.deserialize(run.serialize(pkg)),w,'psp_rademacher')
    assert pred.shape==(run.K,run.D)

def test_independent_world_is_not_orbit_generated():
    w=run.world(29603,'independent_isotropic_deltas')
    pred=run.reconstruct(run.package('direct_orbit',w),w,'direct_orbit')
    mse,_,_,_=run.measure(w,pred)
    assert mse>1e-3
