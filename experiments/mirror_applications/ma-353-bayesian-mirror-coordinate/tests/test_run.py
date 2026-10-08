import importlib.util
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parents[1]/'source'/'run.py';spec=importlib.util.spec_from_file_location('ma353',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_archive_roundtrip_predictions():
    w,j,d,*_=m.world(35301)
    a,meta=m.posterior('mirror_gaussian_coordinate',w,j,d);payload,state,mm=m.pack(a,meta)
    x=np.random.default_rng(1).normal(size=(32,m.D)).astype('f4')
    assert payload and np.array_equal(m.predict('mirror_gaussian_coordinate',state,x),m.predict('mirror_gaussian_coordinate',a,x))

def test_direct_coordinate_is_exact_same_payload():
    w,j,d,*_=m.world(35301)
    a,ma=m.posterior('mirror_gaussian_coordinate',w,j,d);b,mb=m.posterior('direct_gaussian_scalar',w,j,d)
    pa,_,_=m.pack(a,ma);pb,_,_=m.pack(b,mb)
    assert pa==pb

def test_teacher_modes_have_three_components():
    _,j,d,x,_,_,ptrue,_,weights=m.world(35301)
    assert weights.shape==(3,m.D) and np.all((ptrue>0)&(ptrue<1))
