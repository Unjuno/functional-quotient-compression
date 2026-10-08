import importlib.util
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parents[1]/'source'/'run.py'
spec=importlib.util.spec_from_file_location('ma351',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_pack_roundtrip_and_logits():
    state=m.init(7,'mirror_rank1');x=np.random.default_rng(1).normal(size=(11,2)).astype('f4')
    payload,loaded,meta=m.pack(state,{'kind':'test'})
    assert payload and meta['kind']=='test'
    assert np.array_equal(m.forward(x,state,'mirror_rank1'),m.forward(x,loaded,'mirror_rank1'))

def test_mirror_is_native_scalar_control():
    _,train,_=m.data(35101)
    a=m.train(35101,'direct_rank1',train);b=m.train(35101,'mirror_rank1',train)
    assert np.array_equal(m.forward(train[0][:64],a,'direct_rank1'),m.forward(train[0][:64],b,'mirror_rank1'))

def test_metrics_are_finite():
    rng=np.random.default_rng(2);logits=rng.normal(size=(32,4,4));y=rng.integers(0,4,size=32)
    values=m.metrics(logits,y)
    assert np.isfinite(values[0]) and np.isfinite(values[1]) and np.isfinite(values[4])
