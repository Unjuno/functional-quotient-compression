import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('ma357',P);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def test_unique_stored_addresses():
    for k in m.KS:
        p,*_=m.world(35701,k);assert len({q.tobytes() for q in p})==k

def test_hopfield_soft_retrieval_is_finite():
    r=m.run(35701,64,'modern_hopfield');assert 0<=r['query_accuracy']<=1 and np.isfinite(r['decoded_code_nMSE'])

def test_packed_storage_roundtrip():
    p,c,*_=m.world(35702,16);state={'addresses_packed':np.packbits(p,axis=1),'view_codes':c,'temperature':np.array([8.],'f4')};payload,s,_=m.pack(state,{'kind':'test'})
    assert np.array_equal(np.unpackbits(s['addresses_packed'],axis=1)[:,:m.BITS],p)
    assert payload
