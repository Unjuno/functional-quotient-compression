import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]/'source'/'run.py';spec=importlib.util.spec_from_file_location('ma355',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_factorized_recovers_all_maps():
    w=m.bank(35501);arrays,meta=m.methods(w)['mirror_factorized'];_,state,_=m.pack(arrays,meta)
    assert max(np.max(np.abs(m.decode('mirror_factorized',state,i,j)-w[4][i,j])) for i,j in w[5])==0

def test_payload_roundtrip():
    w=m.bank(35501)
    for method,(arrays,meta) in m.methods(w).items():
        payload,state,_=m.pack(arrays,meta);assert payload and len(payload)>0
        for i,j in ((0,0),(15,15)):
            assert np.array_equal(m.decode(method,arrays,i,j),m.decode(method,state,i,j))

def test_direct_and_mirror_same_functions():
    w=m.bank(35502);a,ma=m.methods(w)['mirror_factorized'];b,mb=m.methods(w)['direct_two_index_coefficients']
    for i,j in ((0,0),(4,9),(15,15)):assert np.array_equal(m.decode('mirror_factorized',a,i,j),m.decode('direct_two_index_coefficients',b,i,j))
