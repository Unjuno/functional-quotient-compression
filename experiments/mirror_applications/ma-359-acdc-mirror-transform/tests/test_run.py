import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('ma359',P);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def test_dct_orthonormal():
    q=m.dct();assert np.max(np.abs(q@q.T-np.eye(m.D)))<1e-6

def test_mirror_and_direct_equal_maps():
    w=m.world(35901)
    for i in (0,31,63):assert np.allclose(m.apply('mirror_acdc',m.state('mirror_acdc',w)[0],i,w[-1]),m.apply('direct_coefficients',m.state('direct_coefficients',w)[0],i,w[-1]))

def test_payload_roundtrip():
    w=m.world(35902)
    for method in ('dense_independent','independent_acdc','mirror_acdc','direct_coefficients','diagonal_only'):
        s,meta=m.state(method,w);payload,l,_=m.pack(s,meta);assert payload
        for i in (0,63):assert np.array_equal(m.apply(method,s,i,w[-1]),m.apply(method,l,i,w[-1]))
