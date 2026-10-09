import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma578',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_rotation_inverse_on_head_vectors():
 r=np.random.default_rng(578);x=r.normal(size=(9,64)).astype(np.float32);c=m.new_code(r);np.testing.assert_allclose(m.unrotate(m.rotate(x,c),c),x,atol=2e-6,rtol=2e-6)
def test_int4_kv_round_trip_shapes_and_error():
 x=np.random.default_rng(4).normal(size=(7,64)).astype(np.float32);p,s=m.qint4(x);d=m.dqint4(p,s);assert p.shape==(7,32);assert d.shape==x.shape;assert np.linalg.norm(d-x)/np.linalg.norm(x)<.12
def test_role_count_matches_layers_heads_and_kv_types():
 assert m.ROLES==6*8*2==96
def test_role_index_decodes_layer_head_and_kv_slot():
    sample=[]
    for layer in range(m.LAYERS):
        key=np.full((1,m.HEADS,3,m.DIM),100*layer+0,dtype=np.float32)
        value=np.full((1,m.HEADS,3,m.DIM),100*layer+1,dtype=np.float32)
        sample.append([key,value])
    caches=[sample]
    for layer in range(m.LAYERS):
        for head in range(m.HEADS):
            for slot in range(2):
                role=layer*16+head*2+slot
                got=m.cache_role(caches,role)
                assert np.all(got[0]==100*layer+slot)

def test_signed_permutation_before_hadamard_is_nontrivial():
    r=np.random.default_rng(11);x=r.normal(size=(4,64)).astype(np.float32);a=m.new_code(r);b=m.new_code(r)
    qa,sa=m.qint4(m.rotate(x,a));qb,sb=m.qint4(m.rotate(x,b))
    da=m.unrotate(m.dqint4(qa,sa),a);db=m.unrotate(m.dqint4(qb,sb),b)
    assert not np.array_equal(da,db)
