import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('ma883',ROOT/'source'/'experiment.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_role_ids_cover_180_roles_once():
    ids=[m.role_id(l,h,k) for l in range(m.LAYERS) for h in range(m.HEADS) for k in range(2)]
    assert len(ids)==180 and sorted(ids)==list(range(180))

def test_nearest_relative_depth_is_fixed_end_to_end():
    assert [m.nearest_layer(t,6,30) for t in (0,7,15,22,29)]==[0,1,3,4,5]
    assert [m.nearest_layer(t,12,30) for t in (0,7,15,22,29)]==[0,3,6,8,11]

def test_pair_film_is_scalar_gain_plus_vector_bias():
    x=np.zeros((5,m.ROLES,m.D),np.float32); x[:, :, :] = np.arange(5,dtype=np.float32)[:,None,None]
    maps=np.zeros((m.ROLES,m.D+1,m.D),np.float32)
    maps[:,0,0]=1.; maps[:,-1,1]=2.
    g=np.full((m.ROLES,1),2.,np.float32); b=np.zeros((m.ROLES,m.D),np.float32); b[:,1]=3.
    y=m.film_predict(x,maps,g,b)
    assert y.shape==(5,m.ROLES,m.D)
    assert np.allclose(y[:,0,0],2*np.arange(5))
    assert np.allclose(y[:,0,1],7.)

def test_role_repair_identity_and_gelu_are_deterministically_initialized():
    a=np.zeros((m.ROLES,m.D),np.float32); s=np.ones_like(a); o=np.zeros_like(a); os=np.ones_like(a)
    n1=m.RoleRepair(4,a,s,o,os,'gelu',17); n2=m.RoleRepair(4,a,s,o,os,'gelu',17)
    x=torch.randn(7,m.ROLES,m.D)
    assert torch.equal(n1.w1,n2.w1) and torch.equal(n1.w2,n2.w2)
    y=n1(x); assert y.shape==x.shape and torch.isfinite(y).all()
    y.square().mean().backward(); assert n1.w2.grad is not None

def test_reserved_article_groups_have_48_disjoint_fresh_rows():
    data=json.loads((ROOT/'prompt_manifest.json').read_text())
    assert set(data['buckets'])=={'0','1','2','3'}
    rows=[]; titles=[]; texts=[]
    for key,items in data['buckets'].items():
        assert len(items)==48
        rows.append({x['row'] for x in items}); titles.append({x['title_sha256'] for x in items}); texts.append({x['text_sha256'] for x in items})
    for i in range(4):
        for j in range(i):
            assert rows[i].isdisjoint(rows[j]) and titles[i].isdisjoint(titles[j]) and texts[i].isdisjoint(texts[j])

def test_actual_payload_roundtrip_for_all_modes():
    rng=np.random.default_rng(22); base=rng.normal(size=(m.ROLES,m.D+1,m.D)).astype(np.float32)
    films={}; full={}; repairs={}
    for pair in m.PAIRS:
        pk='>'.join(pair); films[pk]=(np.ones((m.ROLES,1),np.float32),np.zeros((m.ROLES,m.D),np.float32))
        full[pk]=base.copy()
        z=np.zeros((m.ROLES,m.D),np.float32); o=np.ones_like(z)
        repairs[pk]=m.RoleRepair(4,z,o,z,o,'gelu',7)
    for mode in ('shared_linear','pair_film','full_ridge','sparse_linear_residual','sparse_mlp','full_mlp_upper'):
        chosen=repairs if mode=='full_mlp_upper' else {next(iter(repairs)):next(iter(repairs.values()))}
        hard=list(chosen) if mode not in ('shared_linear','pair_film','full_ridge') else []
        raw=m.serialize_payload(mode,base,films,full,chosen,hard); decoded=m.deserialize_payload(raw)
        assert len(raw)>0 and decoded['mode']=={'shared_linear':0,'pair_film':1,'sparse_linear_residual':2,'sparse_mlp':3,'full_ridge':4,'full_mlp_upper':5}[mode]
        if mode in ('shared_linear','pair_film','sparse_linear_residual','sparse_mlp'):
            assert np.array_equal(decoded['base'],base)
        if mode in ('pair_film','sparse_linear_residual','sparse_mlp'):
            assert np.array_equal(decoded['films'][next(iter(films))][0],films[next(iter(films))][0])
        if mode in ('full_ridge','full_mlp_upper'):
            assert np.array_equal(decoded['full_maps'][next(iter(full))],full[next(iter(full))])
        if mode in ('sparse_linear_residual','sparse_mlp','full_mlp_upper'):
            assert set(decoded['repairs'])==set(chosen)
            for key in chosen:
                assert torch.equal(decoded['repairs'][key].w1,chosen[key].w1)
