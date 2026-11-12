import importlib.util,json
from pathlib import Path
import numpy as np
import torch
spec=importlib.util.spec_from_file_location('ma886',Path(__file__).parents[1]/'source'/'experiment.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_target_role_layouts():
 assert m.NLAYERS[m.TARGETS[0]]*m.NHEADS[m.TARGETS[0]]*2==288
 assert m.NLAYERS[m.TARGETS[1]]*m.NHEADS[m.TARGETS[1]]*2==180

def test_global_and_role_map_prediction_shapes():
 rng=np.random.default_rng(2);x=rng.normal(size=(9,5,m.D)).astype(np.float32);w=np.zeros((m.D+1,m.D),np.float32);w[0,0]=2;w[-1,1]=3
 assert m.predict_global(x,w).shape==x.shape
 maps=np.tile(w[None,:,:],(5,1,1));assert np.allclose(m.predict_roles(x,maps),m.predict_global(x,w))

def test_gain_bias_fit_recovers_affine_role_views():
 rng=np.random.default_rng(4);x=rng.normal(size=(7,3,m.D)).astype(np.float32);w=rng.normal(size=(m.D+1,m.D)).astype(np.float32);z=m.predict_global(x,w);g=np.array([[.5],[1.2],[-.2]],np.float32);b=rng.normal(size=(3,m.D)).astype(np.float32);y=z*g[None,:,:]+b[None,:,:]
 data=[{'x':x,'y':y}];gg,bb=m.fit_codes(data,w)
 assert np.allclose(gg,g,atol=1e-5) and np.allclose(bb,b,atol=1e-5)

def test_serialized_payload_roundtrip_all_layouts():
 rng=np.random.default_rng(3);shared=rng.normal(size=(m.D+1,m.D)).astype(np.float32);codes={};tg={};full={}
 for t in m.TARGETS:
  nr=m.NLAYERS[t]*m.NHEADS[t]*2;codes[t]=(rng.normal(size=(nr,1)).astype(np.float32),rng.normal(size=(nr,m.D)).astype(np.float32));tg[t]=shared.copy();full[t]=np.tile(shared[None,:,:],(nr,1,1))
 for mode in ('shared_global_view','target_global','full_ridge'):
  raw=m.serialize(mode,shared,codes,tg,full);dec=m.deserialize(raw);assert len(raw)>0
  if mode=='shared_global_view':
   assert np.array_equal(dec['shared'],shared)
   for t in m.TARGETS: assert all(np.array_equal(a,b) for a,b in zip(dec['codes'][t],codes[t]))
  elif mode=='target_global':
   for t in m.TARGETS: assert np.array_equal(dec['target_global'][t],tg[t])
  else:
   for t in m.TARGETS: assert np.array_equal(dec['full'][t],full[t])

def test_prompt_manifest_disjoint_from_ma883_and_between_buckets():
 p=json.loads((Path(__file__).parents[1]/'prompt_manifest.json').read_text());old=json.loads((Path(__file__).parents[2]/'ma-883-sparse-nonlinear-repair'/'prompt_manifest.json').read_text())
 oldtitles={x['title_sha256'] for v in old['buckets'].values() for x in v};oldtexts={x['text_sha256'] for v in old['buckets'].values() for x in v}
 for key,items in p['buckets'].items():
  assert len(items)==48 and not ({x['title_sha256'] for x in items}&oldtitles) and not ({x['text_sha256'] for x in items}&oldtexts)
 for i in range(4):
  for j in range(i):
   a=p['buckets'][str(i)];b=p['buckets'][str(j)]
   assert {x['title_sha256'] for x in a}.isdisjoint(x['title_sha256'] for x in b)
   assert {x['text_sha256'] for x in a}.isdisjoint(x['text_sha256'] for x in b)
