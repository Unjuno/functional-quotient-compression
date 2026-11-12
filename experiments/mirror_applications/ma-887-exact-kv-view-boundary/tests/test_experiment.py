import importlib.util,json
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('ma887',Path(__file__).parents[1]/'source'/'experiment.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_rope_inverse_and_global_shift_are_exact_algebra():
 class C: model_type='gpt_neox';hidden_size=512;num_attention_heads=8;rotary_pct=.25;rope_theta=10000.
 rng=np.random.default_rng(6);x=rng.normal(size=(8,32,64)).astype(np.float32)
 recovered=m.rotate_np(m.rotate_np(x,C(),offset=0),C(),offset=0,inverse=True)
 shifted=m.rotate_np(x,C(),offset=17)
 key_zero=m.rotate_np(x,C(),offset=0);canonical=m.rotate_np(key_zero,C(),offset=0,inverse=True);shift_from_canonical=m.rotate_np(canonical,C(),offset=17)
 assert np.max(np.abs(recovered-x))<2e-6
 assert np.max(np.abs(shifted-shift_from_canonical))<2e-6

def test_full_ridge_payload_serialization_roundtrip():
 rng=np.random.default_rng(8);maps=rng.normal(size=(180,65,64)).astype(np.float32);raw=m.serialize_full(maps);got=m.deserialize_full(raw)
 assert np.array_equal(got,maps)

def test_phase_coordinate_serialization_and_bytes():
 raw=m.serialize_phase(17);assert m.deserialize_phase(raw)==17
 assert len(raw)<128

def test_reserved_prompts_disjoint_from_both_prior_experiments():
 root=Path(__file__).parents[1];p=json.loads((root/'prompt_manifest.json').read_text());old=[]
 for folder in ('ma-883-sparse-nonlinear-repair','ma-886-cross-model-kv-fanout'):
  x=json.loads((root.parents[0]/folder/'prompt_manifest.json').read_text());old.extend(z for group in x['buckets'].values() for z in group)
 oldt={z['title_sha256'] for z in old};oldx={z['text_sha256'] for z in old}
 for i in range(4):
  a=p['buckets'][str(i)];assert len(a)==48 and not ({z['title_sha256'] for z in a}&oldt) and not ({z['text_sha256'] for z in a}&oldx)
  for j in range(i):
   b=p['buckets'][str(j)];assert {z['title_sha256'] for z in a}.isdisjoint(z['title_sha256'] for z in b);assert {z['text_sha256'] for z in a}.isdisjoint(z['text_sha256'] for z in b)
