import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import experiment as ma

def test_draw37_rejection_sampling_replay():
 d=json.loads((ROOT/'source'/'draw37_exclusions.json').read_text());pool=d['pool_ids'];seed=bytes.fromhex(d['seed_hex']);ph=bytes.fromhex(d['pool_sha256'])
 assert hashlib.sha256('\n'.join(pool).encode()).digest()==ph
 h=hashlib.sha256(seed+ph+d['rejection_counter'].to_bytes(4,'big')).digest();v=int.from_bytes(h,'big');lim=(1<<256)-((1<<256)%len(pool))
 assert v<lim and v%len(pool)==d['selection_index_zero_based'] and pool[v%len(pool)]==d['selected_id']=='MA-841'

def test_world_has_development_and_fresh_boundary_memories():
 for seed in [84111,84112,84113]:
  dev,basis,audit=ma.make_world(seed)
  assert len(dev)==8 and basis.shape==(4,64) and len(audit)==24
  assert sum(t['aligned'] for t in audit)==16 and sum(not t['aligned'] for t in audit)==8

def test_aligned_causal_properties_and_bytes_pass():
 rows=json.loads((ROOT/'source'/'audit_results.json').read_text())
 for seed in [84111,84112,84113]:
  x=[r for r in rows if r['world']==seed and r['stratum']=='aligned' and r['method']=='mirror']
  assert np.mean([r['sufficiency_norm_mse'] for r in x])<.05
  assert abs(np.mean([r['reactivation_cosine'] for r in x])-1)<.05
  assert abs(np.mean([r['specificity_leakage_ratio'] for r in x])-1)<.05
  assert abs(np.mean([r['necessity_effect_ratio'] for r in x])-1)<.05
  native=[r for r in rows if r['world']==seed and r['method']=='native']
  assert x[0]['serialized_bytes']<=.5*native[0]['serialized_bytes']

def test_mirror_equals_matched_pca_control_and_private_fallback_repairs_residuals():
 rows=json.loads((ROOT/'source'/'audit_results.json').read_text())
 for r in rows:
  if r['method']=='mirror':
   p=[x for x in rows if x['world']==r['world'] and x['memory']==r['memory'] and x['method']=='pca'][0]
   assert r['sufficiency_norm_mse']==p['sufficiency_norm_mse']
  if r['method']=='private' and r['stratum']=='residual':
   assert r['sufficiency_norm_mse']<1e-5 and abs(r['reactivation_cosine']-1)<1e-5

def test_actual_payload_safetensors_roundtrip(tmp_path):
 import torch
 from safetensors.torch import load_file
 dev,basis,audit=ma.make_world(84111);codes=np.zeros((len(audit),4),dtype='float32');traces=np.stack([x['M'] for x in audit])
 path=tmp_path/'engram.safetensors';n,digest=ma.serialize('mirror',basis,codes,traces,np.zeros((0,8,8),dtype='float32'),[],path)
 data=load_file(str(path));assert n==path.stat().st_size and len(digest)==64 and tuple(data['code_bank'].shape)==(24,4)
