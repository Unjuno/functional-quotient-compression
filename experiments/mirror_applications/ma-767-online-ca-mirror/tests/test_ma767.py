import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import experiment as ma

def test_draw36_rejection_sampling_replay():
 d=json.loads((ROOT/'source'/'draw36_exclusions.json').read_text());pool=d['pool_ids'];seed=bytes.fromhex(d['seed_hex']);ph=bytes.fromhex(d['pool_sha256'])
 assert hashlib.sha256('\n'.join(pool).encode()).digest()==ph
 h=hashlib.sha256(seed+ph+d['rejection_counter'].to_bytes(4,'big')).digest();v=int.from_bytes(h,'big');limit=(1<<256)-((1<<256)%len(pool))
 assert v<limit and v%len(pool)==d['selection_index_zero_based'] and pool[v%len(pool)]==d['selected_id']=='MA-767'

def test_fresh_data_strata_and_rows():
 rows=json.loads((ROOT/'source'/'audit_results.json').read_text());assert len(rows)==3072
 for seed in [76711,76712,76713]:
  assert len({r['task'] for r in rows if r['world']==seed})==64
  assert {r['stratum'] for r in rows if r['world']==seed}=={'easy','hard'}

def test_registered_easy_quality_storage_and_retention_gates():
 rows=json.loads((ROOT/'source'/'audit_results.json').read_text())
 for seed in [76711,76712,76713]:
  m=[r for r in rows if r['world']==seed and r['stratum']=='easy' and r['method']=='mirror' and r['online_examples']==64]
  f=[r for r in rows if r['world']==seed and r['stratum']=='easy' and r['method']=='full_rule_bank' and r['online_examples']==64]
  assert abs(np.mean([x['accuracy'] for x in m])-np.mean([x['accuracy'] for x in f]))<=.05
  assert m[0]['serialized_bytes']<=.6*f[0]['serialized_bytes']
  retain=[r for r in rows if r['world']==seed and r['method']=='mirror_bank_after_all']
  assert len(retain)==64 and all(r['retention_ratio']==1.0 for r in retain)

def test_code_views_add_no_clear_accuracy_over_frozen_control():
 rows=json.loads((ROOT/'source'/'audit_results.json').read_text())
 m=[r['accuracy'] for r in rows if r['method']=='mirror' and r['online_examples']==64]
 f=[r['accuracy'] for r in rows if r['method']=='frozen_shared']
 assert abs(np.mean(m)-np.mean(f))<.01

def test_contiguous_bank_payload_roundtrip(tmp_path):
 import torch
 from safetensors.torch import load_file
 base=np.zeros(8,dtype='float32');basis=np.zeros((8,2),dtype='float32');states=[np.zeros(2,dtype='float32') for _ in range(64)]
 p=tmp_path/'bank.safetensors';n,digest=ma.pack('mirror',base,basis,states,p)
 data=load_file(str(p));assert n==p.stat().st_size and len(digest)==64 and tuple(data['code_bank'].shape)==(64,2)
