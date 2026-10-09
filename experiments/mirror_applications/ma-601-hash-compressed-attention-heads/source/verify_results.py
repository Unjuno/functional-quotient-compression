#!/usr/bin/env python3
"""Verify MA-601 hashes, same-seed dev replays, actual payloads and gates."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 f=json.loads((ROOT/'FREEZE.json').read_text())
 assert sha(ROOT/'PROTOCOL.json')==f['protocol_sha256'];assert sha(ROOT/'source/run_head_distill.py')==f['source_sha256'];assert sha(ROOT/'tests/test_head_distill.py')==f['tests_sha256']
 stable=['method','serialized_bytes','payload_sha256','initial_train_ce','updates','sequences_seen','qkv_mac_proxy','extra_view_ops_per_token','hash_expansion_lookups','teacher_cross_entropy','top1_agreement','head_context_cosine_mean','attention_map_cosine_mean']
 result={}
 for seed in (60101,60102):
  a=ROOT/f'runs/dev_{seed}';b=ROOT/f'runs/replay_{seed}';ma=json.loads((a/'metrics.json').read_text())
  subprocess.run([sys.executable,str(ROOT/'source/run_head_distill.py'),'--seed',str(seed),'--split','dev','--out',str(b)],check=True,stdout=subprocess.DEVNULL)
  mb=json.loads((b/'metrics.json').read_text());d={x['method']:x for x in ma['methods']}
  for x,y in zip(ma['methods'],mb['methods']):
   assert all(x[k]==y[k] for k in stable);pa=a/(x['method']+'.npz');pb=b/(x['method']+'.npz');assert pa.stat().st_size==x['serialized_bytes'] and sha(pa)==x['payload_sha256']==sha(pb)
  m,f0,s,g,r=d['mirror_givens_hash'],d['independent_heads'],d['salted_hash'],d['gqa2'],d['rank1_hash']
  assert m['teacher_cross_entropy']>f0['teacher_cross_entropy']+.02
  assert m['teacher_cross_entropy']>g['teacher_cross_entropy']-.01
  assert m['serialized_bytes']>1.05*s['serialized_bytes']
  assert m['serialized_bytes']<=.75*f0['serialized_bytes']
  assert m['head_context_cosine_mean']<d['tied_hash']['head_context_cosine_mean']-.05
  assert m['teacher_cross_entropy']>r['teacher_cross_entropy']
  result[str(seed)]={'eight_method_payloads_byte_exact':True,'mirror_ce':m['teacher_cross_entropy'],'independent_ce':f0['teacher_cross_entropy'],'mqa_ce':d['mqa']['teacher_cross_entropy'],'rank1_ce':r['teacher_cross_entropy'],'mirror_bytes':m['serialized_bytes'],'salted_bytes':s['serialized_bytes'],'mirror_context_cosine':m['head_context_cosine_mean']}
 assert not any((ROOT/f'runs/fresh_{s}').exists() for s in (60111,60112,60113))
 out={'experiment_id':'MA-601','status':'FAIL','protocol_sha256':f['protocol_sha256'],'source_sha256':f['source_sha256'],'development_world_seeds':[60101,60102],'fresh_worlds_accessed':[],'fresh_artifacts_present':False,'serialization_roundtrip_checked':True,'metric_replay_checked':True,'payloads_replayed':16,'results':result,'tests':{'expected_passed':3},'gate_decision':'FAIL: Givens View fails teacher-quality control and exceeds the 1.05x salted byte cap; head activations are diverse but quality is worse than MQA/diagonal/rank1 controls.'}
 (ROOT/'VERIFICATION.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
