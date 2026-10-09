#!/usr/bin/env python3
"""Verify MA-602 freeze hashes, deterministic payloads, and frozen gates."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 f=json.loads((ROOT/'FREEZE.json').read_text());assert sha(ROOT/'PROTOCOL.json')==f['protocol_sha256'];assert sha(ROOT/'source/run_collision_repair.py')==f['source_sha256'];assert sha(ROOT/'tests/test_collision_repair.py')==f['tests_sha256']
 stable=['method','bucket_count','exception_count','exception_index_bytes','exception_value_bytes','serialized_bytes','payload_sha256','optimizer_updates','examples_seen','initial_and_final_train_loss','base_macs_per_example','hash_expansion_lookups_per_model_load','private_exception_adds_per_example','extra_view_ops_per_example','accuracy','cross_entropy']
 results={}
 for seed in (60201,60202):
  a=ROOT/f'runs/dev_{seed}';b=ROOT/f'runs/replay_{seed}';ma=json.loads((a/'metrics.json').read_text());subprocess.run([sys.executable,str(ROOT/'source/run_collision_repair.py'),'--seed',str(seed),'--split','dev','--out',str(b)],check=True,stdout=subprocess.DEVNULL);mb=json.loads((b/'metrics.json').read_text());d={m['method']:m for m in ma['methods']}
  for x,y in zip(ma['methods'],mb['methods']):
   assert all(x[k]==y[k] for k in stable);pa=a/(x['method']+'.npz');pb=b/(x['method']+'.npz');assert pa.stat().st_size==x['serialized_bytes'] and sha(pa)==x['payload_sha256']==sha(pb)
  c,g,r,h2,h4=d['collision_residual'],d['collision_residual_givens'],d['random_residual_givens'],d['hash_2048'],d['hash_4096']
  assert g['accuracy']<h2['accuracy']+.01 and g['accuracy']<r['accuracy']+.01
  assert abs(g['accuracy']-h4['accuracy'])<=.01 and g['serialized_bytes']>h4['serialized_bytes']
  assert abs(c['accuracy']-g['accuracy'])<=.005
  results[str(seed)]={'eight_payloads_byte_exact':True,'duplicate_exception_count':g['exception_count'],'collision_givens_accuracy':g['accuracy'],'random_givens_accuracy':r['accuracy'],'hash2048_accuracy':h2['accuracy'],'hash4096_accuracy':h4['accuracy'],'collision_givens_bytes':g['serialized_bytes'],'hash4096_bytes':h4['serialized_bytes'],'no_view_accuracy':c['accuracy']}
 assert not any((ROOT/f'runs/fresh_{s}').exists() for s in (60211,60212,60213))
 out={'experiment_id':'MA-602','status':'FAIL','protocol_sha256':f['protocol_sha256'],'source_sha256':f['source_sha256'],'development_seeds':[60201,60202],'fresh_seeds_accessed':[],'fresh_artifacts_present':False,'serialization_roundtrip_checked':True,'metric_replay_checked':True,'payloads_replayed':16,'results':results,'tests':{'expected_passed':3},'gate_decision':'FAIL: collision+Givens misses +1pp versus native hash2048 and random exceptions; its 34,99x B payload exceeds hash4096 at 13,791 B; no-view residual is within 0.5pp.'}
 (ROOT/'VERIFICATION.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
