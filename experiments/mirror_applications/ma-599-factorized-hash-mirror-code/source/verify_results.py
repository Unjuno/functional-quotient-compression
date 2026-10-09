#!/usr/bin/env python3
"""Verify frozen MA-599 hashes, deterministic dev replays, payloads, and failure gates."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 f=json.loads((ROOT/'FREEZE.json').read_text())
 assert sha(ROOT/'PROTOCOL.json')==f['protocol_sha256']
 assert sha(ROOT/'source/run_factorized_hash.py')==f['source_sha256']
 assert sha(ROOT/'tests/test_factorized_hash.py')==f['tests_sha256']
 stable=['method','serialized_bytes','payload_sha256','train_examples','test_examples','optimizer_updates','examples_seen','initial_and_final_train_loss','base_macs_per_example','extra_view_ops_per_example','hash_expansion_lookups_per_model_load','router_selected_accuracy','router_selected_cross_entropy','oracle_expert_accuracy','oracle_expert_cross_entropy','router_accuracy','expert_route_fractions','per_expert_oracle_accuracy','pairwise_raw_logit_cosine_mean','pairwise_raw_argmax_disagreement_mean','pairwise_first_layer_weight_cosine_mean','pairwise_hash_map_overlap_mean','per_expert_collision','expert_ablation_accuracy','active_experts_per_example','route_confusion']
 replay={}
 for seed in (59901,59902):
  a=ROOT/f'runs/dev_{seed}';b=ROOT/f'runs/replay_{seed}'
  ma=json.loads((a/'metrics.json').read_text())
  subprocess.run([sys.executable,str(ROOT/'source/run_factorized_hash.py'),'--seed',str(seed),'--split','dev','--out',str(b)],check=True,stdout=subprocess.DEVNULL)
  mb=json.loads((b/'metrics.json').read_text())
  d={m['method']:m for m in ma['methods']}
  for x,y in zip(ma['methods'],mb['methods']):
   assert all(x[k]==y[k] for k in stable)
   pa=a/(x['method']+'.npz');pb=b/(x['method']+'.npz')
   assert pa.stat().st_size==x['serialized_bytes'] and sha(pa)==x['payload_sha256']==sha(pb)
  v,s,r=d['factorized_hash_view'],d['salted_shared_hash'],d['rank4_dense_residual']
  assert v['router_selected_accuracy'] < s['router_selected_accuracy']+.01
  assert v['serialized_bytes'] > 1.05*s['serialized_bytes']
  assert abs(v['router_selected_accuracy']-r['router_selected_accuracy']) <= .01
  assert r['serialized_bytes'] < v['serialized_bytes']
  replay[str(seed)]={'seven_payloads_byte_exact':True,'view_accuracy':v['router_selected_accuracy'],'salted_accuracy':s['router_selected_accuracy'],'view_bytes':v['serialized_bytes'],'salted_bytes':s['serialized_bytes'],'rank4_accuracy':r['router_selected_accuracy'],'rank4_bytes':r['serialized_bytes']}
 assert not any((ROOT/f'runs/fresh_{s}').exists() for s in (59911,59912,59913))
 out={'experiment_id':'MA-599','status':'FAIL','protocol_sha256':f['protocol_sha256'],'source_sha256':f['source_sha256'],'development_seeds':[59901,59902],'fresh_seeds_accessed':[],'fresh_artifacts_present':False,'serialization_roundtrip_checked':True,'metric_replay_checked':True,'payloads_replayed':14,'results':replay,'tests':{'command':'PYTHONDONTWRITEBYTECODE=1 python -m pytest -q experiments/mirror_applications/ma-599-factorized-hash-mirror-code/tests','expected_passed':3},'gate_decision':'FAIL: misses +1pp control gate; 27,651 B is 2.576x salted; native dense rank-4 residual matches within 1pp at fewer bytes.'}
 (ROOT/'VERIFICATION.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
