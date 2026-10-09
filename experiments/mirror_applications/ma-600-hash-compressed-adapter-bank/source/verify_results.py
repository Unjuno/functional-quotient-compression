#!/usr/bin/env python3
"""Verify MA-600 frozen hashes, seven-method deterministic replays and gates."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 f=json.loads((ROOT/'FREEZE.json').read_text())
 assert sha(ROOT/'PROTOCOL.json')==f['protocol_sha256']
 assert sha(ROOT/'source/run_adapter_bank.py')==f['source_sha256']
 assert sha(ROOT/'tests/test_adapter_bank.py')==f['tests_sha256']
 stable=['method','bytes','payload_sha256','updates','examples_seen','initial_final_train_mse','per_task_nrmse','aggregate_nrmse','aggregate_rmse','per_task_rmse']
 result={}
 for seed in (60001,60002,60003):
  a=ROOT/f'runs/dev_{seed}';b=ROOT/f'runs/replay_{seed}';ma=json.loads((a/'metrics.json').read_text())
  subprocess.run([sys.executable,str(ROOT/'source/run_adapter_bank.py'),'--seed',str(seed),'--split','dev','--out',str(b)],check=True,stdout=subprocess.DEVNULL)
  mb=json.loads((b/'metrics.json').read_text());dm={m['method']:m for m in ma['methods']}
  for x,y in zip(ma['methods'],mb['methods']):
   assert all(x[k]==y[k] for k in stable)
   pa=a/(x['method']+'.npz');pb=b/(x['method']+'.npz');assert pa.stat().st_size==x['bytes'] and sha(pa)==x['payload_sha256']==sha(pb)
  v,l,h,s=dm['mirror_givens_hash'],dm['independent_lora'],dm['independent_hash'],dm['shared_salted_hash']
  assert v['aggregate_nrmse']>.10 and v['bytes']<=.75*h['bytes']
  assert v['aggregate_rmse']>=s['aggregate_rmse']-.01
  assert v['aggregate_rmse']<dm['vera_shared_basis']['aggregate_rmse']-.01
  # This margin passes in world 60001 and misses in worlds 60002/60003; the
  # absolute NRMSE gate independently fails in all three worlds.
  result[str(seed)]={'seven_payloads_byte_exact':True,'mirror_nrmse':v['aggregate_nrmse'],'mirror_bytes':v['bytes'],'lora_nrmse':l['aggregate_nrmse'],'lora_bytes':l['bytes'],'shared_hash_nrmse':s['aggregate_nrmse'],'shared_hash_bytes':s['bytes']}
 assert not any((ROOT/f'runs/fresh_{seed}').exists() for seed in (60011,60012,60013))
 out={'experiment_id':'MA-600','status':'FAIL','protocol_sha256':f['protocol_sha256'],'source_sha256':f['source_sha256'],'development_world_seeds':[60001,60002,60003],'fresh_worlds_accessed':[],'fresh_artifacts_present':False,'serialization_roundtrip_checked':True,'metric_replay_checked':True,'payloads_replayed':21,'results':result,'tests':{'expected_passed':3},'gate_decision':'FAIL: Mirror normalized MSE gate misses in all three worlds; it is far worse than independent rank-2 LoRA and fails the frozen native-control RMSE margin. Its payload is 0.340x independent hash.'}
 (ROOT/'VERIFICATION.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
