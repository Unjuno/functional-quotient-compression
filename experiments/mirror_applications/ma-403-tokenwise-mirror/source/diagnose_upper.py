"""Post-development diagnostic; does not modify or promote the frozen MA-403 gate."""
from pathlib import Path
import datetime,hashlib,json,sys
import torch
from experiment import ROOT, Adapter, build_world, fit, metrics, save_model, write_csv

torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
out=ROOT/'runs'/'upper_budget_diagnostic';out.mkdir(parents=True,exist_ok=True)
spec={'diagnostic':'POST_HOC_DEV_ONLY','registered_before_diagnostic_measurements_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'reason':'At 800 steps the full-matrix reference misses the frozen 0.03 OOD KL upper-control gate in both rotation worlds although Mirror itself has low KL.',
      'hypothesis':'A fourfold fixed update budget reduces full-matrix reference OOD KL below the original 0.03 threshold.',
      'seeds':protocol['dev_seeds'],'world':'rotation','method':'full_matrix','updates':3200,
      'changes':'From-scratch training at 3200 updates with cosine schedule stretched to 3200. Not a continuation with retained optimizer state.',
      'success':'Both observed development worlds <=0.03 OOD KL. This is not a new MA adoption result.',
      'fresh_seeds':'Remain sealed; none generated','original_verdict':'FAIL retained regardless of diagnostic outcome.'}
(out/'DIAGNOSTIC_PROTOCOL.json').write_text(json.dumps(spec,indent=2)+'\n')
rows=[]
for seed in spec['seeds']:
 b,data=build_world(protocol,seed,'rotation')
 a=Adapter(protocol['architecture'],'full_matrix',seed+300000)
 t=protocol['train'];record=fit(b,a,data['train'],updates=3200,batch=t['batch_sequences'],lr=t['learning_rate'],seed=seed+400000,record_every=400)
 payload=save_model(out/f'{seed}_inference.bin',b,a)
 row={'seed':seed,'updates':3200,'full_bytes':payload['bytes'],'payload_sha256':payload['sha256'],'train_seconds':record['elapsed_s']}
 for split in ['train','iid','ood']:row.update({f'{split}_{k}':v for k,v in metrics(b,a,data[split]).items()})
 rows.append(row)
 (out/f'{seed}_training.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(row),flush=True)
write_csv(out/'RESULTS_CORE.csv',rows)
(out/'CONCLUSION.json').write_text(json.dumps({'diagnostic_gate_pass':all(r['ood_excess_kl']<=.03 for r in rows),'original_MA403_verdict':'FAIL','fresh_executed':False,'interpretation':'Longer-budget observed-world diagnostic only; no fresh-world validation or capacity conclusion.'},indent=2)+'\n')
