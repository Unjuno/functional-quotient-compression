"""Validate MA-921 development provenance and export results without opening fresh data."""
import csv,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'source'
sha=lambda b:hashlib.sha256(b).hexdigest()
conf=json.loads((SRC/'development_config.json').read_text())
assert sha((SRC/'run_experiment.py').read_bytes())==conf['source_sha256']
assert sha((SRC/'frozen_protocol.json').read_bytes())==conf['protocol_sha256']
assert conf['fresh_accessed'] is False
assert not (SRC/'fresh_500_updates.json').exists() and not (SRC/'fresh_200_updates.json').exists()
all_rows=[];by_budget={}
for updates in (200,500):
 path=SRC/f'development_{updates}_updates.json';raw=path.read_bytes()
 assert sha(raw)==conf['development_runs'][str(updates)]['results_sha256']
 data=json.loads(raw);assert data['seeds']==[9211,9212] and len(data['rows'])==18
 by_budget[updates]=data['rows']
 for r in data['rows']:
  assert r['updates']==updates and len(r['payload_sha256'])==64 and r['serialized_bytes']>0
  for key in ['seen_map_mse','composition_vs_true_normalized_mse','semigroup_discrepancy_normalized_mse']:
   assert math.isfinite(r[key])
  if r['method']!='independent':assert math.isfinite(r['heldout_map_mse'])
  all_rows.append({'world_seed':r['world_seed'],'split':'development','method':r['method'],'rank':r['rank'],'updates':updates,
   'serialized_bytes':r['serialized_bytes'],'payload_sha256':r['payload_sha256'],'train_examples':r['train_examples'],
   'optimizer_updates':r['optimizer_updates'],'train_macs_proxy':r['train_macs_proxy'],'train_wall_s':r['train_wall_s'],
   'inference_macs_proxy':r['inference_macs_proxy'],'inference_wall_s':r['inference_wall_s'],'nfe_direct':r['nfe_direct'],'nfe_composition':r['nfe_composition'],
   'heldout_map_mse':'' if r['heldout_map_mse'] is None else r['heldout_map_mse'],'seen_map_mse':r['seen_map_mse'],
   'composition_vs_true_normalized_mse':r['composition_vs_true_normalized_mse'],'semigroup_discrepancy_normalized_mse':r['semigroup_discrepancy_normalized_mse'],
   'status_note':'development-only; fresh not opened'})
passes=[];candidate_summary=[]
for u,rows in by_budget.items():
 lookup={(r['world_seed'],r['method'],r['rank']):r for r in rows}
 for rank in (2,4,8):
  outcomes=[]
  for seed in (9211,9212):
   m=lookup[(seed,'mirror',rank)];a=lookup[(seed,'additive',rank)];f=lookup[(seed,'fmm',4)]
   gates={'native_fmm_quality':m['heldout_map_mse']<=1.10*f['heldout_map_mse'],
          'mirror_specific_additive_quality':m['heldout_map_mse']<=0.80*a['heldout_map_mse'],
          'semigroup':m['semigroup_discrepancy_normalized_mse']<=0.01,
          'storage':m['serialized_bytes']<=0.70*f['serialized_bytes']}
   outcomes.append({'world_seed':seed,'heldout_ratio_to_native_fmm':m['heldout_map_mse']/f['heldout_map_mse'],
                    'heldout_ratio_to_additive':m['heldout_map_mse']/a['heldout_map_mse'],
                    'semigroup_discrepancy':m['semigroup_discrepancy_normalized_mse'],'mirror_bytes':m['serialized_bytes'],'native_fmm_bytes':f['serialized_bytes'],
                    'gates':gates,'world_pass':all(gates.values())})
  passes.append(all(o['world_pass'] for o in outcomes))
  candidate_summary.append({'updates':u,'rank':rank,'worlds':outcomes,'all_world_development_pass':all(o['world_pass'] for o in outcomes)})
fields=list(all_rows[0])
with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(all_rows)
summary={'status':'FAIL','fresh_accessed':False,'development_candidates':candidate_summary,
 'development_candidates_passing_all_worlds':sum(passes),'development_candidate_count':len(passes),
 'reason':'No rank at either permitted equal update budget passes the additive-control quality gate in both development worlds; therefore the protocol keeps fresh unopened.',
 'native_shared_basis_control':{str(u):[{'seed':r['world_seed'],'heldout_mse':r['heldout_map_mse'],'bytes':r['serialized_bytes']} for r in rows if r['method']=='native_basis'] for u,rows in by_budget.items()},
 'result_rows':len(all_rows)}
(SRC/'verification_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'status':summary['status'],'fresh_accessed':False,'passing_all_worlds':summary['development_candidates_passing_all_worlds'],'candidates':summary['development_candidate_count'],'result_rows':summary['result_rows']},indent=2))
