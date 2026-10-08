import csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'source';raw=json.loads((SRC/'development_raw.json').read_text());rows=raw['rows']
idx={(r['seed'],r['condition'],r['method'],r['rank']):r for r in rows}
gates=[]
for sd in (7211,7212):
 for rank in (4,8,16):
  per=[]
  for cond in ('clean','corrupt_sigma0.25'):
   m=idx[(sd,cond,'mirror_rademacher',rank)];s=idx[(sd,cond,'swag_gaussian',rank)]
   per.append({'condition':cond,'mirror_nll':m['metrics']['nll'],'swag_nll':s['metrics']['nll'],'delta_nll':m['metrics']['nll']-s['metrics']['nll'],'mirror_ece':m['metrics']['ece10'],'swag_ece':s['metrics']['ece10'],'delta_ece':m['metrics']['ece10']-s['metrics']['ece10'],'mirror_bytes':m['serialized_payload_bytes'],'swag_bytes':s['serialized_payload_bytes'],'mirror_wall_s':m['inference_wall_s'],'swag_wall_s':s['inference_wall_s']})
  clean,shift=per
  clean_guard=clean['delta_nll']<=.02 and clean['delta_ece']<=.01
  shift_gain=shift['delta_nll']<=-.02 or shift['delta_ece']<=-.01
  bytes_ok=all(z['mirror_bytes']<=z['swag_bytes'] for z in per)
  compute_ok=all(z['mirror_wall_s']<=1.05*z['swag_wall_s'] for z in per)
  gates.append({'seed':sd,'rank':rank,'conditions':per,'clean_quality_guard':clean_guard,'shift_improvement':shift_gain,'serialized_bytes_not_higher':bytes_ok,'compute_within_5pct':compute_ok,'pass':clean_guard and shift_gain and bytes_ok and compute_ok})
pass_by_rank={str(r):all(x['pass'] for x in gates if x['rank']==r) for r in (4,8,16)}
summary=[]
for method in sorted(set(r['method'] for r in rows)):
 for rank in sorted(set(r['rank'] for r in rows if r['method']==method)):
  for cond in ('clean','corrupt_sigma0.25'):
   subset=[r for r in rows if r['method']==method and r['rank']==rank and r['condition']==cond]
   if not subset:continue
   summary.append({'method':method,'rank':rank,'condition':cond,'mean_nll':statistics.mean(r['metrics']['nll'] for r in subset),'mean_accuracy':statistics.mean(r['metrics']['accuracy'] for r in subset),'mean_ece10':statistics.mean(r['metrics']['ece10'] for r in subset),'mean_brier':statistics.mean(r['metrics']['brier'] for r in subset),'mean_disagreement':statistics.mean(r['metrics'].get('pairwise_disagreement',0) for r in subset),'serialized_bytes':subset[0]['serialized_payload_bytes'],'mean_examples_per_s':statistics.mean(r['examples_per_s'] for r in subset),'mean_weight_reconstruction_ops_per_member':statistics.mean(r['weight_reconstruction_ops_per_member'] for r in subset)})
rawsummary={'experiment_id':'MA-721','status':'FAIL-development-gate; fresh unopened','rows':len(rows),'payloads':len(raw['payloads']),'seeds':[7211,7212],'rank_pass':pass_by_rank,'gate_checks':gates,'rank1_sigma_selection':raw['rank1_sigma_selection'],'summary':summary,'fresh_accessed':False,'audit_labels_used':False,'interpretation':'Rademacher Mirror coordinates over the SWAG basis produced near-identical predictive metrics and did not meet the predeclared shifted NLL/ECE improvement threshold. At rank 4 the sampled payload was smaller than an 8-model ensemble but less accurate; rank 16 exceeded that ensemble bytes. The tuned post-hoc rank-1 factor control matched or exceeded Mirror on corruption with far fewer bytes, at a modest clean-NLL tradeoff.','counter_hypothesis':'The 8x8 digits MLP and a single additive Gaussian corruption are narrow; the low posterior disagreement suggests the SGD trajectory subspace itself may be too small or underdispersed. This says little about larger models, realistic distribution shifts, or fully trained Rank-1 BNNs.','uncertainties':['fresh heldout digit split','additional corruption families and natural OOD data','full Rank-1 BNN variational training','larger networks/datasets','near-convergence or fixed-byte capacity frontier'],'raw_sha256':hashlib.sha256((SRC/'development_raw.json').read_bytes()).hexdigest(),'results_core_sha256':hashlib.sha256((ROOT/'RESULTS_CORE.csv').read_bytes()).hexdigest(),'metric_replay':json.loads((SRC/'metric_replay.json').read_text())}
(SRC/'development_summary.json').write_text(json.dumps(rawsummary,indent=2)+'\n')
# Persistent record of implementation-only corrections. Primary metrics were deterministic and unchanged.
attempts={'attempt_01':{'file':'source/development_attempt_01_duplicate_rank1_rows.json','issue':'rank-1 sigma grid was accidentally nested under each covariance rank, yielding 92 rows with exact duplicate rank-1 rows; no settings were tuned from this attempt.'},'attempt_02':{'file':'source/development_attempt_02_tensor_view_bytes.json','issue':'60-row rerun fixed duplicate reporting, but torch.save serialized the backing storage of sliced covariance basis tensors; actual payload bytes included unused directions. Metric values matched final rerun, but these payload byte results were superseded.'},'final_rerun':{'file':'source/development_raw.json','issue':'same predeclared data/model/seed/rank/sigma settings, corrected rank-1 row placement and cloned rank tensors before serialization; no hyperparameters changed.'}}
(SRC/'attempt_log.json').write_text(json.dumps(attempts,indent=2)+'\n')
