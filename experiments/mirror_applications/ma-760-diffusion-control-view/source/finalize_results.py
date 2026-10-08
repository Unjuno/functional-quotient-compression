import csv,json,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'source';raw=json.loads((SRC/'development_raw.json').read_text());rows=raw['results']
fields=['condition','world_or_seed','rho','condition_id','method','rank','serialized_bytes','whole_library_bytes','marginal_condition_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','normalized_residual_mse','status_note']
with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
 for r in rows:
  w.writerow({'condition':r['condition'],'world_or_seed':r['world_or_seed'],'rho':r['rho'],'condition_id':r['condition_id'],'method':r['method'],'rank':r['rank'],'serialized_bytes':r['adapter_payload_bytes'],'whole_library_bytes':r['whole_library_payload_bytes'],'marginal_condition_bytes':r['marginal_condition_payload_bytes'],'train_tokens_or_examples':r['train_tokens_or_examples'],'optimizer_updates':r['optimizer_updates'],'active_compute_proxy':r['active_compute_proxy'],'wall_time_s':r['wall_time_s'],'primary_metric':r['primary_metric'],'primary_value':r['primary_value'],'normalized_residual_mse':r['normalized_residual_mse'],'status_note':'fresh unopened; tiny diffusion UNet mechanism screen'})
g=defaultdict(list)
for r in rows:g[(r['world_or_seed'],r['rho'],r['condition_id'],r['method'],r['rank'])].append(r)
# Per-seed/condition gate table; independent direct fits are the upper control.
gates=[]
for seed in (7601,7602):
 for rho in (0.1,0.4,1.0):
  for c in (6,7):
   ind=g[(seed,rho,c,'independent',0)][0]
   for rank in (1,2,4,8):
    mp=g[(seed,rho,c,'mirror_private',rank)][0];mir=g[(seed,rho,c,'mirror',rank)][0];add=g[(seed,rho,c,'additive',rank)][0]
    quality=mp['primary_value'] <= 1.1*max(ind['primary_value'],1e-12)
    whole=mp['whole_library_payload_bytes'] < ind['whole_library_payload_bytes']
    marginal=mp['marginal_condition_payload_bytes'] <= 0.5*ind['marginal_condition_payload_bytes']
    mirror_specific=mir['primary_value'] <= 0.95*add['primary_value']
    gates.append({'seed':seed,'rho':rho,'condition_id':c,'rank':rank,'private_mse':mp['primary_value'],'independent_mse':ind['primary_value'],'mirror_mse':mir['primary_value'],'additive_mse':add['primary_value'],'quality_within_10pct_independent':quality,'whole_payload_strictly_smaller':whole,'private_marginal_half_independent':marginal,'mirror_5pct_better_than_additive':mirror_specific})
# Exact method/rank summary.
summary=[]
for method in ('no_control','hard_shared','independent','additive','mirror','mirror_private'):
 ranks=(1,2,4,8) if method in ('additive','mirror','mirror_private') else (0,)
 for rank in ranks:
  for rho in (0.1,0.4,1.0):
   subset=[r for r in rows if r['method']==method and r['rank']==rank and r['rho']==rho]
   summary.append({'method':method,'rank':rank,'rho':rho,'mean_heldout_mse':statistics.mean(r['primary_value'] for r in subset),'mean_normalized_mse':statistics.mean(r['normalized_residual_mse'] for r in subset),'whole_library_bytes':subset[0]['whole_library_payload_bytes'],'mean_marginal_condition_bytes':statistics.mean(r['marginal_condition_payload_bytes'] for r in subset),'mean_query_examples_per_s':statistics.mean(r['query_throughput_examples_s'] for r in subset)})
# Development gate is conjunction over all preregistered seeds, heterogeneity levels and two held-out identities.
pass_by_rank={}
for rank in (1,2,4,8):
 points=[x for x in gates if x['rank']==rank]
 pass_by_rank[str(rank)]=all(x['quality_within_10pct_independent'] and x['whole_payload_strictly_smaller'] and x['private_marginal_half_independent'] and x['mirror_5pct_better_than_additive'] for x in points)
raw['development_summary']={'rows':len(rows),'base_seeds':[7601,7602],'heterogeneity_rho':[0.1,0.4,1.0],'heldout_conditions':[6,7],'rank_pass':pass_by_rank,'gate_checks':gates,'fresh_accessed':False,'decision':'FAIL-development-gate; fresh remains unopened','facts':['Independent per-condition adapters fit the synthetic linear target residuals to numerical zero on heldout maps.','Product Mirror and additive low-rank controls have exactly equal heldout MSE for every tested condition and rank.','Private sparse residual lowers residual MSE relative to the corresponding Mirror-only code, but its per-condition serialized state exceeds the independent adapter state in this small 108-parameter fixture.','Some low-rank Mirror whole-library bundles are smaller than independent, but no rank passes the conjunction against the near-zero independent quality upper control and matched additive control.','The common frozen tiny UNet payload is 5,830,450 bytes; adapter-library deltas are only hundreds to a few thousand bytes.'],'interpretation':'The screen shows a storage/quality/private-state tradeoff on synthetic linear spatial control operators, while the multiplicative View is algebraically equivalent to an additive low-rank basis in this construction. The result does not establish utility on real diffusion controls.','counter_hypothesis':'The outcome is dominated by an artificially favorable exact-linear independent control and a tiny test UNet whose base weights overwhelm adapter storage; realistic ControlNet/T2I-Adapter branches may have a different break-even frontier.','uncertainties':['real edge/depth/pose paired data','quality-pretrained diffusion UNet','image-level generation metrics','full ControlNet/T2I-Adapter architecture and branches','GPU memory/FPS/energy','fresh condition families'],'summary':summary,'raw_sha256':None,'results_core_sha256':None}
(SRC/'development_summary.json').write_text(json.dumps(raw['development_summary'],indent=2)+'\n')
raw['raw_sha256']=None
(SRC/'development_summary.json').write_text(json.dumps(raw['development_summary'],indent=2)+'\n')
