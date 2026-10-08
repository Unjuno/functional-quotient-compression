import csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'source';raw=json.loads((SRC/'development_raw.json').read_text());rows=raw['rows'];idx={(r['seed'],r['rho'],r['task_id'],r['method'],r['rank']):r for r in rows}
gates=[]
for seed in raw['seeds']:
 for rho in raw['heterogeneity_rho']:
  for task in (6,7):
   ind=idx[(seed,rho,task,'independent',0)]
   for rank in (1,2,4,8):
    m=idx[(seed,rho,task,'mirror_cp',rank)]
    controls=[idx[(seed,rho,task,method,rank)] for method in ('matrix_lowrank','tt_hyper')]
    near=[c for c in controls if .90*m['library_bytes']<=c['library_bytes']<=1.10*m['library_bytes']]
    best_near=max(near,key=lambda c:c['process_fidelity']) if near else None
    quality=m['process_fidelity']>=.98
    independent_bytes=m['library_bytes']<=.80*ind['library_bytes']
    control_bytes=any(m['library_bytes']<=.95*c['library_bytes'] for c in near)
    specific=bool(best_near and m['process_fidelity']>=best_near['process_fidelity']+.01)
    gates.append({'seed':seed,'rho':rho,'task_id':task,'rank':rank,'mirror_fidelity':m['process_fidelity'],'independent_fidelity':ind['process_fidelity'],'mirror_bytes':m['library_bytes'],'independent_bytes':ind['library_bytes'],'byte_near_controls':[{'method':c['method'],'bytes':c['library_bytes'],'fidelity':c['process_fidelity']} for c in near],'quality':quality,'independent_storage_ratio':independent_bytes,'simple_control_storage':control_bytes,'mirror_specific_gain':specific,'pass':quality and independent_bytes and control_bytes and specific})
pass_by_rank={str(r):all(x['pass'] for x in gates if x['rank']==r) for r in (1,2,4,8)}
summary=[]
for method in sorted({r['method'] for r in rows}):
 for rank in sorted({r['rank'] for r in rows if r['method']==method}):
  for rho in raw['heterogeneity_rho']:
   ss=[r for r in rows if r['method']==method and r['rank']==rank and r['rho']==rho]
   summary.append({'method':method,'rank':rank,'rho':rho,'mean_process_fidelity':statistics.mean(r['process_fidelity'] for r in ss),'mean_basis_state_fidelity':statistics.mean(r['mean_basis_state_fidelity'] for r in ss),'whole_library_bytes':ss[0]['library_bytes'],'mean_condition_state_bytes':statistics.mean(r['condition_state_bytes'] for r in ss),'mean_simulator_wall_s':statistics.mean(r['inference_wall_s'] for r in ss)})
result={'experiment_id':'MA-1018','status':'FAIL-development-gate; fresh unopened','rows':len(rows),'library_payloads':len(raw['libraries']),'seeds':raw['seeds'],'heterogeneity_rho':raw['heterogeneity_rho'],'rank_pass':pass_by_rank,'gate_checks':gates,'summary':summary,'fresh_accessed':False,'interpretation':'On the CP-aligned synthetic teacher, CP Mirror rank 2, matrix low-rank rank 2, and TT rank 2 all achieved approximately 0.9999 process fidelity. CP rank 2 used slightly more bytes than TT rank 2 and did not yield the required >=0.01 fidelity gain over byte-near controls. A rank-1 CP point at rho 0.4 retained >0.98 fidelity but did not beat the byte-near TT rank-1 control by the required margin. Private top-8 angle residuals added bytes for negligible fidelity gains.','counter_hypothesis':'The task family is synthetic and deliberately CP-structured, and held-out Mirror codes are fitted from known target angles. This is an oracle compression result, not task-data learning; other unitary families may favor different representations.','uncertainties':['fresh random task families','real quantum classification tasks','shot noise/device noise','hardware transpilation/connectivity','gradient/trainability behavior','QPU execution and any practical quantum advantage'],'raw_sha256':hashlib.sha256((SRC/'development_raw.json').read_bytes()).hexdigest(),'results_core_sha256':hashlib.sha256((ROOT/'RESULTS_CORE.csv').read_bytes()).hexdigest(),'metric_replay':json.loads((SRC/'metric_replay.json').read_text()),'wall_total_s':raw['wall_total_s']}
(SRC/'development_summary.json').write_text(json.dumps(result,indent=2)+'\n')
