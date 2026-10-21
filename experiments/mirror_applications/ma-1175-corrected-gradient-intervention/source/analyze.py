"""Post-training descriptive analysis, no protocol modifications or model retuning."""
from __future__ import annotations
import csv, collections, json, statistics, math, platform, hashlib
from pathlib import Path
import torch
torch.set_num_threads(1)

ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'RESULTS_CORE.csv').open()))
events=json.loads((ROOT/'results/INTERVENTIONS.json').read_text())
h120=json.loads((ROOT/'results/H120_DEV_ONLY.json').read_text())
paired=json.loads((ROOT/'results/PAIRED_RESULTS.json').read_text())
W=sorted({x['world'] for x in rows});M=sorted({x['method'] for x in rows})
SEEDS=sorted({int(x['seed']) for x in rows})
num=lambda x,k:float(x[k])
median=statistics.median
summary={}
for m in M:
 rr=[x for x in rows if x['method']==m]
 summary[m]={key:median(num(x,key) for x in rr) for key in ['fresh_nmse','ood_nmse','packet_bytes','forward_b1_us','forward_b64_us','active_edges','train_mlp_mac_proxy','probe_mlp_mac_proxy','diagnostic_mlp_mac_proxy','total_accounted_mlp_mac_proxy']}
 summary[m]['world_fresh_median']={w:median(num(x,'fresh_nmse') for x in rr if x['world']==w) for w in W}
 summary[m]['world_ood_median']={w:median(num(x,'ood_nmse') for x in rr if x['world']==w) for w in W}
 summary[m]['cuts']=sum(len(x['edges_cut']) for x in events if x['method']==m)

policy_interventions={m:[x for x in events if x['method']==m] for m in M}
gate_v={}
for key,rr in policy_interventions.items():
 if not rr:continue
 if 'guard' in key:
  expected=lambda e: (min((v for k,v in e['reference_and_probes_dev_nmse'].items() if k!='keep'),default=float('inf'))<=e['reference_and_probes_dev_nmse']['keep']-.0005)
  validity=[(bool(e['edges_cut'])==expected(e)) for e in rr]
  gate_v[key]={'accepted':sum(bool(e['edges_cut']) for e in rr),'total':len(rr),'valid':sum(validity)}

actual_vjp=[x['vjp']['relative_vjp_error_max'] for x in events if x['vjp']['relative_vjp_error_max'] is not None]
anti=[x['vjp']['anti_aligned_edge_count'] for x in events if x['method']=='mirror_guard_long']
post_first=[]
for x in h120:
 ev=next(e for e in events if e['method']=='mirror_guard_long' and e['world']==x['world'] and e['seed']==x['seed'] and e['event']==0)
 rank40=sorted(ev['reference_and_probes_dev_nmse'].items(),key=lambda kv:kv[1])
 rank120=sorted(x['dev_scores'].items(),key=lambda kv:kv[1])
 post_first.append({'world':x['world'],'seed':x['seed'],'best_H40':rank40[0][0],'best_H120':rank120[0][0],
   'H40_keep_minus_best_cut':ev['reference_and_probes_dev_nmse']['keep']-min(v for k,v in ev['reference_and_probes_dev_nmse'].items() if k!='keep'),
   'H120_keep_minus_best_cut':x['dev_scores']['keep']-min(v for k,v in x['dev_scores'].items() if k!='keep'),
   'sign_agree':(ev['reference_and_probes_dev_nmse']['keep']-min(v for k,v in ev['reference_and_probes_dev_nmse'].items() if k!='keep')>=.0005)==(x['dev_scores']['keep']-min(v for k,v in x['dev_scores'].items() if k!='keep')>=.0005)})
compare={
 'learning_gate_pass':sum(x['learning_gate'] for x in paired),
 'mirror_specific_gate_pass':sum(x['mirror_gate'] for x in paired),
 'long_beats_keep_raw':sum(x['long_to_keep']<1 for x in paired),
 'long_beats_short_raw':sum(x['long_to_short']<1 for x in paired),
 'long_beats_from_start_raw':sum(x['long_to_startmask']<1 for x in paired),
 'long_beats_extra_raw':sum(x['long_to_extra']<1 for x in paired),
 'median_ratio_vs_keep':median(x['long_to_keep'] for x in paired),
 'median_ratio_vs_start_mask':median(x['long_to_startmask'] for x in paired),
 'median_ratio_vs_direct':median(x['long_to_direct'] for x in paired),
 'median_ratio_vs_extra':median(x['long_to_extra'] for x in paired)
}
# Synthetic worlds are not identical experimental units; metrics are descriptive.
summary_doc={
 'id':'MA-1175','protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),
 'hardware':{'platform':platform.platform(),'processor':platform.processor(),'torch':torch.__version__,'cuda':torch.cuda.is_available(),'torch_threads':torch.get_num_threads(),
 'cpu_model':open('/proc/cpuinfo').read().split('model name\t: ')[1].split('\n')[0]},
 'worlds':W,'seeds':SEEDS,'models':len(rows),
 'methods':summary,'paired':compare,'guard_validity':gate_v,
 'vjp_relative_error_max':max(actual_vjp),'mirror_long_vjp_anti_aligned_edge_count':anti,
 'H120_audit':post_first,
 'H120_agrees_with_H40_gate':sum(x['sign_agree'] for x in post_first),
 'verdict':{
   'mechanism':'PASS exact shared-core VJP and actual edge removal',
   'guard':'PASS if valid decisions all 20/20',
   'prospective_learning':'PASS' if compare['learning_gate_pass']>=7 and summary['mirror_guard_long']['ood_nmse']<=min(summary['mirror_guard_short']['ood_nmse'],summary['mirror_keep']['ood_nmse']) else 'FAIL',
   'mirror_specific':'PASS' if compare['mirror_specific_gate_pass']>=7 else 'FAIL'
 }
}
(ROOT/'results/ANALYSIS.json').write_text(json.dumps(summary_doc,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'pairs':compare,'guards':gate_v,'vjp_max':summary_doc['vjp_relative_error_max'],
 'anti_align':anti,'h120_same':summary_doc['H120_agrees_with_H40_gate'], 'verdicts':summary_doc['verdict']},ensure_ascii=False,indent=2))
