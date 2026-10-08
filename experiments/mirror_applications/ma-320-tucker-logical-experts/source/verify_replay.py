import json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'));import run as m

def main():
 rows=[];events=[];bad=[];diffs=[]
 with tempfile.TemporaryDirectory() as td:
  for split,seeds in [('development',[32001,32002]),('fresh',[32011,32012,32013])]:
   for seed in seeds:
    old=json.loads((ROOT/'artifacts/results'/f'{"dev" if split=="development" else "fresh"}_{seed}.json').read_text())
    new=m.run(seed,split,Path(td)/'payloads',Path(td)/f'{seed}.json');newtab={x['method']:x for x in new['summary']};oldtab={x['method']:x for x in old['summary']}
    for method,o in oldtab.items():
     x=newtab[method];key=(split,seed,method)
     for field in ['serialized_bytes','fit_compute_proxy','active_ops_proxy_per_request','private_experts','shared_phase_experts','mean_test_n_mse','max_test_n_mse']:
      delta=abs(float(x[field])-float(o[field]));diffs.append(delta)
      if delta>1e-12:bad.append({'key':key,'field':field,'delta':delta})
     if x['payload_sha256']!=o['payload_sha256']:bad.append({'key':key,'payload_hash_mismatch':True})
     rows.append({'split':split,'seed':seed,'method':method,'payload_bytes':x['serialized_bytes'],'payload_sha256':x['payload_sha256'],'mean_n_mse':x['mean_test_n_mse'],'max_n_mse':x['max_test_n_mse'],'fit_compute_proxy':x['fit_compute_proxy']})
    oldev={(x['method'],x['expert_id']):x for x in old['allocation_events']};newev={(x['method'],x['expert_id']):x for x in new['allocation_events']}
    if oldev.keys()!=newev.keys():bad.append({'key':(split,seed),'allocation_event_keys_mismatch':True})
    for key,o in oldev.items():
     x=newev[key]
     if o['path']!=x['path'] or o['group']!=x['group'] or o['validation_n_mse']!=x['validation_n_mse']:bad.append({'key':(split,seed,*key),'allocation_event_mismatch':True})
    events.extend({'split':split,'seed':seed,**x} for x in old['allocation_events'])
 result={'summary_rows_replayed':len(rows),'allocation_events_replayed':len(events),'max_absolute_metric_difference':max(diffs,default=0),'payload_hashes_exact':not any('payload_hash_mismatch' in x for x in bad),'allocation_events_exact':not any('allocation_event' in str(x) for x in bad),'mismatches':bad}
 (ROOT/'REPLAY_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='mismatches'},indent=2))
 if bad:raise SystemExit(1)
if __name__=='__main__':main()
