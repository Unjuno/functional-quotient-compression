import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from run import run

def read(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))
def main():
 rows=read(ROOT/'RESULTS_CORE.csv');events=read(ROOT/'ALLOCATION_EVENTS.csv');files=sorted((ROOT/'source/artifacts').glob('315*.json'));expected=[];expected_e=[];bad=[];diffs=[];tmp=ROOT/'source/artifacts/replay_payloads'
 for f in files:
  old=json.loads(f.read_text());seed=int(f.stem);split='fresh' if seed>=31511 else 'development';actual,ae=run(seed,split,.0001,tmp);expected+=old['summaries'];expected_e+=old['events'];oldtab={(x['condition'],int(x['world_or_seed']),x['method']):x for x in old['summaries']}
  for x in actual:
   key=(x['condition'],int(x['world_or_seed']),x['method']);o=oldtab.get(key)
   if o is None:bad.append({'missing_artifact_summary':key});continue
   for k in ('serialized_bytes','fit_compute_proxy','active_ops_proxy_per_example','primary_value'):
    d=abs(float(x[k])-float(o[k]));diffs.append(d)
    if d>(1e-12 if k=='primary_value' else 0):bad.append({'replay_metric':key,'field':k,'delta':d})
   if x['payload_sha256']!=o['payload_sha256']:bad.append({'replay_hash':key})
 table={(x['condition'],int(x['world_or_seed']),x['method']):x for x in rows}
 for x in expected:
  key=(x['condition'],int(x['world_or_seed']),x['method']);r=table.get(key)
  if r is None:bad.append({'missing_csv_summary':key});continue
  for k in ('serialized_bytes','fit_compute_proxy','active_ops_proxy_per_example','primary_value'):
   d=abs(float(r[k])-float(x[k]));diffs.append(d)
   if d>(1e-12 if k=='primary_value' else 0):bad.append({'csv_metric':key,'field':k,'delta':d})
  if r['payload_sha256']!=x['payload_sha256']:bad.append({'csv_hash':key})
 et={(x['condition'],int(x['world']),x['method'],int(x['task_index'])):x for x in events}
 for x in expected_e:
  key=(x['condition'],int(x['world']),x['method'],int(x['task_index']));e=et.get(key)
  if e is None:bad.append({'missing_event':key});continue
  for k in ('group','choice','private_count','private_fallback'):
   if str(e[k])!=str(x[k]):bad.append({'event_field':key,'field':k})
  d=abs(float(e['validation_n_mse'])-float(x['validation_n_mse']));diffs.append(d)
  if d>1e-12:bad.append({'event_metric':key,'delta':d})
 result={'summary_rows_replayed':len(expected),'allocation_events_replayed':len(expected_e),'artifact_files':len(files),'max_absolute_difference':max(diffs,default=0),'payload_hashes_exact':not bad,'mismatches':bad,'fresh_opened':(ROOT/'FRESH_RESULTS.csv').exists(),'wall_clock_excluded':'Timing and throughput are diagnostics.'}
 (ROOT/'source/REPLAY_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 if bad:raise SystemExit(1)
if __name__=='__main__':main()
