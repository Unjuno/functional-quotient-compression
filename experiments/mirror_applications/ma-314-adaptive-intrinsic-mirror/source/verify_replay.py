import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from run import run

def load_csv(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))

def main():
 rows=load_csv(ROOT/'RESULTS_CORE.csv');events=load_csv(ROOT/'ALLOCATION_EVENTS.csv');artifacts=sorted((ROOT/'source/artifacts').glob('314*_t*.json'));expected=[];expected_events=[];diffs=[];bad=[]
 replay_dir=ROOT/'source/artifacts/replay_payloads'
 for f in artifacts:
  stored=json.loads(f.read_text());seed=int(f.stem.split('_')[0]);thr=float(f.stem.split('_t')[1]);split='fresh' if seed>=31411 else 'development'
  actual,actual_events=run(seed,split,thr,replay_dir);expected.extend(stored['summaries']);expected_events.extend(stored['events'])
  table={(x['condition'],int(x['world_or_seed']),x['method']):x for x in stored['summaries']}
  for x in actual:
   key=(x['condition'],int(x['world_or_seed']),x['method']);old=table.get(key)
   if old is None:bad.append({'missing_artifact_summary':key});continue
   for field in ('serialized_bytes','fit_compute_proxy','active_ops_proxy_per_example','primary_value','max_payload_dimension'):
    delta=abs(float(x[field])-float(old[field]));diffs.append(delta)
    if delta>(1e-12 if field=='primary_value' else 0):bad.append({'replay_field':field,'key':key,'delta':delta})
   if x['payload_sha256']!=old['payload_sha256']:bad.append({'replay_payload_hash':key})
  for x in actual_events:
   key=(x['condition'],int(x['world']),x['method'],int(x['task_index']));old={(e['condition'],int(e['world']),e['method'],int(e['task_index'])):e for e in stored['events']}.get(key)
   if old is None:bad.append({'missing_artifact_event':key});continue
   for field in ('group','choice','dimension','private_fallback'):
    if x[field]!=old[field]:bad.append({'replay_event_field':field,'key':key})
   delta=abs(float(x['validation_n_mse'])-float(old['validation_n_mse']));diffs.append(delta)
   if delta>1e-12:bad.append({'replay_event_metric':key,'delta':delta})
 table={(r['condition'],int(r['world_or_seed']),r['method']):r for r in rows}
 for x in expected:
  key=(x['condition'],int(x['world_or_seed']),x['method']);r=table.get(key)
  if r is None:bad.append({'missing_csv_summary':key});continue
  for field in ('serialized_bytes','fit_compute_proxy','active_ops_proxy_per_example','primary_value','max_payload_dimension'):
   delta=abs(float(r[field])-float(x[field]));diffs.append(delta)
   if delta>(1e-12 if field=='primary_value' else 0):bad.append({'csv_field':field,'key':key,'delta':delta})
  if r['payload_sha256']!=x['payload_sha256']:bad.append({'csv_payload_hash':key})
 evtable={(e['condition'],int(e['world']),e['method'],int(e['task_index'])):e for e in events}
 for x in expected_events:
  key=(x['condition'],int(x['world']),x['method'],int(x['task_index']));e=evtable.get(key)
  if e is None:bad.append({'missing_csv_event':key});continue
  for field in ('group','choice','dimension','private_fallback'):
   if str(e[field])!=str(x[field]):bad.append({'csv_event_field':field,'key':key})
  delta=abs(float(e['validation_n_mse'])-float(x['validation_n_mse']));diffs.append(delta)
  if delta>1e-12:bad.append({'csv_event_metric':key,'delta':delta})
  if int(e['final_payload_bytes'])!=int(x['final_payload_bytes']):bad.append({'csv_event_bytes':key})
 result={'summary_rows_replayed':len(expected),'allocation_events_replayed':len(expected_events),'artifact_files':len(artifacts),'max_absolute_difference':max(diffs,default=0.0),'payload_hashes_exact':not bad,'mismatches':bad,'fresh_opened':(ROOT/'FRESH_RESULTS.csv').exists(),'wall_clock_excluded':'Wall-clock and throughput are nondeterministic diagnostics.'}
 (ROOT/'source/REPLAY_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 if bad:raise SystemExit(1)
if __name__=='__main__':main()
