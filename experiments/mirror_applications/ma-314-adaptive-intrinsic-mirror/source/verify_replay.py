import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def load_csv(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))

def main():
 rows=load_csv(ROOT/'RESULTS_CORE.csv');events=load_csv(ROOT/'ALLOCATION_EVENTS.csv');artifacts=sorted((ROOT/'source/artifacts').glob('314*_t*.json'));expected=[];expected_events=[]
 for f in artifacts:
  d=json.loads(f.read_text());expected.extend(d['summaries']);expected_events.extend(d['events'])
 table={(r['condition'],int(r['world_or_seed']),r['method']):r for r in rows};diffs=[];bad=[]
 for a in expected:
  key=(a['condition'],int(a['world_or_seed']),a['method']);r=table.get(key)
  if r is None:bad.append({'missing_summary':key});continue
  for field in ('serialized_bytes','fit_compute_proxy','active_ops_proxy_per_example','primary_value','max_payload_dimension'):
   delta=abs(float(r[field])-float(a[field]));diffs.append(delta)
   if delta>(1e-12 if field=='primary_value' else 0):bad.append({'field':field,'key':key,'delta':delta})
  if r['payload_sha256']!=a['payload_sha256']:bad.append({'payload_hash':key})
 evtable={(e['condition'],int(e['world']),e['method'],int(e['task_index'])):e for e in events}
 for a in expected_events:
  key=(a['condition'],int(a['world']),a['method'],int(a['task_index']));e=evtable.get(key)
  if e is None:bad.append({'missing_event':key});continue
  for field in ('group','choice','dimension','private_fallback'):
   if str(e[field])!=str(a[field]):bad.append({'event_field':field,'key':key})
  delta=abs(float(e['validation_n_mse'])-float(a['validation_n_mse']));diffs.append(delta)
  if delta>1e-12:bad.append({'event_metric':key,'delta':delta})
  if int(e['final_payload_bytes'])!=int(a['final_payload_bytes']):bad.append({'event_bytes':key})
 result={'summary_rows_replayed':len(expected),'allocation_events_replayed':len(expected_events),'artifact_files':len(artifacts),'max_absolute_difference':max(diffs,default=0.0),'payload_hashes_exact':not bad,'mismatches':bad,'fresh_opened':(ROOT/'FRESH_RESULTS.csv').exists()}
 (ROOT/'source/REPLAY_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 if bad:raise SystemExit(1)
if __name__=='__main__':main()
