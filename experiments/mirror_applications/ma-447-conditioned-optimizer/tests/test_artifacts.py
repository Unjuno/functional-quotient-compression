import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts'

def test_fresh_payload_replay():
 rows=[json.loads(x) for x in open(A/'fresh_runs.jsonl')]
 assert len(rows)==288
 assert {r['world'] for r in rows}=={44710,44711,44712}
 for r in rows:
  b=(ROOT.parents[2]/r['path']).read_bytes()
  assert len(b)==r['payload_bytes'] and hashlib.sha256(b).hexdigest()==r['hash']

def test_frontier_fails_conditioned_gates():
 with open(A/'byte_frontier.csv',newline='') as f:rows=list(csv.DictReader(f))
 assert len(rows)==108
 means={m:sum(float(r['bytes_per_task']) for r in rows if r['method']==m and r['N']=='20')/9 for m in ('adam','conditioned','separate')}
 assert means['conditioned']>means['separate']>means['adam']
 fresh=[json.loads(x) for x in open(A/'fresh_runs.jsonl') if json.loads(x)['step']==4]
 assert sum(r['nrmse_mean'] for r in fresh if r['method']=='conditioned')/len([r for r in fresh if r['method']=='conditioned']) > sum(r['nrmse_mean'] for r in fresh if r['method']=='adam')/len([r for r in fresh if r['method']=='adam'])
