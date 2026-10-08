import csv, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'

def test_fresh_partition_and_payloads():
 rows=[json.loads(x) for x in open(ART/'fresh_runs.jsonl')]
 assert len(rows)==2592
 assert {r['world'] for r in rows}=={44530,44531,44532}
 assert len({(r['world'],r['seed'],r['pair'],r['replicate']) for r in rows if r['method']=='mirror' and r['steps']==0})==162
 for r in rows:
  p=ROOT.parents[2]/r['path']; b=p.read_bytes()
  assert len(b)==r['payload_bytes']
  assert hashlib.sha256(b).hexdigest()==r['hash']

def test_a3_bytes_and_world_aggregates():
 with open(ART/'serialized_accounting.csv',newline='') as f: rows=list(csv.DictReader(f))
 assert len(rows)==486
 means={}
 for method in ('taskvec','mirror','leo'):
  vals=[int(r['amortized_bytes']) for r in rows if r['method']==method and int(r['N'])==20]
  means[method]=sum(vals)/len(vals)
 assert means['mirror']==28645
 assert means['taskvec']==28649
 assert means['leo']>means['mirror']
