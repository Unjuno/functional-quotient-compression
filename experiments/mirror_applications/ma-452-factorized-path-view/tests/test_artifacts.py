import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts'
def rows():return [json.loads(x) for x in open(A/'fresh_runs.jsonl')]
def mean(r,m,n,k):return statistics.mean(x[k] for x in r if x['method']==m and x['n']==n)
def test_fresh_payloads_and_heldout_factor_combinations():
 r=rows();assert len(r)==324;assert {x['world'] for x in r}=={45210,45211,45212}
 for x in r:
  b=(ROOT.parents[2]/x['path']).read_bytes();assert len(b)==x['payload_bytes'];assert hashlib.sha256(b).hexdigest()==x['hash']
  if x['method']=='mirror':assert json.loads(x['teacher_angles'])==json.loads(x['inferred_angles'])
def test_routing_networks_matches_mirror_with_lower_bytes():
 r=rows();
 assert mean(r,'mirror',20,'nrmse_mean')<=1.1*mean(r,'routing',20,'nrmse_mean')+1e-7
 assert mean(r,'mirror',20,'payload_bytes')>mean(r,'routing',20,'payload_bytes')
 assert mean(r,'mirror',64,'bytes_per_task')>mean(r,'routing',64,'bytes_per_task')
 assert mean(r,'mirror',20,'bytes_per_task')<mean(r,'flat',20,'bytes_per_task')
