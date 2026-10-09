import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts'
def rows():return [json.loads(x) for x in open(A/'fresh_runs.jsonl')]
def avg(r,m,n,k):return statistics.mean(x[k] for x in r if x['method']==m and x['n']==n)
def test_fresh_payload_hashes_and_partition():
 r=rows();assert len(r)==108;assert {x['world'] for x in r}=={45310,45311,45312}
 for x in r:
  b=(ROOT.parents[2]/x['path']).read_bytes();assert len(b)==x['payload_bytes'];assert hashlib.sha256(b).hexdigest()==x['hash']
def test_router_pareto_and_private_equivalence():
 r=rows();assert avg(r,'mirror',20,'nrmse_mean')<.1*avg(r,'router8',20,'nrmse_mean')
 assert avg(r,'mirror',20,'bytes_per_task')<avg(r,'router8',20,'bytes_per_task')
 assert avg(r,'mirror',20,'nrmse_mean')==avg(r,'independent',20,'nrmse_mean')
 assert avg(r,'mirror',20,'bytes_per_task')==avg(r,'independent',20,'bytes_per_task')
