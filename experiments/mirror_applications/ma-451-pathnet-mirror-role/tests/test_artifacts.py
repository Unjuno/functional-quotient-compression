import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts'
def rows():return [json.loads(x) for x in open(ART/'fresh_runs.jsonl')]
def mean(r,m,n,k):return statistics.mean(x[k] for x in r if x['method']==m and x['n']==n)
def test_fresh_payloads_and_role_recovery():
 r=rows();assert len(r)==324;assert {x['world'] for x in r}=={45110,45111,45112}
 for x in r:
  b=(ROOT.parents[2]/x['path']).read_bytes();assert len(b)==x['payload_bytes'];assert hashlib.sha256(b).hexdigest()==x['hash']
  if x['method']=='mirror':assert json.loads(x['teacher_angles'])==json.loads(x['inferred_angles'])
def test_quality_and_storage_crossover():
 r=rows()
 for w in (45110,45111,45112):
  pm=[x for x in r if x['world']==w and x['method']=='mirror' and x['n']==20]
  pn=[x for x in r if x['world']==w and x['method']=='pathnet' and x['n']==20]
  assert statistics.mean(x['nrmse_mean'] for x in pm)+.01<statistics.mean(x['nrmse_mean'] for x in pn)
 assert mean(r,'mirror',20,'bytes_per_task')>mean(r,'independent',20,'bytes_per_task')
 assert mean(r,'mirror',64,'bytes_per_task')<mean(r,'independent',64,'bytes_per_task')
