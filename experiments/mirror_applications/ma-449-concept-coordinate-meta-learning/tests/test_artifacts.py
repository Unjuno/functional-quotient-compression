import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts'

def rows():return [json.loads(x) for x in open(A/'fresh_runs.jsonl')]
def test_fresh_payloads_and_factor_interventions():
 r=rows();assert len(r)==36;assert {x['world'] for x in r}=={44910,44911,44912}
 for x in r:
  b=(ROOT.parents[2]/x['path']).read_bytes();assert len(b)==x['payload_bytes'];assert hashlib.sha256(b).hexdigest()==x['hash']
 i=json.loads(r[0]['intervention_coordinates']);assert abs(i[0][0]-1)<1e-4 and abs(i[0][1])<1e-4;assert abs(i[1][1]-1)<1e-4 and abs(i[1][0])<1e-4

def test_registered_mirror_specific_byte_gate_fails():
 r=rows();means={m:statistics.mean(x['bytes_per_task'] for x in r if x['method']==m) for m in ('taskvec','leo','mirror','full')}
 assert means['mirror']==means['taskvec']==means['leo']
 assert statistics.mean(x['nrmse_mean'] for x in r if x['method']=='mirror') < 1e-5
 assert statistics.mean(x['nrmse_mean'] for x in r if x['method']=='leo') > .3
