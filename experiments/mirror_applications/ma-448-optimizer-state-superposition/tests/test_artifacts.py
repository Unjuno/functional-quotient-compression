import csv,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts'

def rows():
 with open(A/'fresh_results.csv',newline='') as f:return list(csv.DictReader(f))

def test_valid_fresh_payloads():
 r=rows();assert len(r)==12;assert {x['world'] for x in r}=={'44830','44831','44832'}
 for x in r:
  b=(ROOT.parents[2]/x['path']).read_bytes();assert len(b)==int(x['payload_bytes']);assert hashlib.sha256(b).hexdigest()==x['sha256']

def test_quality_and_pca_mirror_gates():
 r=rows();mean=lambda method,key:statistics.mean(float(x[key]) for x in r if x['method']==method)
 assert mean('mirror','continuation_nrmse_mean')>1.1*mean('exact','continuation_nrmse_mean')
 assert mean('mirror','continuation_nrmse_mean')>mean('pca','continuation_nrmse_mean')
 assert mean('mirror','bytes_per_task')==mean('pca','bytes_per_task')
 assert mean('shared','continuation_nrmse_mean')<mean('mirror','continuation_nrmse_mean')
