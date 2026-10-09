import csv,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import ROOT,OUT,world,evaluate
rows=list(csv.DictReader((ROOT/'RESULTS_CORE.csv').open()));md=0.
for r in rows:
 p=OUT/'payloads'/r['condition']/r['world_or_seed']/r['learning_rate']/(r['method']+'.pt');b=p.read_bytes();assert len(b)==int(r['serialized_bytes']) and hashlib.sha256(b).hexdigest()==r['payload_sha256']
 w=world(int(r['world_or_seed']));part='va' if r['condition']=='development' else 'te';a,n,_=evaluate(b,w,part);md=max(md,abs(a-float(r['accuracy'])),abs(n-float(r['nll'])))
assert md<1e-7
print(json.dumps({'rows_replayed':len(rows),'payload_hashes_checked':True,'max_metric_difference':md,'max_roundtrip_diff':max(float(r['roundtrip_max_abs_diff']) for r in rows)},sort_keys=True))
