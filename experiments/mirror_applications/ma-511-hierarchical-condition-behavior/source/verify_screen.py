#!/usr/bin/env python3
"""Deterministically replay MA-511 development artifacts; does not open fresh seeds."""
import hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs';SEEDS=(51101,51102)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 count=0
 with tempfile.TemporaryDirectory(prefix='ma511_replay_') as td:
  for seed in SEEDS:
   src=RUN/f'dev_{seed}';dst=Path(td)/f'dev_{seed}';env=os.environ.copy()
   env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
   subprocess.run([sys.executable,str(ROOT/'source'/'run_screen.py'),'--seed',str(seed),'--out',str(dst)],check=True,env=env,stdout=subprocess.DEVNULL)
   a=json.loads((src/'metrics.json').read_text());b=json.loads((dst/'metrics.json').read_text())
   assert len(a['rows'])==len(b['rows'])==27
   for x,y in zip(a['rows'],b['rows']):
    assert (x['key'],x['bytes'],x['ops_per_pair'],x['metrics'],x['native_alias'])==(y['key'],y['bytes'],y['ops_per_pair'],y['metrics'],y['native_alias'])
    pa=src/(x['key']+'.npz');pb=dst/(y['key']+'.npz')
    assert pa.stat().st_size==pb.stat().st_size==x['bytes'] and sha(pa)==sha(pb)
    if x['method']=='native_additive': assert x['native_alias']
    count+=1
   for name in ('visible_pairs','heldout_pairs'):
    with __import__('numpy').load(src/'split_manifest.npz') as x,__import__('numpy').load(dst/'split_manifest.npz') as y:
     assert __import__('numpy').array_equal(x[name],y[name])
  assert not any((RUN/f'dev_{s}').exists() for s in (51111,51112,51113))
 print(json.dumps({'development_seeds':list(SEEDS),'payloads_byte_hash_identical':count,'metrics_exact':True,'split_exact':True,'fresh_directories_absent':True},indent=2))
if __name__=='__main__':main()
