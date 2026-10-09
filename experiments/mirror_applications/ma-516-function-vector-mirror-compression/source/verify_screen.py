#!/usr/bin/env python3
"""Replay MA-516 dev runs against pinned local Pythia artifact; never opens fresh seeds."""
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
MODEL=Path('/workspace/artifacts/pythia-70m-deduped-e93a9faa')
RUNS=ROOT/'runs';SEEDS=(51601,51602)
MODEL_SHA='3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert sha(MODEL/'model.safetensors')==MODEL_SHA
 payload_count=0;maxdiff=0.
 with tempfile.TemporaryDirectory(prefix='ma516_replay_') as td:
  for seed in SEEDS:
   src=RUNS/f'dev_{seed}';dst=Path(td)/f'dev_{seed}';env=os.environ.copy()
   env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='/tmp/ma516-pkgs')
   subprocess.run([sys.executable,str(ROOT/'source'/'run_screen.py'),'--seed',str(seed),'--model-dir',str(MODEL),'--out',str(dst)],check=True,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   a=json.loads((src/'metrics.json').read_text());b=json.loads((dst/'metrics.json').read_text())
   assert len(a['rows'])==len(b['rows'])==12
   for x,y in zip(a['rows'],b['rows']):
    for k in ('method','rank','payload_bytes','total_deployment_bytes','common_base_bytes','extra_compute_proxy','support_examples','support_forward_calls','support_input_tokens','evaluation_queries','metrics','native_alias'):
     assert x[k]==y[k],(seed,x['method'],k,x[k],y[k])
    if x['method'].startswith('native_pca_'): assert x['native_alias']
    if x['payload_bytes']:
     pa=src/(x['method']+'.npz');pb=dst/(y['method']+'.npz')
     assert pa.stat().st_size==pb.stat().st_size==x['payload_bytes'] and sha(pa)==sha(pb)
     with zipfile.ZipFile(pa) as z:
      assert all(i.compress_type==zipfile.ZIP_STORED for i in z.infolist())
     with np.load(pa) as arrays:
      assert len(arrays.files)>0
     payload_count+=1
   for name in ('split_manifest.json','extracted_fvs.npz'):
    assert sha(src/name)==sha(dst/name)
  assert all(not (RUNS/f'dev_{s}').exists() for s in (51611,51612,51613))
 print(json.dumps({'development_seeds':list(SEEDS),'payloads_byte_hash_identical':payload_count,
                   'metrics_exact':True,'task_split_and_extracted_vectors_exact':True,
                   'uncompressed_npz_checked':True,'fresh_directories_absent':True,'max_metric_difference':maxdiff},indent=2))
if __name__=='__main__':main()
