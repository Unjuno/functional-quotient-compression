#!/usr/bin/env python3
"""Deterministic replay of MA-521 development screens and stored payloads."""
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'runs';MODEL=Path('/workspace/artifacts/pythia-70m-deduped-e93a9faa');SEEDS=(52101,52102)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 count=0
 with tempfile.TemporaryDirectory(prefix='ma521_replay_') as td:
  for seed in SEEDS:
   src=RUNS/f'dev_{seed}';dst=Path(td)/f'dev_{seed}';env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='/tmp/ma516-pkgs')
   subprocess.run([sys.executable,str(ROOT/'source'/'run_compiler.py'),'--seed',str(seed),'--model-dir',str(MODEL),'--out',str(dst)],check=True,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   a=json.loads((src/'metrics.json').read_text());b=json.loads((dst/'metrics.json').read_text());assert [r['method'] for r in a['rows']]==[r['method'] for r in b['rows']]
   for x,y in zip(a['rows'],b['rows']):
    for k in ('method','rank','payload_bytes','common_base_bytes','total_deployment_bytes','support_examples','support_forward_calls','support_input_tokens','demonstration_feature_tokens','candidate_input_tokens','optimizer_updates','active_compute_proxy','metrics','vector_metrics','query_context_tokens_per_query'):
     assert x[k]==y[k],(seed,x['method'],k)
    if x['payload_bytes']:
     pa=src/(x['method']+'.npz');pb=dst/(y['method']+'.npz')
     assert pa.stat().st_size==pb.stat().st_size==x['payload_bytes'] and sha(pa)==sha(pb)
     with zipfile.ZipFile(pa) as z:assert all(i.compress_type==zipfile.ZIP_STORED for i in z.infolist())
     count+=1
   for f in ('split_manifest.json',):assert sha(src/f)==sha(dst/f)
   for f in ('teacher_vectors.npz','demo_features.npz'):
    za=np.load(src/f);zb=np.load(dst/f)
    assert za.files==zb.files
    for k in za.files:assert np.array_equal(za[k],zb[k]),(seed,f,k)
  assert all(not (RUNS/f'dev_{s}').exists() for s in (52111,52112,52113))
 print(json.dumps({'dev_seeds':list(SEEDS),'payload_hashes_identical':count,'metrics_exact':True,'splits_teacher_vectors_and_demo_features_exact':True,'uncompressed_npz_checked':True,'fresh_directories_absent':True,'max_metric_difference':0.0},indent=2))
if __name__=='__main__':main()
