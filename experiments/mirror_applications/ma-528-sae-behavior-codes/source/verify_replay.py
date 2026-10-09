#!/usr/bin/env python3
"""Replay MA-528 dev payloads and core causal metrics exactly."""
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'artifacts'/'development';MODEL=Path('/workspace/artifacts/pythia-70m-deduped-e93a9faa');SAE=Path('/workspace/artifacts/pythia-70m-sae-r4-layer3.pt');SEEDS=(52801,52802);SAE_SHA='85b57dfe34d19769e2df0e208e38fda8815da8493c56b703a99ff74cc610dbf4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def core(x):return [{k:r[k] for k in ('method','payload_bytes','model_bytes','sae_bytes','incremental_over_preloaded_sae_bytes','total_deployment_bytes','explicit_fv_total_bytes','global_omp16_total_bytes','support_examples','support_forward_calls','support_input_tokens','candidate_input_tokens','optimizer_updates','active_compute_proxy','metrics')} for r in x['rows']]
def main():
 assert sha(SAE)==SAE_SHA;hashes={}
 with tempfile.TemporaryDirectory(prefix='ma528_replay_') as td:
  for seed in SEEDS:
   src=RUNS/f'seed_{seed}';dst=Path(td)/f'seed_{seed}';env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='/tmp/ma516-pkgs')
   subprocess.run([sys.executable,str(ROOT/'source'/'run_behavior_codes.py'),'--seed',str(seed),'--model-dir',str(MODEL),'--sae',str(SAE),'--out',str(dst)],check=True,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   a=json.loads((src/'metrics.json').read_text());b=json.loads((dst/'metrics.json').read_text());assert core(a)==core(b),(seed,'metrics')
   for fname in ('explicit_fv.npz','global_omp16.npz','aggregate_pool64_omp16.npz','residual_pool64_omp16.npz'):
    pa=src/fname;pb=dst/fname;assert sha(pa)==sha(pb),(seed,fname);assert pa.stat().st_size==pb.stat().st_size
    with zipfile.ZipFile(pa) as z:assert all(i.compress_type==zipfile.ZIP_STORED for i in z.infolist())
    xa=np.load(pa,allow_pickle=False);xb=np.load(pb,allow_pickle=False);assert xa.files==xb.files
    for k in xa.files:assert np.array_equal(xa[k],xb[k]),(seed,fname,k)
    hashes[f'{seed}/{fname}']=sha(pa)
   assert sha(src/'split_manifest.json')==sha(dst/'split_manifest.json')
   xa=np.load(src/'pool_selection_audit.npz',allow_pickle=False);xb=np.load(dst/'pool_selection_audit.npz',allow_pickle=False)
   for k in xa.files:assert np.array_equal(xa[k],xb[k]),(seed,'pool',k)
  assert all(not (RUNS/f'seed_{s}').exists() for s in (52811,52812,52813))
 print(json.dumps({'experiment_id':'MA-528','development_seeds':list(SEEDS),'payload_sha256':hashes,'payloads_exact':len(hashes),'core_metrics_exact':True,'splits_and_pools_exact':True,'uncompressed_npz_exact':True,'sae_sha256_checked':True,'fresh_seeds_accessed':[],'fresh_dirs_absent':True,'max_core_metric_difference':0.0},indent=2))
if __name__=='__main__':main()
