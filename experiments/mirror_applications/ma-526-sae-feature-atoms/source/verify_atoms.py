#!/usr/bin/env python3
"""Replay MA-526 dev worlds and audit model/SAE/payload integrity."""
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'runs';MODEL=Path('/workspace/artifacts/pythia-70m-deduped-e93a9faa');SAE=Path('/workspace/artifacts/pythia-70m-sae-r4-layer3.pt');SEEDS=(52601,52602)
SAE_SHA='85b57dfe34d19769e2df0e208e38fda8815da8493c56b703a99ff74cc610dbf4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert sha(SAE)==SAE_SHA;count=0
 with tempfile.TemporaryDirectory(prefix='ma526_replay_') as td:
  for seed in SEEDS:
   src=RUNS/f'dev_{seed}';dst=Path(td)/f'dev_{seed}';env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='/tmp/ma516-pkgs')
   subprocess.run([sys.executable,str(ROOT/'source'/'run_atoms.py'),'--seed',str(seed),'--model-dir',str(MODEL),'--sae',str(SAE),'--out',str(dst)],check=True,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   a=json.loads((src/'metrics.json').read_text());b=json.loads((dst/'metrics.json').read_text());assert [r['method'] for r in a['rows']]==[r['method'] for r in b['rows']]
   for x,y in zip(a['rows'],b['rows']):
    for k in ('method','payload_bytes','model_bytes','sae_bytes','common_base_bytes','total_deployment_bytes','incremental_over_preloaded_sae_bytes','explicit_fv_total_bytes','support_examples','support_forward_calls','support_input_tokens','candidate_input_tokens','optimizer_updates','active_compute_proxy','metrics','vector_metrics','nonzero_coefficients_per_task'):
     assert x[k]==y[k],(seed,x['method'],k)
    if x['payload_bytes']:
     pa=src/(x['method']+'.npz');pb=dst/(y['method']+'.npz')
     assert pa.stat().st_size==pb.stat().st_size==x['payload_bytes'] and sha(pa)==sha(pb)
     with zipfile.ZipFile(pa) as z:assert all(i.compress_type==zipfile.ZIP_STORED for i in z.infolist())
     count+=1
   for f in ('split_manifest.json',):assert sha(src/f)==sha(dst/f)
   for f in ('sae_dictionary_audit.npz',):
    xa=np.load(src/f);xb=np.load(dst/f);assert xa.files==xb.files
    for k in xa.files:assert np.array_equal(xa[k],xb[k]),(seed,f,k)
  assert all(not (RUNS/f'dev_{s}').exists() for s in (52611,52612,52613))
 print(json.dumps({'dev_seeds':list(SEEDS),'payload_hashes_identical':count,'metrics_exact':True,'task_splits_and_pool_selection_exact':True,'teacher_task_metrics_replayed_exact':True,'SAE_sha256_checked':True,'uncompressed_npz_checked':True,'fresh_directories_absent':True,'preflight_loader_failure_preserved':(RUNS/'preflight_load_failure_TiedSAE_class/failed_launch.log').exists(),'max_metric_difference':0.0},indent=2))
if __name__=='__main__':main()
