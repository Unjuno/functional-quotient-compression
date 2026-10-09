#!/usr/bin/env python3
"""Deterministically replay MA-527 development payloads and metrics."""
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; RUNS=ROOT/'artifacts'/'development'; MODEL=Path('/workspace/artifacts/pythia-70m-deduped-e93a9faa'); SAE=Path('/workspace/artifacts/pythia-70m-sae-r4-layer3.pt')
SEEDS=(52701,52702); SAE_SHA='85b57dfe34d19769e2df0e208e38fda8815da8493c56b703a99ff74cc610dbf4'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def core(x):
 return [{k:r[k] for k in ('method','payload_bytes','model_bytes','sae_bytes','incremental_over_preloaded_sae_bytes','total_deployment_bytes','explicit_fv_total_bytes','support_examples','support_forward_calls','support_input_tokens','candidate_input_tokens','optimizer_updates','active_compute_proxy','metrics')} for r in x['rows']]
def main():
 assert sha(SAE)==SAE_SHA
 count=0; hashes={}
 with tempfile.TemporaryDirectory(prefix='ma527_replay_') as td:
  for seed in SEEDS:
   src=RUNS/f'seed_{seed}'; dst=Path(td)/f'seed_{seed}'; env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='/tmp/ma516-pkgs')
   subprocess.run([sys.executable,str(ROOT/'source'/'run_givens.py'),'--seed',str(seed),'--model-dir',str(MODEL),'--sae',str(SAE),'--out',str(dst)],check=True,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   a=json.loads((src/'metrics.json').read_text());b=json.loads((dst/'metrics.json').read_text());assert core(a)==core(b),(seed,'metrics')
   for fname in ('explicit_fv.npz','shared_givens.npz','native_gains.npz','shared_code.npz','global_omp16.npz'):
    pa=src/fname;pb=dst/fname;assert sha(pa)==sha(pb),(seed,fname);assert pa.stat().st_size==pb.stat().st_size
    with zipfile.ZipFile(pa) as z:assert all(i.compress_type==zipfile.ZIP_STORED for i in z.infolist())
    # Load every paid array and require exact byte-level array roundtrip.
    xa=np.load(pa,allow_pickle=False);xb=np.load(pb,allow_pickle=False);assert xa.files==xb.files
    for k in xa.files: assert np.array_equal(xa[k],xb[k]),(seed,fname,k)
    hashes[f'{seed}/{fname}']=sha(pa);count+=1
   assert sha(src/'split_manifest.json')==sha(dst/'split_manifest.json')
   for name in ('pool_audit.npz',):
    xa=np.load(src/name,allow_pickle=False);xb=np.load(dst/name,allow_pickle=False)
    assert all(np.array_equal(xa[k],xb[k]) for k in xa.files)
  assert all(not (RUNS/f'seed_{s}').exists() for s in (52711,52712,52713))
 print(json.dumps({'experiment_id':'MA-527','development_seeds':list(SEEDS),'payload_hashes_identical':count,'payload_sha256':hashes,'core_metrics_exact':True,'splits_and_pool_selection_exact':True,'serialization_uncompressed_npz_roundtrip_exact':True,'sae_sha256_checked':True,'fresh_seeds_accessed':[],'fresh_directories_absent':True,'max_metric_difference':0.0},indent=2))
if __name__=='__main__':main()
