#!/usr/bin/env python3
"""Replay MA-517 development artifacts from pinned Pythia and MA-516 extractor."""
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODEL=Path('/workspace/artifacts/pythia-70m-deduped-e93a9faa')
RUNS=ROOT/'runs';SEEDS=(51701,51702)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 count=0
 with tempfile.TemporaryDirectory(prefix='ma517_replay_') as td:
  for seed in SEEDS:
   src=RUNS/f'dev_{seed}';dst=Path(td)/f'dev_{seed}';env=os.environ.copy()
   env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='/tmp/ma516-pkgs')
   subprocess.run([sys.executable,str(ROOT/'source'/'run_screen.py'),'--seed',str(seed),'--model-dir',str(MODEL),'--out',str(dst)],check=True,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   a=json.loads((src/'metrics.json').read_text());b=json.loads((dst/'metrics.json').read_text())
   assert len(a['rows'])==len(b['rows'])==14
   for x,y in zip(a['rows'],b['rows']):
    for k in ('method','rank','payload_bytes','total_deployment_bytes','common_base_bytes','extra_compute_proxy','support_examples','support_forward_calls','support_input_tokens','metrics','composed_vector_norms','native_alias'):
     assert x[k]==y[k],(seed,x['method'],k)
    if x['method'].startswith('native_code_product_'):assert x['native_alias']
    if x['payload_bytes']:
     pa=src/(x['method']+'.npz');pb=dst/(y['method']+'.npz')
     assert pa.stat().st_size==pb.stat().st_size==x['payload_bytes'] and sha(pa)==sha(pb)
     with zipfile.ZipFile(pa) as z:assert all(i.compress_type==zipfile.ZIP_STORED for i in z.infolist())
     count+=1
   for name in ('split_manifest.json','operand_fvs.npz'):assert sha(src/name)==sha(dst/name)
  assert all(not (RUNS/f'dev_{s}').exists() for s in (51711,51712,51713))
 print(json.dumps({'dev_seeds':list(SEEDS),'payloads_byte_hash_identical':count,'metrics_exact':True,
                   'paired_split_and_vectors_exact':True,'native_code_product_alias_all_ranks':True,
                   'uncompressed_npz_checked':True,'fresh_directories_absent':True,'max_metric_difference':0.0},indent=2))
if __name__=='__main__':main()
