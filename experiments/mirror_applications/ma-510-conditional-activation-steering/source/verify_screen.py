#!/usr/bin/env python3
"""Replay dev-only MA-510 metrics and payloads; never opens fresh seeds."""
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'runs'; SEEDS=(51001,51002)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    files=0
    with tempfile.TemporaryDirectory(prefix='ma510_replay_') as td:
        tmp=Path(td)
        for seed in SEEDS:
            original=RUN/f'dev_{seed}'; rerun=tmp/f'dev_{seed}'
            env=os.environ.copy()
            env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([sys.executable,str(ROOT/'source'/'run_screen.py'),'--seed',str(seed),'--out',str(rerun)],check=True,env=env,stdout=subprocess.DEVNULL)
            a=json.loads((original/'metrics.json').read_text()); b=json.loads((rerun/'metrics.json').read_text())
            assert len(a['rows'])==len(b['rows'])==26
            for ra,rb in zip(a['rows'],b['rows']):
                assert ra['key']==rb['key'] and ra['bytes']==rb['bytes'] and ra['ops_per_example']==rb['ops_per_example']
                assert ra['metrics']==rb['metrics'] and ra['native_alias']==rb['native_alias']
                if ra['mode']=='native_pca': assert ra['native_alias']
                pa=original/f"{ra['key']}.npz"; pb=rerun/f"{rb['key']}.npz"
                assert pa.stat().st_size==pb.stat().st_size==ra['bytes'] and digest(pa)==digest(pb)
                files+=1
        assert not (RUN/'dev_51011').exists() and not (RUN/'dev_51012').exists() and not (RUN/'dev_51013').exists()
    print(json.dumps({'seeds':list(SEEDS),'payloads_byte_hash_identical':files,'max_metric_replay_difference':0.0,'fresh_directories_absent':True},indent=2))
if __name__=='__main__': main()
