"""Resume independent complete-fit jobs without changing the locked science core."""
import argparse,hashlib,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from gic import CFG,ROOT,make_world,train,save_model

def one_case(job):
    world,init,dirname=job; out=Path(dirname);out.mkdir(parents=True,exist_ok=True);count=0
    for regime in CFG['regimes']:
        w=make_world(world,regime)
        for method in CFG['methods']:
            stem=f'{regime}_{world}_{init}_{method}';meta=out/(stem+'.json');payload=out/(stem+'.bin')
            if meta.exists():
                row=json.loads(meta.read_text())
                assert hashlib.sha256(payload.read_bytes()).hexdigest()==row['payload_sha256']
                continue
            model,row=train(method,w,world,init);row['regime']=regime
            save_model(model,payload);row['payload_sha256']=hashlib.sha256(payload.read_bytes()).hexdigest();row['payload_name']=payload.name
            meta.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');count+=1
        print(f'world={world} init={init} regime={regime} complete',flush=True)
    return {'world':world,'init':init,'new_completed':count}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worlds',required=True);p.add_argument('--inits',default='0,1,2');p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();lock=json.loads((ROOT/'PRE_RUN_LOCK.json').read_text())
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in lock['files'].items())
    jobs=[(int(w),int(i),str(ROOT/'evidence'/'fresh')) for w in a.worlds.split(',') for i in a.inits.split(',')]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for ans in pool.map(one_case,jobs):print(json.dumps(ans),flush=True)
