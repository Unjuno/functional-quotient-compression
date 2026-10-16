"""Independent float64 NumPy/SciPy inference, payload auditing, fixed calibration."""
import argparse,hashlib,json,struct,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr,expit
import torch
from threadpoolctl import threadpool_limits,threadpool_info
from engine import ROOT,CFG,world,load,sha,fit,save

def decode(path):
    b=Path(path).read_bytes()
    if b[:8]!=b'GIC002v1':raise ValueError('magic')
    n=struct.unpack('<I',b[8:12])[0];meta=json.loads(b[12:12+n]);off=768;state={}
    for k,shape,size in meta['tensors']:
        state[k]=np.frombuffer(b[off:off+size],'<f4').reshape(shape).astype(np.float64);off+=size
    if off!=len(b):raise ValueError('size')
    return meta,state

def numpy_predict(path,x,c,reg=None):
    meta,s=decode(path);x=x.astype(float);c=c.astype(float)
    if meta['kind']=='independent':
        z=np.einsum('nd,edh->neh',x,s['w1'])+s['b1'];ex=(z*ndtr(z)*s['w2']).sum(-1)+s['b2']
    else:
        z=x@s['w1']+s['b1'];ex=[]
        for e in (0,1):
            if meta['kind']=='mirror':
                a=np.pad(s['codes'][e],(0,8-meta['q']));cs,sn=np.cos(a),np.sin(a);zz=z.reshape(-1,8,2)
                u,v=zz[...,0],zz[...,1];rz=np.stack((cs*u-sn*v,sn*u+cs*v),-1).reshape(-1,16)
                hz=rz*ndtr(rz);zz=hz.reshape(-1,8,2);u,v=zz[...,0],zz[...,1]
                hidden=np.stack((cs*u+sn*v,-sn*u+cs*v),-1).reshape(-1,16)
            else:hidden=z*ndtr(z)*np.exp(np.pad(s['codes'][e],(0,8-meta['q']))).repeat(2)
            ex.append(hidden@s['w2']+s['b2'])
        ex=np.column_stack(ex)
    p=expit(c@s['router']+s['rb']) if reg is None else reg
    return ex[:,0]*(1-p)+ex[:,1]*p

def verify_all():
    rows=[]
    for stage in ('CALIBRATION','FRESH'):rows.extend(json.loads((ROOT/f'{stage}_RECORDS.json').read_text()))
    datacache={};maximum=0.;payloads=0;bad=[]
    for row in rows:
        key=(row['world'],row['family']);datacache.setdefault(key,world(*key));d=datacache[key]
        for endpoint in ('','best_'):
            path=ROOT/row[endpoint+'file'];h=sha(path)
            if h!=row[endpoint+'sha256'] or path.stat().st_size!=row['bytes']:bad.append(str(path))
            for split in ('train','val','test'):
                n=row['ntrain'] if split=='train' else len(d[split+'_x'])
                reg=d[split+'_reg'][:n] if row['method'].endswith('_oracle') else None
                y=numpy_predict(path,d[split+'_x'][:n],d[split+'_c'][:n],reg)
                err=float(np.mean((y-d[split+'_clean'][:n])**2));recorded=row[endpoint+split+'_mse']
                maximum=max(maximum,abs(err-recorded))
                if not np.isclose(err,recorded,atol=5e-6,rtol=2e-5):bad.append(str(path)+'/'+split)
            payloads+=1
    locks=json.loads((ROOT/'PRE_RUN_LOCK.json').read_text())['files'];matched=all(sha(ROOT/p)==h for p,h in locks.items())
    report={'main_models':len(rows),'payloads':payloads,'independent_float64_metric_max_abs_error':maximum,'errors':bad,'source_lock_match':matched,'same_session_audit':True,'external_replication':False}
    (ROOT/'VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));assert not bad and matched

def retrain():
    lr=json.loads((ROOT/'FRESH_LOCK.json').read_text())['learning_rate'];original=json.loads((ROOT/'FRESH_RECORDS.json').read_text());results=[]
    w=CFG['fresh_worlds'][0];fam='aligned';n=512
    for method in ['mirror1_free','mirror8_free','mirror8_warm','gain8_free','independent_free']:
        r=fit(w,fam,n,method,CFG['fresh_inits'],lr,ROOT/'results'/'replay'/method)
        for row in r['records']:
            parent=next(x for x in original if (x['world'],x['family'],x['ntrain'],x['method'],x['init'])==(w,fam,n,method,row['init']))
            results.append({'method':method,'init':row['init'],'final_exact':row['sha256']==parent['sha256'],'best_exact':row['best_sha256']==parent['best_sha256']})
    (ROOT/'RETRAIN_VERIFICATION.json').write_text(json.dumps(results,indent=2));assert all(r['final_exact'] and r['best_exact'] for r in results);print('exact retrain',len(results))

def benchmark():
    torch.set_num_threads(1);threadpool_limits(1);rows=json.loads((ROOT/'FRESH_RECORDS.json').read_text());results=[]
    for method in ['mirror1_free','mirror4_free','mirror8_free','mirror8_warm','gain8_free','independent_free']:
        row=next(r for r in rows if r['world']==97101 and r['family']=='aligned' and r['ntrain']==512 and r['method']==method and r['init']==0)
        m=load(ROOT/row['file']);d=world(97101,'aligned')
        for batch in [1,64]:
            x=torch.from_numpy(d['test_x'][:batch]);c=torch.from_numpy(d['test_c'][:batch]);groups=[]
            with torch.no_grad():
                for _ in range(30):m(x,c)
                for _ in range(5):
                    t=time.perf_counter_ns()
                    for _ in range(100):m(x,c)
                    groups.append((time.perf_counter_ns()-t)/100/1e6)
            results.append({'method':method,'batch':batch,'median_ms':float(np.median(groups)),'min_ms':min(groups),'max_ms':max(groups),'group_ms':groups})
    report={'results':results,'torch_threads':torch.get_num_threads(),'threadpools':threadpool_info(),'clock_locked':False,'compiled':False,'all_two_experts_evaluated':True}
    (ROOT/'BENCHMARK.json').write_text(json.dumps(report,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['verify','retrain','benchmark']);args=ap.parse_args()
    {'verify':verify_all,'retrain':retrain,'benchmark':benchmark}[args.action]()
