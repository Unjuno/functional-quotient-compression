#!/usr/bin/env python3
"""Fit train-cell calibrators, tune on dev, evaluate frozen audit outputs."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

HERE=Path(__file__).resolve().parents[1]; RUNS=HERE/'runs'; SENSORS=list(range(14,22))
FIT_CELLS=((0,0),(0,1),(1,1)); AUDIT_CELL=(1,0); LAMBDAS=(0.,.01,.1)
METHODS=('native','global','frequency_only','horizon_only','additive_log','mirror_factorized','independent','seasonal_naive')

def read_cell(split,f,h,patch,seed=105701):
    z=np.load(RUNS/split/f'frequency{f}_horizon{h}_patch{patch}_seed{seed}.npz')
    return {k:z[k] for k in z.files}

def meta(): return json.loads((RUNS/'input_manifest.json').read_text())

def scale_info(m,f,v):
    q=m['data']['training_only_series_scales'][str(f)]
    local=v-14
    return float(q['mean'][local]),float(q['std'][local]),float(q['seasonal_mase_scale'][local])

def gather(split,cells,patch,seed=105701):
    rows=[]
    for f,h in cells:
        z=read_cell(split,f,h,patch,seed)
        for i in range(len(z['origins'])):
            rows.append({'f':int(f),'h':int(h),'v':int(z['sensor_ids'][i]),'origin':int(z['origins'][i]),
                'samples':z['samples'][i].astype(np.float64),'target':z['targets'][i].astype(np.float64),
                'naive':z['naive'][i].astype(np.float64)})
    return rows

def decode(method,z,f,h):
    f=np.asarray(f,dtype=int); h=np.asarray(h,dtype=int)
    if method=='global': return np.full(len(f),np.exp(z[0])),np.full(len(f),z[1])
    if method=='frequency_only': return np.exp(z[f]),z[2+f]
    if method=='horizon_only': return np.exp(z[h]),z[2+h]
    if method=='additive_log':
        return np.exp(z[0]+z[1]*(f==1)+z[2]*(h==1)),z[3]+z[4]*(f==1)+z[5]*(h==1)
    if method=='mirror_factorized':
        sp=lambda q:np.logaddexp(0.,q)
        return sp(z[f])*sp(z[2+h]),z[4+f]+z[6+h]
    if method=='independent':
        cell=2*f+h; return np.exp(z[cell]),z[4+cell]
    if method in ('native','seasonal_naive'): return np.ones(len(f)),np.zeros(len(f))
    raise ValueError(method)

def initial(method):
    if method=='global': return np.zeros(2)
    if method in ('frequency_only','horizon_only'): return np.zeros(4)
    if method=='additive_log': return np.zeros(6)
    if method=='mirror_factorized':
        q=np.full(8,0.5413248546); q[4:]=0.; return q
    if method=='independent': return np.zeros(8)
    return np.zeros(0)

def packed(rows,m):
    x=[]; y=[]; f=[]; h=[]
    for r in rows:
        mu,sd,_=scale_info(m,r['f'],r['v'])
        x.extend(((r['samples'].mean(0)-mu)/sd).tolist()); y.extend(((r['target']-mu)/sd).tolist())
        f.extend([r['f']]*len(r['target'])); h.extend([r['h']]*len(r['target']))
    return np.asarray(x),np.asarray(y),np.asarray(f),np.asarray(h)

def fit_code(method,rows,m,lam):
    if method in ('native','seasonal_naive'): return np.zeros(0)
    x,y,f,h=packed(rows,m); z0=initial(method)
    def fun(z):
        s,b=decode(method,z,f,h); err=(s*x+b-y)/np.sqrt(len(y)); pen=[]
        for cf,ch in FIT_CELLS:
            si,bi=decode(method,z,[cf],[ch]); pen.extend([np.log(si[0]),bi[0]])
        return np.r_[err,np.sqrt(lam)*np.asarray(pen)]
    return least_squares(fun,z0,max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-10).x

def transformed(raw,method,z,f,h,v,m):
    mu,sd,_=scale_info(m,f,v); s,b=decode(method,z,[f],[h])
    return (mu+sd*(s[0]*(raw-mu)/sd+b[0])).astype(np.float64)

def crps(samples,y):
    x=np.sort(samples,axis=0); n=len(x); coef=(2*np.arange(1,n+1)-n-1)[:,None]
    pair=(x*coef).sum(0)/(n*n)
    return float(np.mean(np.abs(samples-y[None,:])-pair))

def evaluate(rows,method,z,m):
    keys=('crps','wql','mase'); vals={k:[] for k in keys}; sensors={}
    for r in rows:
        f,h,v=r['f'],r['h'],r['v']; _,_,den=scale_info(m,f,v)
        naive=np.tile(r['naive'],int(np.ceil(r['samples'].shape[1]/len(r['naive']))))[:r['samples'].shape[1]]
        pred=(np.broadcast_to(naive[None,:],r['samples'].shape) if method=='seasonal_naive'
              else r['samples'] if method=='native' else transformed(r['samples'],method,z,f,h,v,m))
        y=r['target']; c=crps(pred,y); q=np.quantile(pred,[.1,.5,.9],axis=0); e=y[None,:]-q; tau=np.array([.1,.5,.9])[:,None]
        w=float(2*np.maximum(tau*e,(tau-1)*e).sum()/(3*max(np.abs(y).sum(),1e-9)))
        mase=float(np.mean(np.abs(np.median(pred,axis=0)-y))/den)
        sensors.setdefault(str(v),{k:[] for k in keys})
        for k,val in zip(keys,(c,w,mase)): vals[k].append(val); sensors[str(v)][k].append(val)
    return {k:float(np.mean(v)) for k,v in vals.items()}, {v:{k:float(np.mean(a)) for k,a in d.items()} for v,d in sensors.items()}

def tune():
    m=meta(); patch_scores={}
    for patch in (8,16,32):
        patch_scores[str(patch)]=evaluate(gather('dev',FIT_CELLS,patch),'native',np.zeros(0),m)[0]
    patch=min((8,16,32),key=lambda p:patch_scores[str(p)]['crps'])
    train=gather('train',FIT_CELLS,patch); dev=gather('dev',FIT_CELLS,patch)
    cand={k:[] for k in METHODS if k not in ('native','seasonal_naive')}
    for method in cand:
        for lam in LAMBDAS:
            code=fit_code(method,train,m,lam); score,_=evaluate(dev,method,code,m)
            cand[method].append({'lambda':lam,'code':code.tolist(),'dev':score})
    # The factorized and additive-log forms share one lambda so both use the same function family.
    pair=[]
    for i,lam in enumerate(LAMBDAS):
        pair.append((.5*(cand['additive_log'][i]['dev']['crps']+cand['mirror_factorized'][i]['dev']['crps']),lam))
    pair_lambda=min(pair)[1]
    codes={}; lambdas={}; devscores={}
    for method,opts in cand.items():
        if method in ('additive_log','mirror_factorized'): chosen=next(q for q in opts if q['lambda']==pair_lambda)
        else: chosen=min(opts,key=lambda q:q['dev']['crps'])
        codes[method]=chosen['code']; lambdas[method]=chosen['lambda']; devscores[method]=chosen['dev']
    devscores['native']=evaluate(dev,'native',np.zeros(0),m)[0]
    devscores['seasonal_naive']=evaluate(dev,'seasonal_naive',np.zeros(0),m)[0]
    out={'patch_size':patch,'patch_native_dev_metrics':patch_scores,'ridge':lambdas,'codes':codes,
         'dev_metrics':devscores,'mirror_additive_common_lambda':pair_lambda,'audit_locked':True,
         'fit_cells':[list(c) for c in FIT_CELLS]}
    (RUNS/'dev/tuning.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))

def write_payloads():
    from huggingface_hub import hf_hub_download
    from collect_samples import MODEL,MODEL_REV,CACHE
    weights=Path(hf_hub_download(MODEL,'model.safetensors',revision=MODEL_REV,cache_dir=CACHE,local_files_only=True))
    config=Path(hf_hub_download(MODEL,'config.json',revision=MODEL_REV,cache_dir=CACHE,local_files_only=True))
    common=weights.stat().st_size+config.stat().st_size; selected=json.loads((RUNS/'dev/tuning.json').read_text()); sizes={}
    methods=('native','global','frequency_only','horizon_only','additive_log','mirror_factorized','independent','seasonal_naive')
    for method in methods:
        code=np.asarray(selected['codes'].get(method,[]),dtype='<f4'); folder=RUNS/'audit/payloads'/method; folder.mkdir(parents=True,exist_ok=True)
        coeff=folder/'coefficients.f32le'; coeff.write_bytes(code.tobytes())
        doc={'method':method,'code_shape':list(code.shape),'code_dtype':'float32-little-endian',
             'code_sha256':hashlib.sha256(coeff.read_bytes()).hexdigest(),'patch_size':selected['patch_size'],
             'sensor_ids':list(range(14,22)),'frequency_minutes':[5,15],'horizon_hours':[1,24],
             'model_revision':'0c24ab99db2c1a70ea2a0fc03bf113329772ac64',
             'decoder':'source/fit_controls.py'}
        meta_path=folder/'method.json'; meta_path.write_text(json.dumps(doc,sort_keys=True,separators=(',',':'))+'\n')
        code_bytes=coeff.stat().st_size+meta_path.stat().st_size
        sizes[method]={'common_model_bytes':common,'method_state_bytes':code_bytes,'complete_payload_bytes':common+code_bytes,
            'coefficients_bytes':coeff.stat().st_size,'metadata_bytes':meta_path.stat().st_size}
    return sizes

def score_audit():
    m=meta(); selected=json.loads((RUNS/'dev/tuning.json').read_text()); patch=selected['patch_size']; results=[]
    codes={k:np.asarray(v) for k,v in selected['codes'].items()}
    for seed in (105702,105703):
        rows=gather('audit',(AUDIT_CELL,),patch,seed)
        for method in METHODS:
            score,per=evaluate(rows,method,codes.get(method,np.zeros(0)),m)
            results.append({'seed':seed,'frequency_id':1,'horizon_id':0,'frequency_minutes':15,'horizon_hours':1,
                'method':method,'metrics':score,'per_sensor':per,'n_windows':len(rows),
                'origins':sorted(set(r['origin'] for r in rows))})
    out={'experiment_id':'MA-1057','split':'audit','patch_size':patch,'results':results,
         'payloads':write_payloads(),'sampling_seeds':[105702,105703]}
    (RUNS/'audit/results.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('tune','audit')); a=p.parse_args()
    tune() if a.mode=='tune' else score_audit()
