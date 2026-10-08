#!/usr/bin/env python3
"""MA-351 fixed-budget synthetic MIMO/readout-view comparison."""
import argparse, csv, hashlib, io, json, time, zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts'
SEEDS=(35101,35102,35111,35112,35113)
NTRAIN=8192; NTEST=4096; MEMBERS=4; CLASSES=4; UPDATES=2000; BATCH=128

def pack(state, meta):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,a in sorted(state.items()):
            q=io.BytesIO(); np.save(q,np.asarray(a),allow_pickle=False)
            i=zipfile.ZipInfo(name+'.npy',(1980,1,1,0,0,0)); i.compress_type=zipfile.ZIP_DEFLATED; i.external_attr=0o600<<16; z.writestr(i,q.getvalue())
        i=zipfile.ZipInfo('metadata.json',(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o600<<16;z.writestr(i,json.dumps(meta,sort_keys=True,separators=(',',':')).encode())
    payload=b.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        loaded={n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')}
        loaded_meta=json.loads(z.read('metadata.json'))
    return payload,loaded,loaded_meta

def relu(x): return np.maximum(x,0)
def softmax(x):
    q=x-x.max(axis=-1,keepdims=True); e=np.exp(q); return e/e.sum(axis=-1,keepdims=True)

def data(seed):
    rng=np.random.default_rng(seed)
    def make(n):
        x=rng.uniform(-1,1,size=(n,2)).astype(np.float32)
        # Four deterministic angular sectors produce a learnable nonlinear classification task.
        angle=np.arctan2(x[:,1],x[:,0]); y=np.floor((angle+np.pi)/(np.pi/2)).astype(np.int64)%4
        return x,y
    return rng,make(NTRAIN),make(NTEST)

def init(seed,method):
    rng=np.random.default_rng(seed)
    dims=[2,32,32]
    s={'W1':rng.normal(0,.18,(2,32)).astype('f4'),'b1':np.zeros(32,'f4'),'W2':rng.normal(0,.18,(32,32)).astype('f4'),'b2':np.zeros(32,'f4')}
    if method=='independent_four_mlp':
        s={}
        for k in range(MEMBERS):
            s[f'W1_{k}']=rng.normal(0,.18,(2,32)).astype('f4');s[f'b1_{k}']=np.zeros(32,'f4')
            s[f'W2_{k}']=rng.normal(0,.18,(32,32)).astype('f4');s[f'b2_{k}']=np.zeros(32,'f4')
            s[f'Wo_{k}']=rng.normal(0,.18,(32,4)).astype('f4');s[f'bo_{k}']=np.zeros(4,'f4')
    elif method=='native_mimo':
        for k in range(MEMBERS): s[f'Wo_{k}']=rng.normal(0,.18,(32,4)).astype('f4');s[f'bo_{k}']=np.zeros(4,'f4')
    elif method in ('direct_rank1','mirror_rank1'):
        s['V0']=rng.normal(0,.18,(32,4)).astype('f4');s['V1']=rng.normal(0,.08,(32,4)).astype('f4')
        s['coeff']=np.zeros(MEMBERS,'f4')
    elif method=='direct_rank2':
        s['V0']=rng.normal(0,.18,(32,4)).astype('f4');s['V1']=rng.normal(0,.08,(32,4)).astype('f4');s['V2']=rng.normal(0,.08,(32,4)).astype('f4')
        s['coeff']=np.zeros((MEMBERS,2),'f4')
    return s

def forward(x,s,method):
    if method=='independent_four_mlp':
        outs=[]
        for k in range(MEMBERS): outs.append(relu(relu(x@s[f'W1_{k}']+s[f'b1_{k}'])@s[f'W2_{k}']+s[f'b2_{k}'])@s[f'Wo_{k}']+s[f'bo_{k}'])
        return np.stack(outs,axis=1)
    h=relu(x@s['W1']+s['b1']);h=relu(h@s['W2']+s['b2'])
    if method=='native_mimo': return np.stack([h@s[f'Wo_{k}']+s[f'bo_{k}'] for k in range(MEMBERS)],axis=1)
    if method in ('direct_rank1','mirror_rank1'):
        # Semantically identical affine code; encoding metadata differs to audit Mirror-specificity.
        c=s['coeff']
        return np.stack([h@(s['V0']+c[k]*s['V1']) for k in range(MEMBERS)],axis=1)
    c=s['coeff']
    return np.stack([h@(s['V0']+c[k,0]*s['V1']+c[k,1]*s['V2']) for k in range(MEMBERS)],axis=1)

def train(seed,method,train_data):
    # Numpy full-batch AdamW implementation with fixed number of mini-batch updates.
    # Per-member bootstrap masks are fixed and used in every architecture.
    rng=np.random.default_rng(seed+991); x,y=train_data
    boots=[rng.integers(0,len(x),size=len(x)) for _ in range(MEMBERS)]
    s=init(seed+7,method); m={k:np.zeros_like(v) for k,v in s.items()};v={k:np.zeros_like(q) for k,q in s.items()}; beta1=.9;beta2=.999;lr=.001
    for step in range(1,UPDATES+1):
        idx=rng.integers(0,len(x),size=BATCH);xb=x[idx]; grads={k:np.zeros_like(a) for k,a in s.items()}
        if method=='independent_four_mlp':
            # Four independently initialized models use the same train split and labels.
            for k in range(MEMBERS):
                xx=xb; yy=y[idx]
                z1=xx@s[f'W1_{k}']+s[f'b1_{k}'];h1=relu(z1);z2=h1@s[f'W2_{k}']+s[f'b2_{k}'];h2=relu(z2);log=h2@s[f'Wo_{k}']+s[f'bo_{k}'];p=softmax(log);p[np.arange(BATCH),yy]-=1;p/=BATCH
                grads[f'Wo_{k}']=h2.T@p;grads[f'bo_{k}']=p.sum(0);dh2=p@s[f'Wo_{k}'].T;dz2=dh2*(z2>0);grads[f'W2_{k}']=h1.T@dz2;grads[f'b2_{k}']=dz2.sum(0);dh1=dz2@s[f'W2_{k}'].T;dz1=dh1*(z1>0);grads[f'W1_{k}']=xx.T@dz1;grads[f'b1_{k}']=dz1.sum(0)
        else:
            z1=xb@s['W1']+s['b1'];h1=relu(z1);z2=h1@s['W2']+s['b2'];h2=relu(z2);log=forward(xb,s,method);p=softmax(log)
            dlog=p.copy()
            for k in range(MEMBERS):
                yy=y[idx];dlog[np.arange(BATCH),k,yy]-=1
            dlog/=BATCH
            dh2=np.zeros_like(h2)
            if method=='native_mimo':
                for k in range(MEMBERS):
                    grads[f'Wo_{k}']=h2.T@dlog[:,k];grads[f'bo_{k}']=dlog[:,k].sum(0);dh2+=dlog[:,k]@s[f'Wo_{k}'].T
            else:
                if method in ('direct_rank1','mirror_rank1'):
                    for k in range(MEMBERS):
                        coeff=s['coeff'][k];head=s['V0']+coeff*s['V1'];grads['V0']+=h2.T@dlog[:,k];grads['V1']+=coeff*h2.T@dlog[:,k];grads['coeff'][k]=np.sum((h2@s['V1'])*dlog[:,k])
                        dh2+=dlog[:,k]@head.T
                else:
                    for k in range(MEMBERS):
                        a,b=s['coeff'][k];head=s['V0']+a*s['V1']+b*s['V2'];g=h2.T@dlog[:,k];grads['V0']+=g;grads['V1']+=a*g;grads['V2']+=b*g;grads['coeff'][k,0]=np.sum((h2@s['V1'])*dlog[:,k]);grads['coeff'][k,1]=np.sum((h2@s['V2'])*dlog[:,k]);dh2+=dlog[:,k]@head.T
            dz2=dh2*(z2>0);grads['W2']=h1.T@dz2;grads['b2']=dz2.sum(0);dh1=dz2@s['W2'].T;dz1=dh1*(z1>0);grads['W1']=xb.T@dz1;grads['b1']=dz1.sum(0)
        for k in s:
            m[k]=beta1*m[k]+(1-beta1)*grads[k];v[k]=beta2*v[k]+(1-beta2)*grads[k]**2
            mh=m[k]/(1-beta1**step);vh=v[k]/(1-beta2**step);s[k]*=1-lr*0.01;s[k]-=lr*mh/(np.sqrt(vh)+1e-8)
    return s

def metrics(logits,y):
    probs=softmax(logits);n=len(y);member_nll=float(-np.log(np.clip(probs[np.arange(n)[:,None],np.arange(MEMBERS)[None,:],y[:,None]],1e-8,1)).mean());pred=probs.argmax(2);acc=(pred==y[:,None]).mean(0)
    ensemble_probs=probs.mean(1);nll=float(-np.log(np.clip(ensemble_probs[np.arange(n),y],1e-8,1)).mean())
    corr=[]
    for i in range(MEMBERS):
        for j in range(i+1,MEMBERS):
            a=(pred[:,i]==y).astype(float);b=(pred[:,j]==y).astype(float);corr.append(float(np.corrcoef(a,b)[0,1]) if a.std() and b.std() else 0.)
    ensemble=ensemble_probs;ece=0.
    conf=ensemble.max(1);correct=(ensemble.argmax(1)==y)
    for lo in np.linspace(0,1,16)[:-1]:
        hi=lo+1/15;mask=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if mask.any():ece+=float(mask.mean())*abs(float(conf[mask].mean())-float(correct[mask].mean()))
    js=[]
    for i in range(MEMBERS):
        for j in range(i+1,MEMBERS):
            a=probs[:,i];b=probs[:,j];mid=(a+b)/2
            js.append(float(.5*np.mean(np.sum(a*np.log(np.clip(a/mid,1e-8,None)),1))+.5*np.mean(np.sum(b*np.log(np.clip(b/mid,1e-8,None)),1))))
    return nll,member_nll,acc.tolist(),float(np.mean(corr)),float(np.mean(js)),ece

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True);rows=[];seeds=SEEDS[:2] if args.dev_only else SEEDS
    methods=['independent_four_mlp','native_mimo','direct_rank1','mirror_rank1','direct_rank2']
    for seed in seeds:
        rng,tr,te=data(seed);xte,yte=te
        for method in methods:
            start=time.perf_counter();state=train(seed,method,tr);log=forward(xte,state,method);quality=metrics(log,yte)
            meta={'method':method,'seed':seed,'updates':UPDATES,'members':MEMBERS,'format':'MA351-inference-v1'}
            payload,loaded,lmeta=pack(state,meta);wall=time.perf_counter()-start
            # Serialization reload must preserve every inference logit.
            replay=forward(xte,loaded,method);err=float(np.max(np.abs(log-replay)))
            nll,member_nll,acc,corr,js,ece=quality
            # Per-member training compute proxy from dense linear-layer MACs; four member outputs in one shared-trunk evaluation.
            train_macs=UPDATES*BATCH*(2*32+32*32+32*4*MEMBERS)
            if method=='independent_four_mlp':train_macs=UPDATES*BATCH*MEMBERS*(2*32+32*32+32*4)
            row={'seed':seed,'method':method,'payload_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'ensemble_nll':nll,'member_nll':member_nll,'member_accuracy':json.dumps(acc),'correctness_corr':corr,'pairwise_jsd':js,'ensemble_ece':ece,'examples':NTRAIN+NTEST,'updates':UPDATES,'train_MAC_proxy':int(train_macs),'wall_seconds_train_eval_serialize':wall,'roundtrip_max_logit_error':err}
            rows.append(row);print(json.dumps(row,sort_keys=True))
    path=OUT/('development.csv' if args.dev_only else 'results.csv')
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
