#!/usr/bin/env python3
"""MA-1053: frozen-encoder MOMENT task heads with shared pre-head basis views.

Official MOMENT code is imported from the pinned checkout given by MOMENT_SOURCE.
All task-side features are captured immediately before the native output head.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'runs/raw'; RUNS=ROOT/'runs'
BASE_REPO='AutonLab/MOMENT-1-base'; BASE_REV='5e44b0ea26376a176360f87831124e018f876d96'
SOURCE_COMMIT='38f7310ad594100747ca2a8357e9c7ca7d323e0e'
D=768; PATCHES=64; PATCH=8; RANK=32; HORIZON=96; TASKS=('forecast','anomaly','classification'); PARITY_DELTAS=[]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)

def load_moment(task):
    src=os.environ.get('MOMENT_SOURCE','/tmp/moment-official')
    if src not in sys.path: sys.path.insert(0,src)
    from momentfm import MOMENTPipeline
    ckpt=Path.home()/'.cache/huggingface/hub/models--AutonLab--MOMENT-1-base/snapshots'/BASE_REV
    kw={'task_name':task,'freeze_encoder':True,'freeze_embedder':True,'freeze_head':False,'enable_gradient_checkpointing':False}
    if task=='forecasting': kw['forecast_horizon']=HORIZON
    if task=='classification': kw.update(n_channels=1,num_class=5,reduction='mean')
    model=MOMENTPipeline.from_pretrained(str(ckpt),model_kwargs=kw,local_files_only=True)
    model.init(); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model

def batch_extract(model, task, xs, masks, batch=8):
    feat=[]; pred=[]
    def prehook(mod, ins): feat.append(ins[0].detach().cpu())
    h=model.head.register_forward_pre_hook(prehook)
    with torch.inference_mode():
        for i in range(0,len(xs),batch):
            x=torch.from_numpy(xs[i:i+batch]).float()
            m=torch.from_numpy(masks[i:i+batch]).float()
            if task=='forecast': out=model.forecast(x_enc=x,input_mask=m); y=out.forecast.detach().cpu()
            elif task=='classification': out=model.classify(x_enc=x,input_mask=m,reduction='mean'); y=out.logits.detach().cpu()
            else: out=model.reconstruct(x_enc=x,input_mask=m,mask=m); y=out.reconstruction.detach().cpu()
            captured=feat[-1]
            h.remove()
            direct=model.head(captured)
            h=model.head.register_forward_pre_hook(prehook)
            if task in ('forecast','anomaly'): direct=model.normalizer(direct,mode='denorm')
            delta=float((direct.detach().cpu()-y).abs().max())
            if delta>1e-4: raise AssertionError(f'{task} native pre-head parity max={delta}')
            PARITY_DELTAS.append({'task':task,'max_abs_delta':delta})
            pred.append(y)
    h.remove()
    feats=torch.cat(feat,dim=0)
    preds=torch.cat(pred,dim=0)
    if task=='classification': feats=feats.mean(dim=1)
    return feats.numpy().astype(np.float32),preds.numpy().astype(np.float32)

def forecast_data(split):
    df=pd.read_csv(RAW/'ETTm2.csv'); vals=df['OT'].to_numpy(np.float32)
    ntr,nv,nte=12*30*24*4,4*30*24*4,4*30*24*4
    mu=float(vals[:ntr].mean()); sd=float(vals[:ntr].std()); z=(vals-mu)/sd
    if split=='train':
        starts=np.arange(0,ntr-512-HORIZON+1,192,dtype=int)
        rng=np.random.RandomState(105301); starts=np.sort(rng.choice(starts,min(128,len(starts)),replace=False))
    elif split=='dev': starts=np.arange(ntr+512,ntr+nv-HORIZON+1,192,dtype=int)
    else: starts=np.arange(ntr+nv,ntr+nv+nte-HORIZON+1,HORIZON,dtype=int)
    x=np.stack([z[s:s+512] for s in starts])[:,None,:]
    y_raw=np.stack([z[s+512:s+512+HORIZON] for s in starts])[:,None,:]
    xmean=x.mean(axis=-1,keepdims=True); xstd=x.std(axis=-1,keepdims=True)+1e-5
    y=(y_raw-xmean)/xstd
    m=np.ones((len(x),512),np.float32)
    mase_scale=float(np.mean(np.abs(z[96:ntr]-z[:ntr-96])))
    return x,m,y,{'starts':starts.tolist(),'mean':mu,'std':sd,'mase_scale':mase_scale,'window_mean':xmean[:,0,0].tolist(),'window_std':xstd[:,0,0].tolist(),'y_raw':y_raw.tolist(),'train_rows':ntr,'dev_rows':nv,'audit_rows':nte}

def read_ecg(split):
    src=os.environ.get('MOMENT_SOURCE','/tmp/moment-official')
    if src not in sys.path: sys.path.insert(0,src)
    from momentfm.utils.data import load_from_tsfile
    tr,yl=load_from_tsfile(str(RAW/'ECG5000_TRAIN.ts')); tr=np.asarray(tr).reshape(len(tr),-1)
    classes=np.unique(yl); mapping={v:i for i,v in enumerate(classes)}
    yl=np.asarray([mapping[v] for v in yl],np.int64)
    if split=='audit':
        te,yt=load_from_tsfile(str(RAW/'ECG5000_TEST.ts')); te=np.asarray(te).reshape(len(te),-1); yt=np.asarray([mapping[v] for v in yt],np.int64)
    else: te=yt=None
    fit_idx,dev_idx=train_test_split(np.arange(len(tr)),test_size=.2,random_state=105301,stratify=yl)
    mu=float(tr[fit_idx].mean()); sd=float(tr[fit_idx].std())
    def prep(arr,labels,idx):
        vals=(arr[idx].astype(np.float32)-mu)/(sd+1e-8)
        n,L=vals.shape; x=np.zeros((n,1,512),np.float32); m=np.zeros((n,512),np.float32)
        x[:,0,512-L:]=vals; m[:,512-L:]=1
        return x,m,labels[idx]
    out={'fit':prep(tr,yl,fit_idx),'dev':prep(tr,yl,dev_idx)}
    if split=='audit': out['audit']=prep(te,yt,np.arange(len(te)))
    return out, {'mean':mu,'std':sd,'class_names':[str(c) for c in classes], 'fit_indices':fit_idx.tolist(),'dev_indices':dev_idx.tolist()}

def read_anomaly(split):
    arr=pd.read_csv(RAW/'198_UCR_Anomaly_tiltAPB2_50000_124159_124985.out',header=None).to_numpy(np.float32)
    vals=arr[:,0]; labels=arr[:,1].astype(np.int64); ntrain=50000
    mean=float(vals[:45000].mean()); std=float(vals[:45000].std())
    if split=='train': starts=np.arange(0,45000-512+1,256); end=ntrain; which='train'
    elif split=='dev': starts=np.arange(45000,50000,512); end=ntrain; which='calib'
    else: starts=np.arange(ntrain,len(vals),512); end=len(vals); which='audit'
    xs=[]; ms=[]; ys=[]; ls=[]; means=[]; stds=[]; starts_saved=[]
    for s in starts:
        e=min(s+512,end); raw=vals[s:e]
        mask=np.zeros(512,np.float32); mask[:len(raw)]=1
        x=np.zeros(512,np.float32); x[:len(raw)]=raw
        xs.append(x[None,:]); ms.append(mask); means.append(float(raw.mean())); stds.append(float(raw.std()+1e-6)); starts_saved.append(int(s))
        if which=='train':
            ys.append(((raw-raw.mean())/(raw.std()+1e-6)).astype(np.float32)); ls.append(np.zeros(len(raw),np.int64))
        elif which=='calib': ls.append(np.zeros(len(raw),np.int64))
        else: ls.append(labels[s:e])
    return np.asarray(xs,np.float32),np.asarray(ms,np.float32),{'targets':ys,'labels':ls,'means':means,'stds':stds,'starts':starts_saved,'raw_values':[vals[s:min(s+512,end)].tolist() for s in starts],'raw_labels':[labels[s:min(s+512,end)].tolist() for s in starts]}

def load_all(split):
    # Fit sets only use the normal training prefix / official train split.
    fx,fm,fy,meta=forecast_data('train')
    train_ecg,ecgm=read_ecg(split)
    ax,am,adata=read_anomaly('train')
    train={}
    train['forecast']={'x':fx,'mask':fm,'y':fy}
    x,m,y=train_ecg['fit']; train['classification']={'x':x,'mask':m,'y':y}
    train['anomaly']={'x':ax,'mask':am,'y':adata['targets'],'mask_y':[np.ones(len(v),np.float32) for v in adata['targets']]}
    data={}
    for t in TASKS:
        model=load_moment({'forecast':'forecasting','classification':'classification','anomaly':'reconstruction'}[t])
        if t=='forecast':
            cf=Path('/tmp/ma1053_feature_cache_forecast.npy')
            if cf.exists(): train[t]['x']=np.load(cf)
            else:
                trf,_=batch_extract(model,'forecast',train[t]['x'],train[t]['mask']); train[t]['x']=trf; np.save(cf,trf)
            xx,mm,yy,info=forecast_data(split); ff,nn=batch_extract(model,'forecast',xx,mm); data[t]={'f':ff,'y':yy,'meta':info,'native':nn,'raw_x':xx}
        elif t=='classification':
            cf=Path('/tmp/ma1053_feature_cache_classification.npy')
            if cf.exists(): train[t]['x']=np.load(cf)
            else:
                trf,_=batch_extract(model,'classification',train[t]['x'],train[t]['mask']); train[t]['x']=trf; np.save(cf,trf)
            xx,mm,yy=train_ecg[split]; ff,nn=batch_extract(model,'classification',xx,mm); data[t]={'f':ff,'y':yy,'meta':ecgm,'native':nn}
        else:
            cf=Path('/tmp/ma1053_feature_cache_anomaly.npy')
            if cf.exists(): train[t]['x']=np.load(cf)
            else:
                trf,_=batch_extract(model,'anomaly',train[t]['x'],train[t]['mask']); train[t]['x']=trf; np.save(cf,trf)
            xx,mm,info=read_anomaly(split); ff,nn=batch_extract(model,'anomaly',xx,mm); data[t]={'f':ff,'meta':info,'native':nn}
        del model
    feature_bytes=sum(v['x'].nbytes for v in train.values())+sum(d['f'].nbytes for d in data.values())
    assert all(len(train[t]['x'])==len(train[t]['y']) for t in ('forecast','classification'))
    assert len(train['anomaly']['x'])==len(train['anomaly']['y'])
    return train,data,meta,ecgm,feature_bytes

class TaskHeads(nn.Module):
    def __init__(self,mode,rank=RANK,independent=False):
        super().__init__(); self.mode=mode; self.rank=rank; self.independent=independent
        n=3
        self.proj=nn.ModuleList([nn.Linear(D,rank,bias=False) for _ in range(n if independent else 1)])
        self.forecast=nn.Linear(PATCHES*rank,HORIZON)
        self.anomaly=nn.Linear(rank,PATCH,bias=True)
        self.classification=nn.Linear(rank,5)
        self.codes=nn.Parameter(torch.zeros(n,8)) if mode in ('mirror','diag') else None
    def transform(self,x,ti):
        if self.mode=='mirror':
            # Eight learned angles; each controls 48 adjacent pairs in its feature group.
            c=self.codes[ti]
            blocks=x.reshape(*x.shape[:-1],8,96)
            pairs=blocks.reshape(*blocks.shape[:-1],48,2)
            ang=(math.pi*torch.tanh(c)).view(*([1]*(pairs.ndim-3)),8,1)
            co=torch.cos(ang); si=torch.sin(ang)
            a,b=pairs[...,0],pairs[...,1]
            rotated=torch.stack((co*a-si*b,si*a+co*b),dim=-1)
            return rotated.reshape_as(x)
        if self.mode=='diag':
            g=torch.exp(torch.tanh(self.codes[ti])).repeat_interleave(96)
            return x*g
        return x
    def forward_task(self,x,ti):
        v=self.transform(x,ti)
        p=self.proj[ti if self.independent else 0](v)
        if ti==0: return self.forecast(p.flatten(start_dim=1))
        if ti==1: return self.anomaly(p)
        return self.classification(p)

def task_index(t): return {'forecast':0,'anomaly':1,'classification':2}[t]

def task_batches(train,seed,batch=8):
    rng=np.random.RandomState(seed); packs={}
    for t in TASKS:
        n=len(train[t]['x']); packs[t]=[rng.randint(0,n,size=batch) for _ in range(30)]
    return packs

def loss_for(m,t,b,idx):
    i=task_index(t); x=torch.from_numpy(b['x'][idx]).float(); y=b['y']
    yp=m.forward_task(x,i)
    if t=='forecast': return F.mse_loss(yp,torch.from_numpy(y[idx,0]).float())
    if t=='classification': return F.cross_entropy(yp,torch.from_numpy(y[idx]).long())
    target=np.stack([b['y'][j] for j in idx]); target=torch.from_numpy(target).float()
    target=target.reshape(-1,1,PATCHES,PATCH)
    return F.mse_loss(yp,target)

def init_model(mode,seed,independent=False):
    seed_all(seed); return TaskHeads(mode,independent=independent)

def train_variant(name,mode,train,seed,independent=False,updates=30):
    model=init_model(mode,seed,independent); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    packs=task_batches(train,seed+1); start=time.perf_counter(); model.train()
    for k in range(updates):
        for t in TASKS:
            opt.zero_grad(set_to_none=True); loss=loss_for(model,t,train[t],packs[t][k]); loss.backward(); opt.step()
    sec=time.perf_counter()-start; model.eval()
    return model,sec

def eval_variant(model,t,d):
    i=task_index(t); f=torch.from_numpy(d['f']).float()
    with torch.inference_mode():
        pred=[]
        for s in range(0,len(f),32): pred.append(model.forward_task(f[s:s+32],i).cpu().numpy())
    return np.concatenate(pred,axis=0)

def metric_forecast(y,p,meta):
    y=np.asarray(meta['y_raw'])[:,0,:]; p=np.asarray(p)
    if p.ndim==3: p=p[:,0,:]
    mu=np.asarray(meta['window_mean'])[:,None]; sd=np.asarray(meta['window_std'])[:,None]
    p=p*sd+mu; e=p-y; scale=meta['mase_scale']
    return {'mse':float(np.mean(e**2)),'mae':float(np.mean(np.abs(e))),'mase':float(np.mean(np.abs(e))/scale)}

def metric_class(y,p):
    pred=p.argmax(-1); y=np.asarray(y); classes=np.unique(np.concatenate((y,pred)))
    f1=[]
    for c in range(5):
        tp=np.sum((pred==c)&(y==c)); fp=np.sum((pred==c)&(y!=c)); fn=np.sum((pred!=c)&(y==c)); f1.append(2*tp/(2*tp+fp+fn+1e-12))
    return {'accuracy':float(np.mean(pred==y)),'macro_f1':float(np.mean(f1))}

def anomaly_point_errors(pred,meta,normalized=False):
    scores=[]; labels=[]; starts=[]
    for i,(raw,lab,mu,sd,s) in enumerate(zip(meta['raw_values'],meta['raw_labels'],meta['means'],meta['stds'],meta['starts'])):
        raw=np.asarray(raw,np.float32); lab=np.asarray(lab,np.int64)
        if pred is None: continue
        if normalized:
            z=np.asarray(pred[i,0]).reshape(-1)
            rec=z[:len(raw)]*sd+mu
        else: rec=np.asarray(pred[i,0]).reshape(-1)[:len(raw)]
        scores.append((rec-raw)**2); labels.append(lab); starts.extend(range(s,s+len(raw)))
    return np.concatenate(scores),np.concatenate(labels),np.asarray(starts)

def anomaly_metrics(scores,labels,threshold):
    from sklearn.metrics import average_precision_score
    pred=scores>threshold; y=labels.astype(bool)
    tp=int(np.sum(pred&y)); fp=int(np.sum(pred&~y)); fn=int(np.sum(~pred&y))
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
    f1=2*precision*recall/max(precision+recall,1e-12)
    return {'point_f1':float(f1),'precision':float(precision),'recall':float(recall),'tp':tp,'fp':fp,'fn':fn,'auprc':float(average_precision_score(y.astype(int),scores)) if y.any() else None,'fpr':float(np.mean(pred[~y])) if (~y).any() else None}

def pack_payload(model,path,metadata):
    st={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    save_file(st,str(path)); mpath=path.with_suffix('.json'); mpath.write_text(json.dumps(metadata,sort_keys=True,indent=2)+'\n')
    return path.stat().st_size+mpath.stat().st_size

def run(split,seed):
    seed_all(seed); t0=time.perf_counter(); train,data,forecast_meta,ecg_meta,feature_bytes=load_all(split)
    variants=[('shared','shared',False),('diag','diag',False),('mirror','mirror',False),('independent','shared',True)]
    results={'experiment_id':'MA-1053','split':split,'seed':seed,'base_model':BASE_REPO,'base_revision':BASE_REV,'official_code_commit':SOURCE_COMMIT,'feature_cache_bytes':feature_bytes,'methods':{},'data':{}}
    anomaly_dev=read_anomaly('dev') if split=='dev' else None
    for name,mode,indep in variants:
        model,sec=train_variant(name,mode,train,seed,indep)
        state_path=RUNS/split/f'{name}.safetensors'; state_path.parent.mkdir(parents=True,exist_ok=True)
        adap=pack_payload(model,state_path,{'method':name,'mode':mode,'rank':RANK,'task_order':list(TASKS),'task_codes_per_role':8 if mode in ('mirror','diag') else 0,'task_codes_total':24 if mode in ('mirror','diag') else 0,'seed':seed,'independent':indep})
        reloaded=init_model(mode,seed,indep); reloaded.load_state_dict(load_file(str(state_path))); reloaded.eval()
        results['methods'][name]={'training_wall_s':sec,'adaptation_bytes':adap,'predictions':{},'metrics':{},'serialized_replay_max_delta':0.0}
        for t,d in data.items():
            p=eval_variant(model,t,d); pr=eval_variant(reloaded,t,d)
            results['methods'][name]['serialized_replay_max_delta']=max(results['methods'][name]['serialized_replay_max_delta'],float(np.max(np.abs(p-pr))))
            if results['methods'][name]['serialized_replay_max_delta']>1e-6: raise AssertionError('payload replay mismatch')
            results['methods'][name]['predictions'][t]=p.tolist()
            if t=='forecast': results['methods'][name]['metrics'][t]=metric_forecast(d['y'],p,d['meta'])
            elif t=='classification': results['methods'][name]['metrics'][t]=metric_class(d['y'],p)
            else:
                if split=='dev':
                    _,_,cal=anomaly_dev
                    # The dev partition contains normal calibration rows only.
                    raw_preds=[]
                    for i,(mu,sd,raw) in enumerate(zip(cal['means'],cal['stds'],cal['raw_values'])):
                        z=p[i,0].reshape(-1)[:len(raw)]; rec=z*sd+mu; raw_preds.extend(((rec-np.asarray(raw))**2).tolist())
                    results['methods'][name]['metrics'][t]={'calibration_error_q99':float(np.quantile(raw_preds,.99)),'calibration_normal_fpr_q99':0.01}
                else:
                    # Threshold provenance is loaded from the frozen dev run.
                    dev=json.loads((RUNS/'dev'/'run.json').read_text())
                    threshold=dev['methods'][name]['metrics']['anomaly']['calibration_error_q99']
                    score,label,_=anomaly_point_errors(p,d['meta'],normalized=True)
                    results['methods'][name]['metrics'][t]=anomaly_metrics(score,label,threshold)
        # Pay all inference metadata, including task scalers, routing and anomaly threshold.
        threshold=(results['methods'][name]['metrics']['anomaly']['calibration_error_q99'] if split=='dev' else json.loads((RUNS/'dev'/'run.json').read_text())['methods'][name]['metrics']['anomaly']['calibration_error_q99'])
        payload_meta={'method':name,'mode':mode,'rank':RANK,'task_order':list(TASKS),'task_codes_per_role':8 if mode in ('mirror','diag') else 0,'task_codes_total':24 if mode in ('mirror','diag') else 0,'seed':seed,'independent':indep,'forecast_scaler':{'mean':forecast_meta['mean'],'std':forecast_meta['std']},'classification_scaler':{'mean':ecg_meta['mean'],'std':ecg_meta['std'],'class_names':ecg_meta['class_names']},'anomaly_threshold_squared_error_q99':threshold}
        state_path.with_suffix('.json').write_text(json.dumps(payload_meta,sort_keys=True,indent=2)+'\n')
        results['methods'][name]['adaptation_bytes']=state_path.stat().st_size+state_path.with_suffix('.json').stat().st_size
        results['methods'][name]['inference_metadata']=payload_meta
        del reloaded,model
    # Native controls: pretrained reconstruction head and fitted native forecaster/classifier.
    results['native']={}
    for t,d in data.items():
        if t=='forecast':
            native=nn.Linear(PATCHES*D,HORIZON)
        elif t=='classification': native=nn.Linear(D,5)
        else: native=None
        if native is None:
            p=d['native']
            if split=='dev':
                _,_,cal=anomaly_dev; errs=[]
                for i,(mu,sd,raw) in enumerate(zip(cal['means'],cal['stds'],cal['raw_values'])): errs.extend(((p[i,0,:len(raw)]-raw)**2).tolist())
                threshold=float(np.quantile(errs,.99)); metric={'calibration_error_q99':threshold,'calibration_normal_fpr_q99':0.01}
            else:
                dev=json.loads((RUNS/'dev'/'run.json').read_text()); threshold=dev['native']['anomaly']['metrics']['calibration_error_q99']
                score,label,_=anomaly_point_errors(p,d['meta'],normalized=False); metric=anomaly_metrics(score,label,threshold)
            nmeta={'method':'native_pretrained_reconstruction','task':t,'anomaly_threshold_squared_error_q99':threshold}; npath=RUNS/split/'native_anomaly.json'; npath.write_text(json.dumps(nmeta,sort_keys=True,indent=2)+'\n')
            results['native'][t]={'metrics':metric,'adaptation_bytes':npath.stat().st_size,'predictions':p.tolist(),'inference_metadata':nmeta}
            continue
        opt=torch.optim.AdamW(native.parameters(),lr=1e-3,weight_decay=1e-4); rng=np.random.RandomState(seed+task_index(t)+999); st=time.perf_counter()
        for _ in range(30):
            ix=rng.randint(0,len(train[t]['x']),size=8); f=torch.from_numpy(train[t]['x'][ix]).float()
            opt.zero_grad(set_to_none=True)
            if t=='forecast': pred=native(f.flatten(1)); y=torch.from_numpy(train[t]['y'][ix]).float(); loss=F.mse_loss(pred,y[:,0])
            else: pred=native(f); y=torch.from_numpy(train[t]['y'][ix]).long(); loss=F.cross_entropy(pred,y)
            loss.backward(); opt.step()
        sec=time.perf_counter()-st; f=torch.from_numpy(d['f']).float()
        with torch.inference_mode(): p=native(f.flatten(1) if t=='forecast' else f)
        p=p.numpy().astype(np.float32); path=RUNS/split/f'native_{t}.safetensors'; native_meta={'method':'native_head','task':t,'seed':seed,'forecast_scaler':{'mean':forecast_meta['mean'],'std':forecast_meta['std']} if t=='forecast' else None,'classification_scaler':{'mean':ecg_meta['mean'],'std':ecg_meta['std'],'class_names':ecg_meta['class_names']} if t=='classification' else None}; byte=pack_payload(native,path,native_meta)
        native_reload=nn.Linear(PATCHES*D,HORIZON) if t=='forecast' else nn.Linear(D,5)
        native_reload.load_state_dict(load_file(str(path))); native_reload.eval()
        with torch.inference_mode(): pr=native_reload(f.flatten(1) if t=='forecast' else f).numpy().astype(np.float32)
        replay_delta=float(np.max(np.abs(p-pr)))
        if replay_delta>1e-6: raise AssertionError(f'native {t} payload replay mismatch {replay_delta}')
        metric=metric_forecast(d['y'],p[:,None,:],d['meta']) if t=='forecast' else metric_class(d['y'],p)
        results['native'][t]={'metrics':metric,'adaptation_bytes':byte,'training_wall_s':sec,'predictions':p.tolist(),'inference_metadata':native_meta,'serialized_replay_max_delta':replay_delta}
    # Cheap task-specific controls; quality metrics remain in native units.
    fd=data['forecast']; last=fd['raw_x'][:,0,-HORIZON:]
    results['simple_baselines']={'seasonal_naive_forecast':metric_forecast(fd['y'],last[:,None,:],fd['meta'])}
    ytrain=train['classification']['y']; majority=int(np.bincount(ytrain,minlength=5).argmax())
    yhat=np.zeros((len(data['classification']['y']),5),np.float32); yhat[:,majority]=1.0
    results['simple_baselines']['class_majority']=metric_class(data['classification']['y'],yhat)
    results['simple_baselines']['class_majority_label']=majority
    results['simple_baselines']['class_majority_payload_bytes']=16
    results['simple_baselines']['seasonal_naive_payload_bytes']=16
    # Per-example task-side MAC proxies; shared MOMENT encoder is 5.51G MAC/example.
    projection=PATCHES*D*RANK; rot=PATCHES*D*3; diagonal=PATCHES*D
    results['active_macs_per_example']={'shared_encoder':5511315456,'shared_projection':projection,'forecast_head':PATCHES*RANK*HORIZON,'anomaly_head':PATCHES*RANK*PATCH,'classification_head':RANK*5,'mirror_rotation_extra':rot,'diagonal_gain_extra':diagonal,'full_native_forecast_head':PATCHES*D*HORIZON,'native_classification_head':D*5,'native_pretrained_anomaly_head':PATCHES*D*PATCH}
    results['training_budget']={'task_updates_each':30,'updates_total_shared_methods':90,'batch_size':8,'examples_per_task':240,'train_windows_per_task':{t:len(train[t]['x']) for t in TASKS}}
    common=453_941_069
    for methods in list(results['methods'].values())+[v for v in results['native'].values() if isinstance(v,dict)]:
        if 'adaptation_bytes' in methods: methods['full_payload_bytes']=common+methods['adaptation_bytes']
    results['common_base_and_config_bytes']=common
    results['native_prehead_parity_max_abs_delta']=max((v['max_abs_delta'] for v in PARITY_DELTAS),default=0.0)
    results['native_prehead_parity_checks']=len(PARITY_DELTAS)
    results['feature_extraction_and_run_wall_s']=time.perf_counter()-t0
    results['status']='SCREENING_COMPLETE'
    outfile=RUNS/split/'run.json'; outfile.write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--split',choices=['dev','audit'],required=True); ap.add_argument('--seed',type=int,default=None); a=ap.parse_args()
    seed=a.seed or (105301 if a.split=='dev' else 105302)
    r=run(a.split,seed)
    for name,v in r['methods'].items(): print(name, json.dumps(v['metrics'],sort_keys=True),v['adaptation_bytes'],round(v['training_wall_s'],2))
    print('native',json.dumps({k:v['metrics'] for k,v in r['native'].items()},sort_keys=True))
    print('wall_s',round(r['feature_extraction_and_run_wall_s'],2),'feature_bytes',r['feature_cache_bytes'])
