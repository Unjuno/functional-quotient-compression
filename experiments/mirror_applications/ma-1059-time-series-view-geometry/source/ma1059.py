#!/usr/bin/env python3
"""MA-1059 frozen Chronos adapter screen over Monash monthly Tourism series."""
import argparse, hashlib, json, math, os, time, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from chronos import ChronosPipeline

ROOT=Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'; P=json.loads((ROOT/'PROTOCOL.json').read_text())
DATA=P['data']; SERIES=DATA['series_ids']; FIT_IDS=SERIES['fit']; DEV_IDS=SERIES['development']; AUDIT_IDS=SERIES['audit']
MODEL=P['model']; H=MODEL['forecast_horizon']; CTX=MODEL['context_length']; NSAMP=MODEL['sample_paths']; RANK=2
DATA_REPO=DATA['repository']; DATA_REV=DATA['revision']; DATA_FILE=DATA['file']; DATA_SHA=DATA['file_sha256']
MODEL_ID=MODEL['id']; MODEL_REV=MODEL['revision']; CACHE='/tmp/ma1059-hf'; torch.set_num_threads(1)
SNAP=Path('/home/agent/.cache/huggingface/hub/models--amazon--chronos-t5-tiny/snapshots')/MODEL_REV

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def download_data():
    p=hf_hub_download(DATA_REPO,DATA_FILE,repo_type='dataset',revision=DATA_REV,cache_dir=CACHE,local_files_only=True)
    if sha(p)!=DATA_SHA: raise ValueError('Monash archive hash mismatch')
    return p

def parse_tsf_line(line):
    """Parse a TSF row. Returns series name and values; metadata stays opaque."""
    name, _start, values=line.rstrip('\n').split(':',2)
    return name, np.asarray([float(v) for v in values.split(',')],dtype=np.float32)

def load_series(ids):
    """Only selected named rows are converted to numeric arrays."""
    want=set(ids); path=download_data(); found={}
    with zipfile.ZipFile(path) as z:
        member=next(n for n in z.namelist() if n.endswith('.tsf'))
        with z.open(member) as raw:
            for line in raw:
                s=line.decode('latin-1').strip()
                if not s or s.startswith('#') or s.startswith('@'): continue
                name=s.split(':',1)[0]
                if name in want:
                    parsed,values=parse_tsf_line(s)
                    if parsed in found: raise ValueError('duplicate ID '+parsed)
                    if len(values)<80 or not np.isfinite(values).all(): raise ValueError(f'predeclared series {parsed} structurally ineligible')
                    found[parsed]=values
    if found.keys()!=want: raise ValueError(f'missing registered IDs: {sorted(want-found.keys())}')
    return path,found

def windows(x):
    n=len(x); train_end=int(.60*n); query_start=int(.80*n)
    lo=CTX; hi=train_end-H
    if hi<lo or query_start+H>n: raise ValueError('series too short for registered windows')
    os_=np.unique(np.rint(np.linspace(lo,hi,4)).astype(int)).tolist()
    support=[(x[o-CTX:o],x[o:o+H],o) for o in os_]
    query=[(x[query_start-CTX:query_start],x[query_start:query_start+H],query_start)]
    return {'support':support,'query':query,'train_end':train_end}

def load_pipe():
    pipe=ChronosPipeline.from_pretrained(str(SNAP),device_map='cpu',torch_dtype=torch.float32); pipe.model.eval(); pipe.tokenizer.config.prediction_length=H
    for p in pipe.model.parameters(): p.requires_grad_(False)
    return pipe

class OutputLoRA:
    def __init__(self,pipe,A=None,B=None,train_basis=False):
        self.pipe=pipe; self.layer=pipe.model.model.lm_head
        self.A=torch.nn.Parameter(torch.randn(RANK,256)*.002 if A is None else A.detach().clone(),requires_grad=train_basis)
        self.B=torch.nn.Parameter(torch.zeros(4096,RANK) if B is None else B.detach().clone(),requires_grad=train_basis)
        self.mode='inactive'; self.code=None; self.handle=self.layer.register_forward_hook(self._hook)
    def set(self,mode,code): self.mode=mode; self.code=code
    def _hook(self,module,args,out):
        if self.mode=='inactive': return out
        h=args[0]; z=F.linear(h,self.A); r=RANK
        if self.mode in ('mirror','pooled'):
            t=self.code.reshape(-1)[0]; c=torch.cos(t); s=torch.sin(t)
            zr=torch.stack((c*z[...,0]-s*z[...,1],s*z[...,0]+c*z[...,1]),dim=-1)
        elif self.mode=='diag':
            g=self.code.reshape(-1); scale=torch.stack((1.+g[0],torch.ones_like(g[0]))); zr=z*scale
        elif self.mode=='full': zr=z*self.code.reshape(-1)[:r]
        elif self.mode=='independent':
            a,b=self.code; return out+F.linear(F.linear(h,a),b)
        else: raise ValueError(self.mode)
        return out+F.linear(zr,self.B)
    def loss(self,ctx,target,mode,code):
        self.set(mode,code); x=torch.as_tensor(ctx,dtype=torch.float32).reshape(1,-1); y=torch.as_tensor(target,dtype=torch.float32).reshape(1,-1)
        ids,mask,state=self.pipe.tokenizer.context_input_transform(x); labels,_=self.pipe.tokenizer.label_input_transform(y,state)
        out=self.pipe.model.model(input_ids=ids,attention_mask=mask,labels=labels)
        return F.cross_entropy(out.logits[:,:H].reshape(-1,4096),labels[:,:H].reshape(-1))
    def close(self): self.handle.remove()

def fit_shared(pipe, series, mode='mirror', updates=30, seed=105901):
    hook=OutputLoRA(pipe,train_basis=True)
    if mode=='mirror':
        codes=torch.nn.ParameterDict({i:torch.nn.Parameter(torch.tensor(0.0)) for i in FIT_IDS}); pars=[hook.A,hook.B,*codes.values()]
    else:
        codes=torch.nn.Parameter(torch.zeros(1)); pars=[hook.A,hook.B,codes]
    opt=torch.optim.AdamW(pars,lr=.01,weight_decay=.01)
    rows=[(sid,w) for sid in FIT_IDS for w in windows(series[sid])['support']]; rng=np.random.default_rng(seed); start=time.perf_counter()
    for _ in range(updates):
        sid,(ctx,y,_)=rows[int(rng.integers(len(rows)))]; opt.zero_grad()
        code=codes[sid] if mode=='mirror' else codes
        loss=hook.loss(ctx,y,mode if mode=='mirror' else 'pooled',code); loss.backward(); opt.step()
    sec=time.perf_counter()-start
    saved={k:v.detach().cpu() for k,v in codes.items()} if mode=='mirror' else {'shared':codes.detach().cpu()}
    out={'A':hook.A.detach().cpu(),'B':hook.B.detach().cpu(),'fit_codes':saved,'wall_seconds':sec,'updates':updates}; hook.close(); return out

def adapt(pipe,A,B,support,mode,updates,seed):
    hook=OutputLoRA(pipe,A,B,False)
    if mode=='independent':
        a=torch.nn.Parameter(A.detach().clone()); b=torch.nn.Parameter(B.detach().clone()); code=(a,b); params=[a,b]
    elif mode=='full': code=torch.nn.Parameter(torch.ones(2)); params=[code]
    else: code=torch.nn.Parameter(torch.zeros(1)); params=[code]
    opt=torch.optim.AdamW(params,lr=.01,weight_decay=.01); rng=np.random.default_rng(seed); t=time.perf_counter()
    for _ in range(updates):
        ctx,y,_=support[int(rng.integers(len(support)))]; opt.zero_grad(); loss=hook.loss(ctx,y,mode,code); loss.backward(); opt.step()
    elapsed=time.perf_counter()-t
    if mode=='independent': saved=(code[0].detach().cpu(),code[1].detach().cpu())
    else: saved=code.detach().cpu()
    hook.close(); return saved,elapsed

@torch.no_grad()
def predict(pipe,hook,mode,code,ctx,seed):
    hook.set(mode,code); torch.manual_seed(int(seed))
    return pipe.predict(torch.as_tensor(ctx,dtype=torch.float32),prediction_length=H,num_samples=NSAMP)[0].numpy().astype(np.float32)

def trend_residual(pipe,hook,mode,code,support,seed):
    """Fit one scalar along a fixed centered horizon ramp from support residuals only."""
    ramp=np.linspace(-.5,.5,H,dtype=np.float64); num=den=0.
    for j,(ctx,y,o) in enumerate(support):
        p=predict(pipe,hook,mode,code,ctx,seed+j); err=np.asarray(y,dtype=np.float64)-p.mean(axis=0,dtype=np.float64)
        num+=float(err@ramp); den+=float(ramp@ramp)
    return float(num/max(den,1e-12))

def apply_resid(samples,r):
    ramp=np.linspace(-.5,.5,H,dtype=np.float32)
    return (samples+r*ramp[None,:]).astype(np.float32)

def mase(pred,target,den): return float(np.mean(np.abs(np.median(pred,axis=0)-target))/den)
def mase_den(x):
    n=int(.60*len(x)); return float(np.mean(np.abs(x[12:n]-x[:n-12])))
def crps(samples,target):
    x=np.sort(np.asarray(samples,dtype=np.float64),axis=0); n=len(x); c=(2*np.arange(1,n+1)-n-1)[:,None]
    return float(np.mean(np.abs(samples-target[None,:])-(x*c).sum(0)/(n*n)))
def row_metrics(samples,target,den): return {'mase':mase(samples,target,den),'crps':crps(samples,target)}
def summarize(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in ('mase','crps')}

def run_dev(shared_updates=30):
    torch.manual_seed(105900)
    ids=FIT_IDS+DEV_IDS; _,series=load_series(ids); pipe=load_pipe(); shared=fit_shared(pipe,series,'mirror',shared_updates); pooled=fit_shared(pipe,series,'pooled',shared_updates,105902)
    (RUNS/'dev').mkdir(parents=True,exist_ok=True); torch.save(shared,RUNS/'dev/shared.pt'); torch.save(pooled,RUNS/'dev/pooled.pt')
    hook=OutputLoRA(pipe,shared['A'],shared['B'],False); models={}; chosen={}
    for method in ('mirror','diag','full','independent'):
        for updates in (10,30,60):
            rows=[]; residual_rows=[]; elapsed=0.; latency=[]
            for sid in DEV_IDS:
                x=series[sid]; w=windows(x); code,sec=adapt(pipe,shared['A'],shared['B'],w['support'],method,updates,105910+updates+int(sid[1:]));elapsed+=sec
                r=trend_residual(pipe,hook,method,code,w['support'],106000+int(sid[1:])) if method=='mirror' else 0.
                ctx,y,o=w['query'][0]; t=time.perf_counter(); pred=predict(pipe,hook,method,code,ctx,106100+int(sid[1:]));latency.append(time.perf_counter()-t)
                den=mase_den(x); rows.append(row_metrics(pred,y,den)); residual_rows.append({'base':mase(pred,y,den),'resid':mase(apply_resid(pred,r),y,den),'r':r})
            key=f'{method}_u{updates}'; models[key]={'metrics':summarize(rows),'fit_seconds':elapsed,'query_p50_s':float(np.median(latency)),'query_p95_s':float(np.quantile(latency,.95))}
            if method=='mirror': models[key]['with_resid_all_mase']=float(np.mean([z['resid'] for z in residual_rows]))
            if method=='mirror': models[key]['residual_improvement_fraction'] = float(np.mean([z['resid']<z['base'] for z in residual_rows]))
        vals={u:models[f'{method}_u{u}']['metrics']['mase'] for u in (10,30,60)}; chosen[method]=min(vals,key=vals.get)
    # Native and pooled controls at one frozen deterministic sample seed per series.
    native=[]; pooled_rows=[]; persist=[]; seasonal=[]; resid_dev=[]
    pa=OutputLoRA(pipe,pooled['A'],pooled['B'],False); pcode=pooled['fit_codes']['shared']
    for sid in DEV_IDS:
        pa.set('inactive',None); hook.set('inactive',None)
        x=series[sid]; w=windows(x); ctx,y,_=w['query'][0]; den=mase_den(x)
        native.append(row_metrics(predict(pipe,hook,'inactive',None,ctx,106200+int(sid[1:])),y,den))
        hook.set('inactive',None); pa.set('pooled',pcode)
        pooled_rows.append(row_metrics(predict(pipe,pa,'pooled',pcode,ctx,106200+int(sid[1:])),y,den))
        persist.append(float(np.mean(np.abs(np.tile(ctx[-1],H)-y))/den)); seasonal.append(float(np.mean(np.abs(np.tile(ctx[-12:],1)-y)/den)) if len(ctx)>=12 else float('nan'))
        # Support-only held-fold validation; ensure the pooled control hook is inactive.
        pa.set('inactive',None); hook.set('inactive',None)
        code,_=adapt(pipe,shared['A'],shared['B'],w['support'][:3],'mirror',chosen['mirror'],106300+int(sid[1:]))
        rr=trend_residual(pipe,hook,'mirror',code,w['support'][:3],106400+int(sid[1:])); vctx,vy,_=w['support'][3]; bp=predict(pipe,hook,'mirror',code,vctx,106500+int(sid[1:])); base=mase(bp,vy,den); corrected=mase(apply_resid(bp,rr),vy,den); resid_dev.append({'series_id':sid,'r':rr,'base':base,'corrected':corrected,'relative_gain':(base-corrected)/max(base,1e-8)})
    pa.set('inactive',None)
    thresholds=[None,0.,.01,.025,.05,.1]
    def apply_threshold(z,th): return th is not None and z['relative_gain']>=th
    eligible=[th for th in thresholds if sum(apply_threshold(z,th) for z in resid_dev)/len(resid_dev)<=.25]
    threshold=min(eligible,key=lambda th:(float(np.mean([z['corrected'] if apply_threshold(z,th) else z['base'] for z in resid_dev])),-(float('inf') if th is None else th)))
    residual_selected=[z['series_id'] for z in resid_dev if apply_threshold(z,threshold)]
    out={'fit_series':FIT_IDS,'development_series':DEV_IDS,'models':models,'selected_updates':chosen,'residual_min_relative_gain':threshold,'residual_selected_ids':residual_selected,'residual_allocation_fraction':len(residual_selected)/len(resid_dev),'native':summarize(native),'pooled_lora':summarize(pooled_rows),'persistence_mase':float(np.mean(persist)),'seasonal_naive_mase':float(np.mean(seasonal)),'shared_fit_seconds':shared['wall_seconds'],'pooled_fit_seconds':pooled['wall_seconds'],'dataset_sha256':sha(download_data()),'model_weights_sha256':sha(SNAP/'model.safetensors'),'support_fold_residual_mase':{'base':float(np.mean([x['base'] for x in resid_dev])),'corrected_all':float(np.mean([x['corrected'] for x in resid_dev]))}}
    out['selection_locked']=True; (RUNS/'dev/development.json').write_text(json.dumps(out,indent=2)+'\n')
    selection={'selected_updates':chosen,'residual_min_relative_gain':threshold,'shared_updates':shared_updates,'residual_fraction_gate':.25,'selected_before_audit_access':True,'hook_isolation':'all unused adapters explicitly inactive during each forecast','selection_criterion':'minimum support-fold MASE among thresholds whose dev residual allocation is <=25%; null threshold disables residual','development_residual_allocation_fraction':len(residual_selected)/len(resid_dev),'development_residual_selected_ids':residual_selected}
    (RUNS/'dev/selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    hook.close(); pa.close(); print(json.dumps(out,indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['dev','audit'],required=True); ap.add_argument('--shared-updates',type=int,default=30); a=ap.parse_args()
    if a.phase=='dev': run_dev(a.shared_updates)
    else: run_audit()


def serialize_state(method, codes, residuals, selected_residual_ids, common_bytes):
    out=RUNS/'audit'/'payloads'/method; out.mkdir(parents=True,exist_ok=True)
    meta={'experiment_id':'MA-1059','method':method,'series_ids':AUDIT_IDS,'model_revision':MODEL_REV,'residual_series_ids':selected_residual_ids if method=='mirror_trend' else []}
    (out/'method.json').write_text(json.dumps(meta,indent=2)+'\n')
    paid=[out/'method.json']
    if method=='pooled_lora':
        q=out/'global_angle.f32le'; q.write_bytes(codes['shared'].detach().cpu().numpy().astype('<f4').tobytes()); paid.append(q)
    if method in ('mirror','mirror_trend','diag','full','independent'):
        if method=='independent':
            arr=np.stack([np.concatenate([codes[s][0].numpy().reshape(-1),codes[s][1].numpy().reshape(-1)]) for s in AUDIT_IDS]).astype('<f4')
        else:
            arr=np.stack([codes[s].numpy().reshape(-1) for s in AUDIT_IDS]).astype('<f4')
        q=out/'series_codes.f32le'; q.write_bytes(arr.tobytes()); paid.append(q)
    if method=='mirror_trend' and selected_residual_ids:
        arr=np.asarray([residuals[s] for s in selected_residual_ids],dtype='<f4')
        q=out/'trend_residuals.f32le'; q.write_bytes(arr.tobytes()); paid.append(q)
    if method in ('mirror','mirror_trend','diag','full'):
        bdir=RUNS/'audit'/'payloads'/'shared_basis'; bdir.mkdir(parents=True,exist_ok=True)
        for nm,t in [('A',codes['_A']),('B',codes['_B'])]:
            q=bdir/f'{nm}.f32le'; q.write_bytes(t.detach().cpu().numpy().astype('<f4').tobytes())
        paid.extend([bdir/'A.f32le',bdir/'B.f32le'])
    if method=='pooled_lora':
        bdir=RUNS/'audit'/'payloads'/'pooled_basis'; bdir.mkdir(parents=True,exist_ok=True)
        for nm,t in [('A',codes['_A']),('B',codes['_B'])]:
            q=bdir/f'{nm}.f32le'; q.write_bytes(t.detach().cpu().numpy().astype('<f4').tobytes())
        paid.extend([bdir/'A.f32le',bdir/'B.f32le'])
    method_bytes=sum(q.stat().st_size for q in paid)
    return {'method_state_bytes':method_bytes,'complete_payload_bytes':(0 if method in ('persistence','seasonal_naive') else common_bytes)+method_bytes,'files':{str(q.relative_to(RUNS/'audit'/'payloads')):q.stat().st_size for q in paid}}

def run_audit():
    selection=json.loads((RUNS/'dev/selection.json').read_text())
    if not selection.get('selected_before_audit_access'): raise ValueError('development choices are not frozen')
    path,series=load_series(AUDIT_IDS)
    pipe=load_pipe()
    if sha(SNAP/'model.safetensors')!='eed458758bd8165d80f496a90bcd2cfed9f1bf7d7b08677acd64e0f7d72bdcf2': raise ValueError('model weights hash mismatch')
    shared=torch.load(RUNS/'dev/shared.pt',map_location='cpu',weights_only=False); pooled=torch.load(RUNS/'dev/pooled.pt',map_location='cpu',weights_only=False)
    shared_hook=OutputLoRA(pipe,shared['A'],shared['B'],False); pooled_hook=OutputLoRA(pipe,pooled['A'],pooled['B'],False)
    selected=selection['selected_updates']; threshold=selection['residual_min_relative_gain']; codes={m:{} for m in ('mirror','diag','full','independent')}; residuals={}; residual_ids=[]; fit_seconds={m:0. for m in codes}; residual_selector_seconds=0.
    for sid in AUDIT_IDS:
        support=windows(series[sid])['support']
        for mode in codes:
            code,elapsed=adapt(pipe,shared['A'],shared['B'],support,mode,int(selected[mode]),108000+int(sid[1:])+{'mirror':1,'diag':2,'full':3,'independent':4}[mode])
            codes[mode][sid]=code; fit_seconds[mode]+=elapsed
        pooled_hook.set('inactive',None); shared_hook.set('inactive',None); selector_t=time.perf_counter()
        if threshold is not None:
            cv_code,_=adapt(pipe,shared['A'],shared['B'],support[:3],'mirror',int(selected['mirror']),108510+int(sid[1:]))
            r_cv=trend_residual(pipe,shared_hook,'mirror',cv_code,support[:3],108520+int(sid[1:])); vctx,vy,_=support[3]
            bp=predict(pipe,shared_hook,'mirror',cv_code,vctx,108530+int(sid[1:])); den=mase_den(series[sid]); base=mase(bp,vy,den); corrected=mase(apply_resid(bp,r_cv),vy,den); gain=(base-corrected)/max(base,1e-8)
            if gain>=float(threshold):
                residual_ids.append(sid); residuals[sid]=trend_residual(pipe,shared_hook,'mirror',codes['mirror'][sid],support,108540+int(sid[1:]))
        residual_selector_seconds+=time.perf_counter()-selector_t
    codes['mirror_trend']=codes['mirror']
    # Build actual serialized states; codes include all required series addresses and metadata.
    common_files=['model.safetensors','config.json','generation_config.json']; model_bytes=sum((SNAP/f).stat().st_size for f in common_files)
    common_manifest={'revision':MODEL_REV,'files':{f:{'bytes':(SNAP/f).stat().st_size,'sha256':sha(SNAP/f)} for f in common_files},'complete_model_bytes':model_bytes}
    pd=RUNS/'audit'/'payloads'; pd.mkdir(parents=True,exist_ok=True); (pd/'common_model_manifest.json').write_text(json.dumps(common_manifest,indent=2)+'\n')
    payloads={}
    payloads['native']=serialize_state('native',{}, {}, [],model_bytes)
    payloads['pooled_lora']=serialize_state('pooled_lora',{'_A':pooled['A'],'_B':pooled['B'],'shared':pooled['fit_codes']['shared']},{},[],model_bytes)
    for m in ('mirror','diag','full','independent'):
        codes[m]['_A']=shared['A']; codes[m]['_B']=shared['B']
        payloads[m]=serialize_state(m,codes[m],{},[],model_bytes)
    codes['mirror_trend']['_A']=shared['A']; codes['mirror_trend']['_B']=shared['B']
    payloads['mirror_trend']=serialize_state('mirror_trend',codes['mirror_trend'],residuals,residual_ids,model_bytes)
    payloads['persistence']=serialize_state('persistence',{}, {}, [],model_bytes)
    payloads['seasonal_naive']=serialize_state('seasonal_naive',{}, {}, [],model_bytes)
    names=('native','pooled_lora','mirror','diag','full','independent','mirror_trend','persistence','seasonal_naive')
    results={'experiment_id':'MA-1059','audit_series_ids':AUDIT_IDS,'seeds':[105903,105904],'data':{'archive_sha256':sha(path),'series_lengths':{s:len(series[s]) for s in AUDIT_IDS}},'selected_updates':selected,'residual_min_relative_gain':threshold,'residual_series_ids':residual_ids,'residual_fraction':len(residual_ids)/len(AUDIT_IDS),'methods':{},'payloads':payloads,'compute':{'support_fit_wall_seconds':fit_seconds,'residual_selector_wall_seconds':residual_selector_seconds,'query_wall_seconds':{}}}
    outputs=RUNS/'audit'; outputs.mkdir(parents=True,exist_ok=True)
    for seed in (105903,105904):
        results['methods'][str(seed)]={}; query_times={m:[] for m in names}
        for method in names:
            preds=[]; targets=[]; ids=[]; origins=[]; metric_rows=[]
            for sid in AUDIT_IDS:
                x=series[sid]; w=windows(x); ctx,y,o=w['query'][0]; den=mase_den(x)
                pooled_hook.set('inactive',None); shared_hook.set('inactive',None)
                if method=='native': mode='inactive'; code=None; hook=shared_hook
                elif method=='pooled_lora': mode='pooled'; code=pooled['fit_codes']['shared']; hook=pooled_hook
                elif method=='mirror_trend': mode='mirror'; code=codes['mirror'][sid]; hook=shared_hook
                elif method in codes: mode=method; code=codes[method][sid]; hook=shared_hook
                elif method in ('persistence','seasonal_naive'): hook=None
                else: raise ValueError(method)
                if method=='pooled_lora': shared_hook.set('inactive',None); pooled_hook.set('pooled',code)
                elif method not in ('persistence','seasonal_naive'): pooled_hook.set('inactive',None); shared_hook.set(mode,code)
                t=time.perf_counter()
                if method=='persistence': pred=np.tile(ctx[-1],H)[None,:].astype(np.float32)
                elif method=='seasonal_naive': pred=np.tile(ctx[-12:],1)[None,:].astype(np.float32)
                else:
                    pred=predict(pipe,hook,mode,code,ctx,seed+int(sid[1:])*31)
                    if method=='mirror_trend' and sid in residual_ids: pred=apply_resid(pred,residuals[sid])
                query_times[method].append(time.perf_counter()-t)
                metric_rows.append({'series_id':sid,**row_metrics(pred,y,den)})
                preds.append(pred);targets.append(y);ids.append(sid);origins.append(o)
            results['methods'][str(seed)][method]={'macro_mase':float(np.mean([r['mase'] for r in metric_rows])),'macro_crps':float(np.mean([r['crps'] for r in metric_rows])),'worst_quartile_mase':float(np.quantile([r['mase'] for r in metric_rows],.75)),'per_series':metric_rows,'complete_payload_bytes':payloads[method]['complete_payload_bytes']}
            np.savez_compressed(outputs/f'{method}_seed{seed}.npz',samples=np.stack(preds),targets=np.stack(targets),series_ids=np.array(ids),origins=np.array(origins))
        results['compute']['query_wall_seconds'][str(seed)]={m:{'p50':float(np.median(t)),'p95':float(np.quantile(t,.95))} for m,t in query_times.items()}
    (outputs/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    shared_hook.close();pooled_hook.close();print(json.dumps({'status':'audit_complete','residual_fraction':results['residual_fraction'],'methods':{s:{m:{'MASE':v['macro_mase'],'CRPS':v['macro_crps']} for m,v in rows.items()} for s,rows in results['methods'].items()},'payloads':{m:v['complete_payload_bytes'] for m,v in payloads.items()}},indent=2))

if __name__=='__main__': main()
