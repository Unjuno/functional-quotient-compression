#!/usr/bin/env python3
"""MA-1064 frozen-base DHE private fallback experiment."""
import argparse, copy, hashlib, importlib.util, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'; ARCHIVE=Path('/tmp/ma1063-ml-1m.zip')
P=json.loads((ROOT/'PROTOCOL.json').read_text()); M=P['inherited_dhe']; CUTS=P['frequency_cutoff_candidates']
BASE_DIR=ROOT.parent/'ma-1063-dhe-id-view'; BASE_SRC=BASE_DIR/'source'/'train.py'
spec=importlib.util.spec_from_file_location('ma1063_base',BASE_SRC); B=importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
METHODS=['dhe_shared','id_residual','private_replace','id_mirror','mirror_residual','full_tables']
ADAPTERS=['id_residual','private_replace','id_mirror','mirror_residual']
MAXU,MAXI=6040,3952; EPOCHS=[1,2,3]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def legacy_audit(u,i): return int(hashlib.sha256(f'{u}:{i}'.encode()).hexdigest(),16)%10==9
def new_bucket(u,i): return int(hashlib.sha256(f'MA-1064:{u}:{i}'.encode()).hexdigest(),16)%10
def partition_bucket(u,i): return None if legacy_audit(u,i) else new_bucket(u,i)
def read_rows(include_audit=False):
    B.verify_archive(); train=[]; dev=[]; audit=[]; excluded=0
    for line in B.raw_lines('ml-1m/ratings.dat'):
        us,is_,rs=line.split('::')[:3]; u=int(us); i=int(is_)
        q=partition_bucket(u,i)
        if q is None: excluded+=1; continue
        if q==9 and not include_audit: continue
        row=(u,i,int(rs)); (train if q<8 else dev if q==8 else audit).append(row)
    tr=np.asarray(train,dtype=np.int64); dv=np.asarray(dev,dtype=np.int64); au=np.asarray(audit,dtype=np.int64)
    pairs=lambda a:set((int(u),int(i)) for u,i,_ in a)
    assert not(pairs(tr)&pairs(dv) or pairs(tr)&pairs(au) or pairs(dv)&pairs(au))
    return tr,dv,au,excluded
def base_rows(rows):
    a=np.zeros((len(rows),6),dtype=np.float32); a[:,0]=rows[:,0]; a[:,1]=rows[:,1]; a[:,4]=rows[:,2]; a[:,5]=(rows[:,2]>=4)
    return a
def id_frequencies(rows): return np.bincount(rows[:,0],minlength=MAXU+1),np.bincount(rows[:,1],minlength=MAXI+1)
def selected_ids(rows,cut):
    uf,if_=id_frequencies(rows); return [np.flatnonzero((uf>0)&(uf<=cut)).astype(int).tolist(),np.flatnonzero((if_>0)&(if_<=cut)).astype(int).tolist()]
def dense_hash_tables():
    hu=np.zeros((MAXU+1,B.K),np.float32); hi=np.zeros((MAXI+1,B.K),np.float32)
    hu[1:]=B.dense_hash(np.arange(1,MAXU+1,dtype=np.int64),0); hi[1:]=B.dense_hash(np.arange(1,MAXI+1,dtype=np.int64),1000000)
    return torch.from_numpy(hu),torch.from_numpy(hi)
class BaseDHE(nn.Module):
    def __init__(self):
        super().__init__(); self.decoder=B.Decoder(); self.head=nn.Linear(32,1); self.hash_u=None; self.hash_i=None
    def embed(self,ids,role): return self.decoder((self.hash_u if role==0 else self.hash_i)[ids])
    def forward(self,u,i): return self.head(self.embed(u,0)*self.embed(i,1)).squeeze(-1)
def train_base(rows,seed):
    torch.manual_seed(seed); np.random.seed(seed); start=time.perf_counter(); model=BaseDHE(); model.hash_u,model.hash_i=dense_hash_tables()
    u=torch.from_numpy(rows[:,0].astype(np.int64)); i=torch.from_numpy(rows[:,1].astype(np.int64)); y=torch.from_numpy((rows[:,2]>=4).astype(np.float32)); opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-6); updates=0
    for ep in range(1,4):
        model.train(); order=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+ep*313))
        for s in range(0,len(y),4096):
            ix=order[s:s+4096]; opt.zero_grad(set_to_none=True); logits=model(u[ix],i[ix]); loss=F.binary_cross_entropy_with_logits(logits,y[ix]); loss.backward(); opt.step(); updates+=1
    model.eval(); return model,{'wall_seconds':time.perf_counter()-start,'updates':updates,'training_examples':len(rows)*3}
def load_base(state):
    model=BaseDHE(); model.load_state_dict(state); model.hash_u,model.hash_i=dense_hash_tables(); model.eval(); return model
class PrivateView(nn.Module):
    def __init__(self,base,method,ids):
        super().__init__(); self.base=base; self.method=method; self.selected_ids=ids
        for p in self.base.parameters(): p.requires_grad_(False)
        self.slot_maps=[]; self.codes=nn.ParameterList()
        for role,(selected,n) in enumerate(zip(ids,(MAXU,MAXI))):
            lut=torch.full((n+1,),-1,dtype=torch.long)
            if selected: lut[torch.tensor(selected,dtype=torch.long)]=torch.arange(len(selected),dtype=torch.long)
            self.slot_maps.append(lut)
            if method=='id_residual': init=torch.zeros((len(selected),32))
            elif method=='private_replace':
                with torch.no_grad(): init=base.embed(torch.tensor(selected,dtype=torch.long),role).detach().clone() if selected else torch.zeros((0,32))
            elif method=='id_mirror': init=torch.zeros((len(selected),16))
            elif method=='mirror_residual': init=torch.zeros((len(selected),48))
            else: init=torch.zeros((0,0))
            self.codes.append(nn.Parameter(init))
        self.base.eval()
    def train(self,mode=True):
        super().train(mode); self.base.eval(); return self
    def _embed(self,ids,role):
        with torch.no_grad(): x=self.base.embed(ids,role)
        slot=self.slot_maps[role][ids]; active=slot>=0; ix=slot.clamp_min(0); code=self.codes[role]
        if code.shape[1]==0 or len(code)==0: return x
        if self.method=='id_residual': return x+code[ix]*active.unsqueeze(1)
        if self.method=='private_replace': return torch.where(active.unsqueeze(1),code[ix],x)
        if self.method in ('id_mirror','mirror_residual'):
            angles=code[ix,:16] if self.method=='mirror_residual' else code[ix]
            p=x.reshape(-1,16,2); c=torch.cos(angles); s=torch.sin(angles)
            rotated=torch.stack((c*p[:,:,0]-s*p[:,:,1],s*p[:,:,0]+c*p[:,:,1]),dim=-1).reshape(-1,32)
            x=torch.where(active.unsqueeze(1),rotated,x)
            if self.method=='mirror_residual': x=x+code[ix,16:48]*active.unsqueeze(1)
        return x
    def forward(self,u,i):
        x=self._embed(u,0); y=self._embed(i,1)
        return self.base.head(x*y).squeeze(-1)
class FullTables(nn.Module):
    def __init__(self):
        super().__init__(); self.u=nn.Embedding(MAXU+1,32); self.i=nn.Embedding(MAXI+1,32); self.head=nn.Linear(32,1)
        nn.init.normal_(self.u.weight,std=.02); nn.init.normal_(self.i.weight,std=.02)
    def forward(self,u,i): return self.head(self.u(u)*self.i(i)).squeeze(-1)

def metric(logits,y): return B.metric(np.asarray(logits),np.asarray(y))
def evaluate(model,rows,trainrows):
    u=torch.from_numpy(rows[:,0].astype(np.int64)); i=torch.from_numpy(rows[:,1].astype(np.int64)); y=(rows[:,2]>=4).astype(np.float32)
    model.eval(); outs=[]
    with torch.no_grad():
        for s in range(0,len(y),4096): outs.extend(model(u[s:s+4096],i[s:s+4096]).cpu().numpy().tolist())
    pred=np.asarray(outs,dtype=np.float32); uf,if_=id_frequencies(trainrows); uc=uf[rows[:,0]]; ic=if_[rows[:,1]]; slices={}
    for name,mask in [('user_freq_le5',uc<=5),('item_freq_le5',ic<=5),('user_freq_0',uc==0),('item_freq_0',ic==0)]:
        if int(mask.sum())>=100: slices[name]={'n':int(mask.sum()),**metric(pred[mask],y[mask])}
        else: slices[name]={'n':int(mask.sum()),'auc':None,'logloss':None}
    both=np.array([slices['user_freq_le5']['auc'],slices['item_freq_le5']['auc']],dtype=object)
    macro=float(np.mean([x for x in both if x is not None]))
    return {'overall':metric(pred,y),'slices':slices,'macro_rare_auc':macro},pred
def train_adapter(method,cut,tr,dv,base,seed,max_epochs=3):
    model=PrivateView(copy.deepcopy(base),method,selected_ids(tr,cut)); train_u=torch.from_numpy(tr[:,0].astype(np.int64)); train_i=torch.from_numpy(tr[:,1].astype(np.int64)); y=torch.from_numpy((tr[:,2]>=4).astype(np.float32))
    dvfreq=id_frequencies(tr); opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=1e-6)
    hist={}; start=time.perf_counter(); updates=0
    for ep in range(1,max_epochs+1):
        model.train(); gen=torch.Generator().manual_seed(seed+ep*313); order=torch.randperm(len(y),generator=gen)
        for s in range(0,len(y),4096):
            ix=order[s:s+4096]; opt.zero_grad(set_to_none=True); logits=model(train_u[ix],train_i[ix]); loss=F.binary_cross_entropy_with_logits(logits,y[ix]); loss.backward(); opt.step(); updates+=1
        if dv is not None: hist[ep]=evaluate(model,dv,tr)[0]
    return model,hist,{'wall_seconds':time.perf_counter()-start,'updates':updates,'training_examples':len(tr)*max_epochs}
def train_full(tr,dv,seed,max_epochs=3):
    torch.manual_seed(seed); model=FullTables(); u=torch.from_numpy(tr[:,0].astype(np.int64)); i=torch.from_numpy(tr[:,1].astype(np.int64)); y=torch.from_numpy((tr[:,2]>=4).astype(np.float32)); opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-6); hist={}; start=time.perf_counter(); updates=0
    for ep in range(1,max_epochs+1):
        model.train(); order=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+ep*313))
        for s in range(0,len(y),4096):
            ix=order[s:s+4096]; opt.zero_grad(set_to_none=True); logits=model(u[ix],i[ix]); loss=F.binary_cross_entropy_with_logits(logits,y[ix]); loss.backward(); opt.step(); updates+=1
        if dv is not None: hist[ep]=evaluate(model,dv,tr)[0]
    return model,hist,{'wall_seconds':time.perf_counter()-start,'updates':updates,'training_examples':len(tr)*max_epochs}
def choose_epochs(hist,seedlist):
    return min(EPOCHS,key=lambda e:np.mean([hist[str(s)][str(e)]['overall']['logloss'] for s in seedlist]))
def run_dev():
    tr,dv,_,excluded=read_rows(False); devseeds=P['development']['seeds']; run=RUNS/'development'; run.mkdir(parents=True,exist_ok=True); details={}; base_info={}
    for seed in devseeds:
        base,comp=train_base(tr,seed); base_info[str(seed)]={'compute':comp,'metrics':evaluate(base,dv,tr)[0]}
        torch.save({'state_dict':base.state_dict()},run/f'base_{seed}.pt')
    for method in ADAPTERS:
        details[method]={}
        for cut in CUTS:
            details[method][str(cut)]={}
            for seed in devseeds:
                obj=torch.load(run/f'base_{seed}.pt',map_location='cpu',weights_only=False); base=load_base(obj['state_dict'])
                model,hist,comp=train_adapter(method,cut,tr,dv,base,seed)
                details[method][str(cut)][str(seed)]={'metrics':hist,'compute':comp,'selected_id_counts':[len(x) for x in model.selected_ids]}
    full={}
    for seed in devseeds:
        model,hist,comp=train_full(tr,dv,seed); full[str(seed)]={'metrics':hist,'compute':comp}
    # Select epochs per family/cut by mean dev logloss.
    epochs={m:{c:min(EPOCHS,key=lambda e:np.mean([details[m][c][str(s)]['metrics'][e]['overall']['logloss'] for s in devseeds])) for c in details[m]} for m in ADAPTERS}
    full_epoch=min(EPOCHS,key=lambda e:np.mean([full[str(s)]['metrics'][e]['overall']['logloss'] for s in devseeds]))
    # Select the Mirror+residual cutoff by dev rare-slice AUC, subject to overall logloss <= baseline + 0.002.
    candidates=[]; baseline_ll=np.mean([base_info[str(s)]['metrics']['overall']['logloss'] for s in devseeds])
    for cut in CUTS:
        ep=epochs['mirror_residual'][str(cut)]; vals=[details['mirror_residual'][str(cut)][str(s)]['metrics'][ep] for s in devseeds]
        ll=float(np.mean([x['overall']['logloss'] for x in vals])); rare=float(np.mean([x['macro_rare_auc'] for x in vals])); nids=np.mean([sum(details['mirror_residual'][str(cut)][str(s)]['selected_id_counts']) for s in devseeds])
        candidates.append({'cutoff':cut,'epoch':ep,'mean_macro_rare_auc':rare,'mean_logloss':ll,'mean_selected_ids':float(nids),'eligible':ll<=baseline_ll+.002})
    eligible=[x for x in candidates if x['eligible']]; pool=eligible or candidates
    selected=max(pool,key=lambda x:(x['mean_macro_rare_auc'],-x['mean_selected_ids']))
    out={'experiment_id':'MA-1064','archive_sha256':sha(ARCHIVE),'legacy_MA1063_audit_rows_excluded':excluded,'train_rows':len(tr),'dev_rows':len(dv),'audit_opened':False,'development_seeds':devseeds,'base':base_info,'adapters':details,'adapter_selected_epochs':epochs,'full_tables':full,'full_selected_epoch':full_epoch,'mirror_residual_cutoff_candidates':candidates,'selected_cutoff':selected['cutoff'],'selected_mirror_residual_epoch':selected['epoch'],'selected_full_epoch':full_epoch}
    (run/'development.json').write_text(json.dumps(out,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
    selection={'experiment_id':'MA-1064','selected_cutoff':selected['cutoff'],'adapter_selected_epochs':epochs,'full_selected_epoch':full_epoch,'cutoff_rule':'highest mean development macro rare AUC among Mirror+residual cutoffs with overall logloss <= shared DHE +0.002; ties fewer selected IDs','audit_opened':False}
    (RUNS/'DEVELOPMENT_SELECTION.json').write_text(json.dumps(selection,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n'); print(json.dumps({'status':'development_complete','train_rows':len(tr),'dev_rows':len(dv),'excluded_legacy_audit_rows':excluded,'selected_cutoff':selected['cutoff'],'selected_epochs':epochs,'full_epoch':full_epoch},indent=2))
def load_freeze():
    f=json.loads((ROOT/'FRESH_FREEZE.json').read_text())
    checks={'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'runner_sha256':sha(ROOT/'source/train.py'),'base_runner_sha256':sha(BASE_SRC),'selection_sha256':sha(RUNS/'DEVELOPMENT_SELECTION.json'),'amendments_sha256':sha(ROOT/'AMENDMENTS.md'),'failed_attempt_sha256':sha(RUNS/'FAILED_ATTEMPT_DEV01.json')}
    if any(f[k]!=v for k,v in checks.items()): raise ValueError('FRESH_FREEZE hash mismatch')
    return f
def latency(model,rows):
    u=torch.from_numpy(rows[:4096,0].astype(np.int64)); i=torch.from_numpy(rows[:4096,1].astype(np.int64)); n=min(1024,len(rows)); vals=[]; model.eval()
    with torch.no_grad():
        for _ in range(8): model(u[:n],i[:n])
        for _ in range(40):
            t=time.perf_counter(); model(u[:n],i[:n]); vals.append(time.perf_counter()-t)
    return {'batch_size':n,'p50_ms':float(np.quantile(vals,.5)*1000),'p95_ms':float(np.quantile(vals,.95)*1000),'qps':float(n/np.mean(vals))}
def run_audit():
    freeze=load_freeze(); sel=json.loads((RUNS/'DEVELOPMENT_SELECTION.json').read_text()); tr,_,au,excluded=read_rows(True); audit=au; seeds=P['fresh']['seeds']; cut=int(sel['selected_cutoff']); d=RUNS/'audit'; d.mkdir(parents=True,exist_ok=True); payloadmeta=d/'payload_metadata'; payloadmeta.mkdir(exist_ok=True); results={}
    for seed in seeds:
        base,basecomp=train_base(tr,seed)
        # A common, frozen backbone is used by every method for this seed.
        configs=[('dhe_shared',None),('id_residual',cut),('private_replace',cut),('id_mirror',cut),('mirror_residual',cut),('full_tables',None)]
        for method,ct in configs:
            if method=='dhe_shared': model=base; comp=basecomp; ids=[[],[]]; ep=3
            elif method=='full_tables':
                ep=int(sel['full_selected_epoch']); model,_,comp=train_full(tr,None,seed,max_epochs=ep); ids=[[],[]]
            else:
                ep=int(sel['adapter_selected_epochs'][method][str(cut)]); model,_,comp=train_adapter(method,cut,tr,None,base,seed,max_epochs=ep); ids=model.selected_ids
            # Save a complete inference checkpoint plus external metadata, including all per-ID addresses.
            ck=d/f'{method}_{seed}.pt'; torch.save({'method':method,'state_dict':model.state_dict(),'cutoff':ct},ck)
            meta={'experiment_id':'MA-1064','method':method,'seed':seed,'cutoff':ct,'selected_user_ids':ids[0],'selected_item_ids':ids[1],'hash':B.M['dense_hash'],'embedding_dim':32,'decoder':M['decoder'],'head':M['head'],'epochs':ep,'model_version':'MA-1063 scoped DHE'}
            mp=payloadmeta/f'{method}_{seed}.json'; mp.write_text(json.dumps(meta,sort_keys=True,separators=(',',':'))+'\n')
            # Reload using exactly the serialized model state; no audit labels/IDs are persisted.
            obj=torch.load(ck,map_location='cpu',weights_only=False)
            if method=='full_tables': replay=FullTables(); replay.load_state_dict(obj['state_dict'])
            elif method=='dhe_shared': replay=load_base(obj['state_dict'])
            else:
                replaybase=load_base({k.removeprefix('base.'):v for k,v in obj['state_dict'].items() if k.startswith('base.')})
                replay=PrivateView(replaybase,method,ids); replay.load_state_dict(obj['state_dict'])
            metrics,pred=evaluate(replay,audit,tr); np.savez_compressed(d/f'{method}_{seed}.npz',logits=pred)
            row={'metrics':metrics,'payload_bytes':ck.stat().st_size+mp.stat().st_size,'checkpoint_bytes':ck.stat().st_size,'metadata_bytes':mp.stat().st_size,'checkpoint_sha256':sha(ck),'metadata_sha256':sha(mp),'logits_sha256':sha(d/f'{method}_{seed}.npz'),'logits_bytes':(d/f'{method}_{seed}.npz').stat().st_size,'latency':latency(replay,audit),'compute':{**comp,'base_wall_seconds':basecomp['wall_seconds'],'selected_epoch':ep},'selected_id_counts':[len(x) for x in ids]}
            results.setdefault(method,{})[str(seed)]=row
    out={'experiment_id':'MA-1064','audit_rows':len(audit),'excluded_MA1063_audit_rows':excluded,'selected_cutoff':cut,'selected_epochs':sel,'methods':results,'audit_labels_or_ids_persisted':False}
    (d/'results.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'status':'audit_complete','results':{m:{s:{'auc':v['metrics']['overall']['auc'],'logloss':v['metrics']['overall']['logloss'],'macro_rare_auc':v['metrics']['macro_rare_auc'],'payload_bytes':v['payload_bytes']} for s,v in z.items()} for m,z in results.items()}},indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['dev','audit'],required=True); a=ap.parse_args(); torch.set_num_threads(1)
    run_dev() if a.phase=='dev' else run_audit()
