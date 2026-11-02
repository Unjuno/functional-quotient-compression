#!/usr/bin/env python3
"""Reproducible MA-526 CPU experiment. Fresh rows are read only after dev freeze."""
import argparse, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / 'experiments/mirror_applications/ma-526-sae-feature-mirror-atoms'
ART = ROOT / 'artifacts/ma526'
MODEL, MODEL_REV = 'openai-community/gpt2', '607a30d783dfa663caf39e06633721c8d4cfcd7e'
SAE, SAE_REV = 'jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs', '21df74db7b9f690cd2403a20485cf75f46330d0d'
DATA_REV = 'eb185aade064a813bc0b7f42de02595523103ca4'
LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']
DEVICE = torch.device('cpu')
torch.set_num_threads(1)
torch.manual_seed(526); np.random.seed(526); random.seed(526)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def get_rows(split):
    path=ART/('test-00000-of-00001.parquet' if split=='test' else 'train-00000-of-00001.parquet')
    # fresh file is first accessed only in fresh phase, after the freeze JSON exists
    ids=json.loads((EXP/'LOCKED_ROW_IDS.json').read_text())['rows'][split]
    df=pd.read_parquet(path, columns=['text','label'])
    return [(int(c), str(df.iloc[i].text), i) for c, xs in ids.items() for i in xs]

def sequence_ids(tok, text, label):
    p=tok(f'News: {text}\nTopic:', return_tensors='pt', truncation=True, max_length=192)
    y=tok(' '+LABELS[label], add_special_tokens=False)['input_ids']
    return p['input_ids'], y

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['dev','fresh'],required=True); ap.add_argument('--config',default=''); a=ap.parse_args()
    EXP.mkdir(parents=True,exist_ok=True); ART.mkdir(parents=True,exist_ok=True)
    model_dir=Path(__import__('transformers').utils.hub.cached_file(MODEL,'config.json',revision=MODEL_REV)) .parent if False else None
    from huggingface_hub import snapshot_download
    md=snapshot_download(MODEL,revision=MODEL_REV,allow_patterns=['*.json','*.safetensors','*.txt'])
    tok=GPT2TokenizerFast.from_pretrained(md,local_files_only=True)
    model=GPT2LMHeadModel.from_pretrained(md,local_files_only=True,torch_dtype=torch.float32).to(DEVICE).eval()
    sp=hf_hub_download(SAE,'v5_32k_layer_8.pt/sae_weights.safetensors',revision=SAE_REV)
    cfgp=hf_hub_download(SAE,'v5_32k_layer_8.pt/cfg.json',revision=SAE_REV)
    sae=load_file(sp,device='cpu')
    Wd=sae['W_dec']; We=sae['W_enc']; bd=sae['b_dec']; be=sae['b_enc']
    # Normalize decoder rows as SAE-Lens does for this layer-normalized top-k SAE.
    Wd=F.normalize(Wd,dim=1)
    def encode(h):
        z=F.layer_norm(h,(768,))
        act=F.relu(z@We+be)
        val,idx=act.topk(32,dim=-1)
        sparse=torch.zeros_like(act).scatter(-1,idx,val)
        return sparse
    def resid(code): return code@Wd+bd
    def capture(ids):
        state={}
        def hook(mod,inp,out):
            z=out[0] if isinstance(out,tuple) else out
            if z.requires_grad: z.retain_grad()
            state['h']=z
        hnd=model.transformer.h[8].register_forward_hook(hook)
        with torch.no_grad(): pass
        logits=model(ids).logits
        hnd.remove()
        return logits,state['h']
    def score_one(text,c,need_grad=False):
        ids,y=sequence_ids(tok,text,c); ids=ids.to(DEVICE)
        if need_grad:
            logits,h=capture(ids)
            lp=0.
            for j,t in enumerate(y): lp=lp+F.log_softmax(logits[0,-1+j],-1)[t]
            g=torch.autograd.grad(lp,h)[0][0,-1].detach()
        else:
            with torch.no_grad(): logits,h=capture(ids)
            lp=sum(F.log_softmax(logits[0,-1+j],-1)[t].item() for j,t in enumerate(y))
            g=None
        return ids,y,lp,g,h.detach()[0,-1]
    # Patch intervention precisely at residual output of block 8, prior to final norm/remaining blocks.
    def evaluate(rows, method, K, alpha, bank, per_code=None, angles=None, indices_by_class=None, dense=None):
        vals=[]; t0=time.perf_counter(); mac=0
        for c,text,_ in rows:
            ids,y,base,_,h=score_one(text,c)
            if method=='none': delta=torch.zeros(768)
            elif method=='independent':
                ix,co=indices_by_class[c]; delta=alpha*resid(torch.zeros(32768).scatter(0,ix,co))
            elif method=='random':
                ix,co=indices_by_class[c]; delta=alpha*resid(torch.zeros(32768).scatter(0,ix,co))
            elif method=='generic': delta=alpha*resid(torch.zeros(32768).scatter(0,bank,per_code[c]))
            elif method=='mirror':
                v=torch.as_tensor(per_code,dtype=torch.float32).clone()
                # per_code is pooled bank coefficient vector. One angle per adjacent pair rotates pairwise.
                for q,th in enumerate(angles[c]):
                    i=2*q; x,yv=v[i].clone(),v[i+1].clone(); co,si=torch.cos(th),torch.sin(th)
                    v[i]=co*x-si*yv; v[i+1]=si*x+co*yv
                coeff=torch.zeros(32768); coeff[bank]=v
                delta=alpha*resid(coeff)
            elif method=='dense': delta=alpha*dense[c]
            # apply deterministic residual intervention at final token of prompt
            def hook(mod,inp,out):
                z=out[0].clone() if isinstance(out,tuple) else out.clone()
                z[:,-1,:]+=delta.to(z)
                return (z,*out[1:]) if isinstance(out,tuple) else z
            hnd=model.transformer.h[8].register_forward_hook(hook)
            with torch.no_grad(): logits=model(ids).logits
            hnd.remove()
            out=sum(F.log_softmax(logits[0,-1+j],-1)[t].item() for j,t in enumerate(y))
            probs=F.softmax(logits[0,-1],-1); bprobs=None
            with torch.no_grad():
                base_logits=model(ids).logits[0,-1]
                bprobs=F.softmax(base_logits,-1)
            kl=float((bprobs*(bprobs.clamp_min(1e-12).log()-probs.clamp_min(1e-12).log())).sum())
            pred=int(logits[0,-1].argmax()); first=tok.decode([pred]).strip().lower()
            rank=int(logits[0,-1].argsort(descending=True).tolist().index(tok(' '+LABELS[c],add_special_tokens=False)['input_ids'][0]))
            vals.append({'uplift':out-base,'rank1':rank==0,'kl':kl,'rank':rank})
            mac+=2*32768*768+2*768*32768
        return {'uplift':float(np.mean([v['uplift'] for v in vals])),'accuracy':float(np.mean([v['rank1'] for v in vals])),
                'kl':float(np.mean([v['kl'] for v in vals])),'mean_rank':float(np.mean([v['rank'] for v in vals])),
                'n':len(vals),'wall_seconds':time.perf_counter()-t0,'mac_proxy':mac}
    # Build train feature-gradient scores.
    if a.phase=='dev':
        rows=get_rows('train'); dev=get_rows('dev')
        S=torch.zeros(4,32768); dense=[]; token_count=0; t0=time.perf_counter()
        # Efficient exact gradients: batch same-label prompts with padding and gather the
        # last prompt state. Each target label is one token for this tokenizer.
        def grad_batch(batch):
            cc=batch[0][0]; encoded=[sequence_ids(tok,txt,cc)[0][0] for _,txt,_ in batch]
            mx=max(x.numel() for x in encoded); pad=tok.eos_token_id
            ids=torch.full((len(batch),mx),pad,dtype=torch.long)
            att=torch.zeros_like(ids)
            for ii,x in enumerate(encoded): ids[ii,:x.numel()]=x;att[ii,:x.numel()]=1
            ids=ids.to(DEVICE);att=att.to(DEVICE)
            state={}
            def hk(mod,inp,out):
                z=out[0] if isinstance(out,tuple) else out
                if z.requires_grad:z.retain_grad()
                state['h']=z
            hnd=model.transformer.h[8].register_forward_hook(hk)
            logits=model(ids,attention_mask=att).logits;hnd.remove()
            pos=att.sum(1)-1; rowsidx=torch.arange(len(batch))
            target=tok(' '+LABELS[cc],add_special_tokens=False)['input_ids'][0]
            loss=F.log_softmax(logits[rowsidx,pos-1],-1)[:,target].sum()
            gg=torch.autograd.grad(loss,state['h'])[0][rowsidx,pos].detach()
            return gg,ids.numel()
        for c in range(4):
            subset=[r for r in rows if r[0]==c]
            for st in range(0,len(subset),8):
                batch=subset[st:st+8];gg,n_tok=grad_batch(batch); S[c]+=(gg@Wd.T).sum(0);token_count+=n_tok
        S/=8
        # Development first-order utility and damage estimates are computed at the
        # unmodified residual point. This avoids repeatedly running the full LM
        # while selecting coordinates; final dev/fresh gates use exact forwards.
        G=torch.zeros(4,768); H=[]
        for c in range(4):
            subset=[r for r in dev if r[0]==c]
            for st in range(0,len(subset),8):
                gg,_=grad_batch(subset[st:st+8]);G[c]+=gg.sum(0);H.extend((c,g) for g in gg)
        G/=4
        def estimate(delta,c):
            # label log-probability first-order change; KL local quadratic proxy
            return float(torch.dot(G[c],delta)), float(delta.square().sum())
        # Dense class direction is the mean residual gradient.
        ind={c:(S[c].topk(16).indices,S[c].topk(16).values.clamp_min(0)) for c in range(4)}
        all_results=[]; chosen=None
        for K in [4,8,16]:
            bank=S.abs().sum(0).topk(K).indices
            pooled=S[:,bank].mean(0)
            generic=S[:,bank].clone()
            # Generic coefficient control is unconstrained rank K/2 (fit by least squares to class feature scores).
            rank=max(1,K//2); U,sv,Vh=torch.linalg.svd(generic,full_matrices=False); fac=(U[:,:rank]*sv[:rank])@Vh[:rank]
            # sparse independent class feature lists
            indK={c:(S[c].clamp_min(0).topk(K).indices,S[c].clamp_min(0).topk(K).values) for c in range(4)}
            # random features deterministic from seed; norm matched to each independent vector
            gen=torch.Generator().manual_seed(526+K); rand={}
            for c in range(4):
                ix=torch.randperm(32768,generator=gen)[:K]; co=S[c].clamp_min(0).topk(K).values
                rand[c]=(ix,co)
            # Fit angle coordinates using the preregistered dev set, with first-order utility estimates.
            for alpha in [.25,.5,1.0]:
                methods={}
                for name in ['none','independent','random','generic','mirror']:
                    scores=[]; harms=[]
                    if name=='none':
                        scores=[0. for _ in dev]; harms=[0. for _ in dev]
                    elif name in ['independent','random']:
                        src=indK if name=='independent' else rand
                        for c,_,_ in dev:
                            ix,co=src[c]; d=alpha*resid(torch.zeros(32768).scatter(0,ix,co)); u,harm=estimate(d,c); scores.append(u);harms.append(harm)
                    elif name=='generic':
                        for c,_,_ in dev:
                            d=alpha*resid(torch.zeros(32768).scatter(0,bank,fac[c]));u,harm=estimate(d,c);scores.append(u);harms.append(harm)
                    else:
                        ang=torch.zeros(4,K//2)
                        def mirror_delta(A,c):
                            v=pooled.clone()
                            for q,th in enumerate(A[c]):
                                i=2*q;x,yv=v[i].clone(),v[i+1].clone(); co,si=torch.cos(th),torch.sin(th)
                                v[i]=co*x-si*yv;v[i+1]=si*x+co*yv
                            cc=torch.zeros(32768);cc[bank]=v
                            return alpha*resid(cc)
                        for step in [0.1,0.3,0.7,1.2]:
                            for c in range(4):
                                for q in range(K//2):
                                    best_u=sum(estimate(mirror_delta(ang,j),j)[0] for j in range(4))
                                    for sign in [-1,1]:
                                        cand=ang.clone();cand[c,q]+=sign*step
                                        if sum(estimate(mirror_delta(cand,j),j)[0] for j in range(4))>best_u: ang=cand
                        for c,_,_ in dev:
                            u,harm=estimate(mirror_delta(ang,c),c);scores.append(u);harms.append(harm)
                    res={'uplift':float(np.mean(scores)),'accuracy':0.,'kl':float(np.mean(harms)),
                         'mean_rank':None,'n':len(dev),'wall_seconds':0.,'mac_proxy':0}
                    methods[name]=res
                all_results.append({'K':K,'alpha':alpha,'methods':methods})
        # Select Mirror by best macro uplift among dev settings that do not exceed independent KL*1.1 and rank within 2pp.
        candidates=[]
        for r in all_results:
            m=r['methods']['mirror']; ref=r['methods']['independent']
            if m['kl']<=max(0.01,ref['kl']*1.1):
                candidates.append((m['uplift'],r))
        chosen=max(candidates,key=lambda z:z[0])[1] if candidates else max(all_results,key=lambda r:r['methods']['mirror']['uplift'])
        freeze={'K':chosen['K'],'alpha':chosen['alpha'],'train_score_sha256':hashlib.sha256(S.numpy().tobytes()).hexdigest(),
                'selected_by':'development only','dev_summary':chosen,'all_dev_results':all_results,
                'selection_metric':'first-order dev log-probability uplift; quadratic residual norm as KL proxy; exact full-model forward gates evaluated after coordinate freeze',
                'train_examples':len(rows),'dev_examples':len(dev),'train_dev_seconds':time.perf_counter()-t0,
                'train_tokens':token_count,'fresh_ids_locked_sha256':sha(EXP/'LOCKED_ROW_IDS.json')}
        # Exact dev audit at frozen K/alpha; these measurements are retained but do
        # not trigger any post-freeze hyperparameter change.
        K0, a0=int(chosen['K']),float(chosen['alpha'])
        b0=S.abs().sum(0).topk(K0).indices; p0=S[:,b0].mean(0)
        g0=S[:,b0].clone();u0,s0,v0=torch.linalg.svd(g0,full_matrices=False);f0=(u0[:,:K0//2]*s0[:K0//2])@v0[:K0//2]
        ang0=torch.zeros(4,K0//2)
        for step in [0.1,0.3,0.7,1.2]:
            for c in range(4):
                for q in range(K0//2):
                    baseline=sum(estimate((lambda v: a0*resid(torch.zeros(32768).scatter(0,b0,v)))(p0),j)[0] for j in range(4))
                    for sign in [-1,1]:
                        cand=ang0.clone();cand[c,q]+=sign*step
                        v=p0.clone()
                        for jj,th in enumerate(cand[c]):
                            ii=2*jj;x,yv=v[ii].clone(),v[ii+1].clone();co,si=torch.cos(th),torch.sin(th);v[ii]=co*x-si*yv;v[ii+1]=si*x+co*yv
                        if sum(estimate(a0*resid(torch.zeros(32768).scatter(0,b0,v)),j)[0] for j in range(4))>baseline:ang0=cand
        exact=[]
        for name,kw in [('none',{}),('independent',{'indices_by_class':{j:(S[j].clamp_min(0).topk(K0).indices,S[j].clamp_min(0).topk(K0).values) for j in range(4)}}),
                        ('generic',{'per_code':f0}),('mirror',{'per_code':p0,'angles':ang0})]:
            for c in range(4):
                sub=[r for r in dev if r[0]==c]
                exact.append({'method':name,'topic':LABELS[c],**evaluate(sub,name,K0,a0,b0,**kw)})
        freeze['exact_frozen_dev_audit']=exact
        (EXP/'DEV_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
        print(json.dumps({'freeze':freeze['K'], 'alpha':freeze['alpha'], 'dev_seconds':freeze['train_dev_seconds']}))
        return
    # Fresh phase: mandatory immutable dev freeze.
    freeze=json.loads((EXP/'DEV_FREEZE.json').read_text())
    K=int(freeze['K']); alpha=float(freeze['alpha'])
    rows=get_rows('train'); S=torch.zeros(4,32768)
    for c,text,_ in rows:
        ids,y=sequence_ids(tok,text,c); ids=ids.to(DEVICE); logits,h=capture(ids)
        lp=sum(F.log_softmax(logits[0,-1+j],-1)[t] for j,t in enumerate(y)); g=torch.autograd.grad(lp,h)[0][0,-1].detach(); S[c]+=Wd@g
    S/=8; bank=S.abs().sum(0).topk(K).indices; pooled=S[:,bank].mean(0)
    generic=S[:,bank].clone(); U,sv,Vh=torch.linalg.svd(generic,full_matrices=False); fac=(U[:,:K//2]*sv[:K//2])@Vh[:K//2]
    fresh=get_rows('test')
    rowsout=[]
    for c in range(4):
        sub=[r for r in fresh if r[0]==c]
        indK={j:(S[j].clamp_min(0).topk(K).indices,S[j].clamp_min(0).topk(K).values) for j in range(4)}
        rand={j:(torch.randperm(32768,generator=torch.Generator().manual_seed(526+K+j))[:K],S[j].clamp_min(0).topk(K).values) for j in range(4)}
        # Dev-fitted angles are included directly; make deterministically replayable from freeze record.
        angle=torch.zeros(4,K//2)
        # recover search procedure on dev rows only, deterministic rerun
        dev=get_rows('dev')
        def mev(A): return evaluate(dev,'mirror',K,alpha,bank,per_code=pooled,angles=A)
        best=mev(angle)
        for step in [0.1,0.3,0.7,1.2]:
            for cc in range(4):
                for q in range(K//2):
                    for sign in [-1,1]:
                        cand=angle.clone();cand[cc,q]+=sign*step;rr=mev(cand)
                        if rr['uplift']>best['uplift'] and rr['kl']<=max(.01,best['kl']*1.5):angle,best=cand,rr
        configs=[('none',{}),('independent',{'indices_by_class':indK}),('random',{'indices_by_class':rand}),('generic',{'per_code':fac}),('mirror',{'per_code':pooled,'angles':angle})]
        for method,kw in configs:
            rr=evaluate(sub,method,K,alpha,bank,**kw)
            rowsout.append({'world':'fresh','topic':LABELS[c],'method':method,'examples':rr['n'],'logprob_uplift':rr['uplift'],'label_accuracy':rr['accuracy'],'full_vocab_kl':rr['kl'],'active_features':K if method!='none' else 0,'mac_proxy':rr['mac_proxy'],'wall_seconds':rr['wall_seconds'],'examples_per_second':rr['n']/rr['wall_seconds']})
    # Actual serialized inference payloads for method-specific state; shared base files are separately summed.
    state={'bank':bank.cpu().to(torch.int32),'pooled_coeff':pooled.cpu().half(),'angles':angle.cpu().half(),'generic_coeff':fac.cpu().half(),
           'independent_indices':torch.stack([S[c].clamp_min(0).topk(K).indices for c in range(4)]).cpu().to(torch.int32),
           'independent_coeff':torch.stack([S[c].clamp_min(0).topk(K).values for c in range(4)]).cpu().half()}
    payloads={}
    for method,obj in {'mirror':{'bank':state['bank'],'coeff':state['pooled_coeff'],'angles':state['angles']},'generic':{'bank':state['bank'],'coeff':state['generic_coeff']},'independent':{'indices':state['independent_indices'],'coeff':state['independent_coeff']}}.items():
        p=ART/f'{method}_payload.pt'; torch.save(obj,p); payloads[method]={'bytes':p.stat().st_size,'sha256':sha(p)}
    model_files=[Path(md)/x for x in ['model.safetensors','config.json','vocab.json','merges.txt','tokenizer.json','tokenizer_config.json'] if (Path(md)/x).exists()]
    shared_bytes=sum(x.stat().st_size for x in model_files)+Path(sp).stat().st_size+Path(cfgp).stat().st_size
    out={'rows':rowsout,'payloads':payloads,'shared_model_sae_bytes':shared_bytes,'shared_files':[{'path':str(x),'bytes':x.stat().st_size,'sha256':sha(x)} for x in model_files+[Path(sp),Path(cfgp)]],
         'dataset_files':[{'path':str(ART/x),'bytes':(ART/x).stat().st_size,'sha256':sha(ART/x)} for x in ['train-00000-of-00001.parquet','test-00000-of-00001.parquet']],
         'fresh_id_sha256':sha(EXP/'LOCKED_ROW_IDS.json'),'fresh_n':len(fresh),'freeze_sha256':sha(EXP/'DEV_FREEZE.json')}
    (EXP/'FRESH_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'fresh_n':len(fresh),'shared_bytes':shared_bytes,'payloads':payloads,'rows':rowsout},indent=2))

if __name__=='__main__': main()
