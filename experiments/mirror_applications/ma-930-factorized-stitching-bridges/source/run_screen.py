"""Frozen real-LM feature-bridge screen for MA-930 amendment 01."""
import csv, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer, GPTNeoForCausalLM

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path('/tmp/ma930-cache'); CACHE.mkdir(exist_ok=True)
torch.set_num_threads(1)
DEVICE='cpu'
MODELS=[
 ('roneneldan/TinyStories-1M','77f1b168e219585646439073245fe87e56b3023e',64),
 ('roneneldan/TinyStories-3M','cfaf26ec85ecdfc1bd7c2638104cce55cb67f894',128),
 ('roneneldan/TinyStories-8M','8612e3b15c66ffa94eaa6ee0de5c96edd2d630af',256),
]
DATA_REV='f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
TRAIN_FILE='data/train-00000-of-00004-2d5a1467fff1081b.parquet'
VAL_FILE='data/validation-00000-of-00001-869c898b519ad725.parquet'
TRAIN_PAIRS=[(0,1),(1,0),(1,2),(2,1)]
ALL_PAIRS=[(i,j) for i in range(3) for j in range(3) if i!=j]
HELDOUT=[(0,2),(2,0)]
LAYERS=[2,4,6,8]
RANK=8
BATCH=256
UPDATES=1000
SEEDS=[93001,93007,93013]

def get_data():
    tr=hf_hub_download('roneneldan/TinyStories',TRAIN_FILE,repo_type='dataset',revision=DATA_REV,cache_dir=str(CACHE))
    va=hf_hub_download('roneneldan/TinyStories',VAL_FILE,repo_type='dataset',revision=DATA_REV,cache_dir=str(CACHE))
    trds=Dataset.from_parquet(tr); vads=Dataset.from_parquet(va)
    return [trds[i]['text'] for i in range(256)], [vads[i]['text'] for i in range(128)], [tr,va]

def tokenize(tokenizer,texts):
    tok=tokenizer(texts,truncation=True,max_length=32,padding='max_length',return_tensors='pt')
    # Exclude empty/one-token documents if any; token windows remain paired across models.
    keep=tok['attention_mask'].sum(1)>=4
    return tok['input_ids'][keep],tok['attention_mask'][keep]

@torch.inference_mode()
def features(models, ids, mask, batch=4):
    all_h=[]
    for model in models:
        hs=[[] for _ in range(9)]
        for st in range(0,len(ids),batch):
            ii=ids[st:st+batch].to(DEVICE); mm=mask[st:st+batch].to(DEVICE)
            out=model(input_ids=ii,attention_mask=mm,output_hidden_states=True,use_cache=False)
            for li,h in enumerate(out.hidden_states): hs[li].append(h.cpu().float())

        all_h.append([torch.cat(v,0) for v in hs])
    return all_h

class Factor(nn.Module):
    def __init__(self,dims):
        super().__init__(); self.A=nn.ParameterList([nn.Parameter(torch.empty(RANK,d)) for d in dims]); self.B=nn.ParameterList([nn.Parameter(torch.empty(d,RANK)) for d in dims]); self.C=nn.ParameterList([nn.Parameter(torch.empty(RANK,RANK)) for _ in LAYERS]); self.bias=nn.ParameterList([nn.Parameter(torch.zeros(d)) for d in dims for _ in LAYERS])
        for p in self.A: nn.init.xavier_uniform_(p)
        for p in self.B: nn.init.xavier_uniform_(p)
        for p in self.C: nn.init.xavier_uniform_(p)
    def forward(self,x,s,t,l):
        z=F.linear(x,self.A[s]); z=F.linear(z,self.C[LAYERS.index(l)]); z=F.linear(z,self.B[t]); return z+self.bias[t*len(LAYERS)+LAYERS.index(l)]

class Independent(nn.Module):
    def __init__(self,dims):
        super().__init__(); self.keys=[(s,t,l) for s,t in ALL_PAIRS for l in LAYERS]; self.W=nn.ParameterList(); self.b=nn.ParameterList(); self.index={k:i for i,k in enumerate(self.keys)}
        for s,t,l in self.keys:
            w=nn.Parameter(torch.empty(dims[t],dims[s])); nn.init.xavier_uniform_(w); self.W.append(w); self.b.append(nn.Parameter(torch.zeros(dims[t])))
    def forward(self,x,s,t,l):
        k=self.index[(s,t,l)]; return F.linear(x,self.W[k],self.b[k])

def flatten(h, mask):
    keep=mask[:,1:].reshape(-1).bool()
    return [z[:,1:,:].reshape(-1,z.shape[-1])[keep] for z in h]  # exclude first and padded positions

def fit(model,train_h,train_mask,dims,seed,factorized):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); data_rng=torch.Generator().manual_seed(seed+501)
    opt=torch.optim.Adam(model.parameters(),lr=.003); t0=time.perf_counter()
    pairs=TRAIN_PAIRS if factorized else ALL_PAIRS
    flat=[flatten(x,train_mask) for x in train_h]
    for _ in range(UPDATES):
        s,t=random.choice(pairs); l=random.choice(LAYERS); si=LAYERS.index(l)
        # hidden tuple index l is after l decoder blocks
        xs=flat[s][l]; ys=flat[t][l]
        idx=torch.randint(0,xs.shape[0],(BATCH,),generator=data_rng)
        pred=model(xs[idx],s,t,l); loss=F.mse_loss(pred,ys[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return time.perf_counter()-t0

def tail_logits(model,h,layer,attention_mask):
    tr=model.transformer; seq=h.shape[1]; pos=torch.arange(seq,device=h.device)
    mask=tr._update_causal_mask(attention_mask,h,pos,None,False)
    x=h
    for block in tr.h[layer:]:
        x=block(x,attention_mask=mask,cache_position=pos,use_cache=False,output_attentions=False)[0]
    if layer<8: x=tr.ln_f(x)
    return model.lm_head(x)

@torch.inference_mode()
def evaluate(connector,pair,src_h,tgt_h,target_model,ids,mask,layer,dims):
    s,t=pair; src=src_h[s][layer]; tgt=tgt_h[t][layer]
    mapped=connector(src.reshape(-1,dims[s]),s,t,layer).reshape_as(tgt)
    valid=mask[:,1:].bool(); hidden=(mapped[:,1:,:]-tgt[:,1:,:]).pow(2).mean(-1)[valid].mean().item()
    kl_sum=0.0; nll_sum=0.0; native_sum=0.0; count=0
    for st in range(0,len(ids),4):
        ii=ids[st:st+4].to(DEVICE); mm=mask[st:st+4].to(DEVICE); h=mapped[st:st+4].to(DEVICE)
        pred=tail_logits(target_model,h,layer,mm)
        native=target_model(input_ids=ii,attention_mask=mm,use_cache=False).logits
        p=F.log_softmax(pred[:,:-1,:],-1); q=F.softmax(native[:,:-1,:],-1)
        vm=mm[:,1:].bool()
        kl=F.kl_div(p,q,reduction='none').sum(-1)
        labels=ii[:,1:]
        nll=F.cross_entropy(pred[:,:-1,:].reshape(-1,pred.shape[-1]),labels.reshape(-1),reduction='none').reshape_as(vm)
        nll0=F.cross_entropy(native[:,:-1,:].reshape(-1,native.shape[-1]),labels.reshape(-1),reduction='none').reshape_as(vm)
        kl_sum += kl[vm].sum().item(); nll_sum += nll[vm].sum().item(); native_sum += nll0[vm].sum().item(); count += vm.sum().item()
    return hidden,kl_sum/count,(nll_sum-native_sum)/count

def main():
    for stale in [ROOT/'source'/'factor_payload.pt',ROOT/'source'/'independent_payload.pt']:
        if stale.exists(): stale.unlink()
    texts_train,texts_val,data_paths=get_data()
    tokenizer=AutoTokenizer.from_pretrained(MODELS[0][0],revision=MODELS[0][1],cache_dir=str(CACHE))
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    ids_tr,mask_tr=tokenize(tokenizer,texts_train); ids_va,mask_va=tokenize(tokenizer,texts_val)
    models=[]; model_dirs=[]; model_hashes=[]
    for mid,rev,_ in MODELS:
        path=snapshot_download(mid,revision=rev,cache_dir=str(CACHE),allow_patterns=['*.bin','*.json','*.txt','*.model','*.tiktoken'])
        model_dirs.append(path); model_hashes.append(rev)
        m=GPTNeoForCausalLM.from_pretrained(mid,revision=rev,cache_dir=str(CACHE)).eval().to(DEVICE)
        for p in m.parameters(): p.requires_grad_(False)
        models.append(m)
    dims=[d for _,_,d in MODELS]
    base_bytes=sum(f.stat().st_size for d in model_dirs for f in Path(d).rglob('*') if f.is_file())
    tr_h=features(models,ids_tr,mask_tr); va_h=features(models,ids_va,mask_va)
    results=[]
    for seed in SEEDS:
        torch.manual_seed(seed)
        fac=Factor(dims); sec_fac=fit(fac,tr_h,mask_tr,dims,seed,True)
        ind=Independent(dims); sec_ind=fit(ind,tr_h,mask_tr,dims,seed,False)
        factor_path=ROOT/'source'/'factor_payload.pt'; ind_path=ROOT/'source'/'independent_payload.pt'
        torch.save(fac.state_dict(),factor_path); torch.save(ind.state_dict(),ind_path)
        fbytes=factor_path.stat().st_size; ibytes=ind_path.stat().st_size
        fac2=Factor(dims); fac2.load_state_dict(torch.load(factor_path,map_location='cpu',weights_only=True))
        ind2=Independent(dims); ind2.load_state_dict(torch.load(ind_path,map_location='cpu',weights_only=True))
        pair_results=[]
        for pair in HELDOUT:
            for l in LAYERS:
                for name,model in [('factorized',fac2),('independent',ind2)]:
                    h,kl,nll=evaluate(model,pair,va_h,va_h,models[pair[1]],ids_va,mask_va,l,dims)
                    mac_f=RANK*dims[pair[0]]+RANK*RANK+RANK*dims[pair[1]]+dims[pair[1]]
                    mac_i=dims[pair[0]]*dims[pair[1]]+dims[pair[1]]
                    pair_results.append({'seed':seed,'source':pair[0],'target':pair[1],'layer':l,'condition':name,'hidden_mse':h,'target_tail_kl':kl,'nll_delta':nll,'connector_bytes':fbytes if name=='factorized' else ibytes,'full_deployment_bytes':base_bytes+(fbytes if name=='factorized' else ibytes),'mac_proxy':mac_f if name=='factorized' else mac_i,'train_seconds':sec_fac if name=='factorized' else sec_ind})
        # exact replay on one representative heldout map
        replay_input=va_h[0][2][:2,1:,:].reshape(-1,dims[0]); replay_ref=fac2(replay_input,0,2,2)
        replay=torch.max(torch.abs(replay_ref-fac(replay_input,0,2,2))).item()
        results.extend(pair_results)
        factor_path.unlink(); ind_path.unlink()
        print('seed',seed,'factor_bytes',fbytes,'independent_bytes',ibytes,'replay',replay,flush=True)
    with (ROOT/'source'/'results.json').open('w') as f: json.dump({'results':results,'base_model_bytes':base_bytes,'model_revisions':model_hashes,'dataset_revision':DATA_REV,'train_sequences':len(ids_tr),'validation_sequences':len(ids_va),'torch':torch.__version__,'device':DEVICE},f,indent=2)
    fields=list(results[0].keys())+['status','notes']
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in results:w.writerow({**r,'status':'SCREENING','notes':'Held-out ordered model pair; validation TinyStories'})
    (ROOT/'source'/'data_manifest.json').write_text(json.dumps({'train_file':str(data_paths[0]),'validation_file':str(data_paths[1]),'train_sequences':len(ids_tr),'validation_sequences':len(ids_va),'base_model_bytes':base_bytes},indent=2)+'\n')

if __name__=='__main__': main()
