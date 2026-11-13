"""Frozen TinyStories probe-transfer screen for MA-931 amendment 01."""
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
CACHE=Path('/tmp/ma930-cache')
torch.set_num_threads(1)
MODELS=[
 ('roneneldan/TinyStories-1M','77f1b168e219585646439073245fe87e56b3023e',64),
 ('roneneldan/TinyStories-3M','cfaf26ec85ecdfc1bd7c2638104cce55cb67f894',128),
 ('roneneldan/TinyStories-8M','8612e3b15c66ffa94eaa6ee0de5c96edd2d630af',256),
]
DATA_REV='f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
TRAIN_FILE='data/train-00000-of-00004-2d5a1467fff1081b.parquet'; VAL_FILE='data/validation-00000-of-00001-869c898b519ad725.parquet'
TRAIN_PAIRS=[(0,1),(1,0),(1,2),(2,1)]; ALL_PAIRS=[(i,j) for i in range(3) for j in range(3) if i!=j]; HELDOUT=[(0,2),(2,0)]
LAYERS=[4,8]; RANK=8; UPDATES=500; BATCH=256; PROBE_UPDATES=300; SEEDS=[93101,93107,93113]
ANIMALS={'dog','cat','puppy','kitten','bird','duck','rabbit','bunny','mouse','fox','bear','lion','tiger','elephant','horse','cow','sheep','pig','chicken','frog','turtle','fish','butterfly','owl','deer'}

@torch.inference_mode()
def get_data(tokenizer):
 tr=hf_hub_download('roneneldan/TinyStories',TRAIN_FILE,repo_type='dataset',revision=DATA_REV,cache_dir=str(CACHE)); va=hf_hub_download('roneneldan/TinyStories',VAL_FILE,repo_type='dataset',revision=DATA_REV,cache_dir=str(CACHE))
 trds=Dataset.from_parquet(tr); vads=Dataset.from_parquet(va)
 trtexts=[trds[i]['text'] for i in range(256,512)]; vtexts=[vads[i]['text'] for i in range(128,256)]
 a=tokenizer(trtexts,truncation=True,max_length=32,padding='max_length',return_tensors='pt'); b=tokenizer(vtexts,truncation=True,max_length=32,padding='max_length',return_tensors='pt')
 return (a['input_ids'],a['attention_mask']),(b['input_ids'],b['attention_mask']),tr,va

def labels_for(ids,mask,tokenizer):
 toks=tokenizer.convert_ids_to_tokens(ids.reshape(-1).tolist()); toks=np.asarray(toks,dtype=object).reshape(ids.shape)
 struct=torch.zeros_like(ids,dtype=torch.float32); sem=torch.zeros_like(ids,dtype=torch.float32); wordmask=torch.zeros_like(mask,dtype=torch.bool)
 for i in range(ids.shape[0]):
  for j in range(ids.shape[1]):
   if mask[i,j].item()==0: continue
   tok=toks[i,j]; boundary=tok.startswith('Ġ') or j==0
   struct[i,j]=float(boundary)
   word=tok[1:] if tok.startswith('Ġ') else (tok if j==0 else '')
   word=word.lower()
   if boundary and word.isalpha():
    wordmask[i,j]=True; sem[i,j]=float(word in ANIMALS)
 return {'structural':(struct,mask.bool()),'animal_lexical':(sem,wordmask)}

@torch.inference_mode()
def extract(models,ids,mask,batch=4):
 outs=[]
 for model in models:
  hs=[[] for _ in range(9)]
  for i in range(0,len(ids),batch):
   out=model(input_ids=ids[i:i+batch],attention_mask=mask[i:i+batch],output_hidden_states=True,use_cache=False)
   for k,h in enumerate(out.hidden_states):hs[k].append(h.cpu().float())
  outs.append([torch.cat(v,0) for v in hs])
 return outs

def flat_tokens(h,mask):
 keep=mask.reshape(-1).bool(); return [x.reshape(-1,x.shape[-1])[keep] for x in h]

class Factor(nn.Module):
 def __init__(self,dims):
  super().__init__();self.A=nn.ParameterList([nn.Parameter(torch.empty(RANK,d)) for d in dims]);self.B=nn.ParameterList([nn.Parameter(torch.empty(d,RANK)) for d in dims]);self.C=nn.ParameterList([nn.Parameter(torch.empty(RANK,RANK)) for _ in LAYERS]);self.bias=nn.ParameterList([nn.Parameter(torch.zeros(d)) for d in dims for _ in LAYERS])
  for p in list(self.A)+list(self.B)+list(self.C):nn.init.xavier_uniform_(p)
 def forward(self,x,s,t,l):
  z=F.linear(x,self.A[s]);z=F.linear(z,self.C[LAYERS.index(l)]);z=F.linear(z,self.B[t]);return z+self.bias[t*len(LAYERS)+LAYERS.index(l)]
class Independent(nn.Module):
 def __init__(self,dims):
  super().__init__();self.keys=[(s,t,l) for s,t in ALL_PAIRS for l in LAYERS];self.idx={k:i for i,k in enumerate(self.keys)};self.W=nn.ParameterList();self.b=nn.ParameterList()
  for s,t,l in self.keys:
   w=nn.Parameter(torch.empty(dims[t],dims[s]));nn.init.xavier_uniform_(w);self.W.append(w);self.b.append(nn.Parameter(torch.zeros(dims[t])))
 def forward(self,x,s,t,l):
  k=self.idx[(s,t,l)];return F.linear(x,self.W[k],self.b[k])
class Probe(nn.Module):
 def __init__(self,d):
  super().__init__();self.linear=nn.Linear(d,1)
 def forward(self,x):return self.linear(x).squeeze(-1)

def fit_connector(model,h,masks,dims,seed,factorized):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed+11);flat=[flat_tokens(x,masks) for x in h];pairs=TRAIN_PAIRS if factorized else ALL_PAIRS;opt=torch.optim.Adam(model.parameters(),lr=.003);t0=time.perf_counter()
 for _ in range(UPDATES):
  s,t=random.choice(pairs);l=random.choice(LAYERS);x=flat[s][l];y=flat[t][l];idx=torch.randint(x.shape[0],(BATCH,),generator=rng);loss=F.mse_loss(model(x[idx],s,t,l),y[idx]);opt.zero_grad();loss.backward();opt.step()
 return time.perf_counter()-t0

def fit_probe(x,y,mask,seed,shuffle=False):
 torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed+71);x=x[mask];y=y[mask].float()
 if shuffle: y=y[torch.randperm(len(y),generator=rng)]
 pos=max(float(y.sum()),1);neg=max(float(len(y)-y.sum()),1);pw=torch.tensor([neg/pos]);p=Probe(x.shape[-1]);opt=torch.optim.Adam(p.parameters(),lr=.01)
 for _ in range(PROBE_UPDATES):
  ix=torch.randint(len(y),(BATCH,),generator=rng);loss=F.binary_cross_entropy_with_logits(p(x[ix]),y[ix],pos_weight=pw);opt.zero_grad();loss.backward();opt.step()
 return p, int(y.sum()),len(y)

@torch.no_grad()
def score(probe,x,y,mask):
 x=x[mask];y=y[mask].float();z=probe(x).detach(); order=torch.argsort(z,descending=True); yy=y[order]; positives=max(float(y.sum()),1)
 precision=torch.cumsum(yy,0)/torch.arange(1,len(yy)+1,dtype=torch.float32); ap=float((precision*yy).sum()/positives)
 pred=(z>=0).float(); tp=((pred==1)&(y==1)).sum().item();tn=((pred==0)&(y==0)).sum().item();p=max((y==1).sum().item(),1);n=max((y==0).sum().item(),1);bal=.5*(tp/p+tn/n)
 return ap,bal,int(y.sum()),len(y)

def main():
 tokenizer=AutoTokenizer.from_pretrained(MODELS[0][0],revision=MODELS[0][1],cache_dir=str(CACHE))
 if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
 (ids_tr,mask_tr),(ids_va,mask_va),trpath,vapath=get_data(tokenizer)
 train_labels=labels_for(ids_tr,mask_tr,tokenizer);val_labels=labels_for(ids_va,mask_va,tokenizer)
 # Stop without label tuning if the frozen lexical concept is too rare.
 train_pos=int(train_labels['animal_lexical'][0].sum());val_pos=int(val_labels['animal_lexical'][0].sum())
 if train_pos<30 or val_pos<10: raise RuntimeError(f'NOT_ESTABLISHED: animal positives train={train_pos} validation={val_pos} below preregistered minimum')
 models=[];dirs=[]
 for mid,rev,_ in MODELS:
  path=snapshot_download(mid,revision=rev,cache_dir=str(CACHE),allow_patterns=['*.bin','*.json','*.txt','*.model','*.tiktoken']);dirs.append(path)
  m=GPTNeoForCausalLM.from_pretrained(mid,revision=rev,cache_dir=str(CACHE)).eval()
  for p in m.parameters():p.requires_grad_(False)
  models.append(m)
 dims=[d for _,_,d in MODELS];train_h=extract(models,ids_tr,mask_tr);val_h=extract(models,ids_va,mask_va)
 basebytes=sum(f.stat().st_size for d in dirs for f in Path(d).rglob('*') if f.is_file());rows=[];payload_notes=[]
 for seed in SEEDS:
  torch.manual_seed(seed);fac=Factor(dims);tf=fit_connector(fac,train_h,mask_tr,dims,seed,True);ind=Independent(dims);ti=fit_connector(ind,train_h,mask_tr,dims,seed,False)
  probes={};shuffled={}; probe_start=time.perf_counter()
  for target in [0,2]:
   for layer in LAYERS:
    for family,(yl,maskl) in train_labels.items():
     x=train_h[target][layer].reshape(-1,dims[target]);y=yl.reshape(-1);mk=maskl.reshape(-1)
     probes[(target,layer,family)],npos,nobs=fit_probe(x,y,mk,seed+target+layer,False)
     shuffled[(target,layer,family)],_,_=fit_probe(x,y,mk,seed+target+layer,True)
  tprobe=time.perf_counter()-probe_start
  fpath=ROOT/'source'/'connector_payload.pt';ipath=ROOT/'source'/'independent_payload.pt'
  torch.save(fac.state_dict(),fpath);torch.save(ind.state_dict(),ipath)
  fb=fpath.stat().st_size;ib=ipath.stat().st_size
  fac2=Factor(dims);fac2.load_state_dict(torch.load(fpath,map_location='cpu',weights_only=True));ind2=Independent(dims);ind2.load_state_dict(torch.load(ipath,map_location='cpu',weights_only=True))
  probe_bytes=sum(sum(t.numel()*t.element_size() for t in p.parameters()) for p in probes.values());metadata={'models':[m[0] for m in MODELS],'heldout_pairs':HELDOUT,'layers':LAYERS,'animal_lexicon':sorted(ANIMALS),'features':['structural','animal_lexical']}
  metadata_bytes=len(json.dumps(metadata,sort_keys=True).encode())
  probe_state={}
  for k,p in probes.items(): probe_state[str(k)]=p.state_dict()
  pp=ROOT/'source'/'probes.pt';torch.save(probe_state,pp);probe_file_bytes=pp.stat().st_size;loaded_probe=torch.load(pp,map_location='cpu',weights_only=True);probe_replay=0.0
  with torch.no_grad():
   for key,pb in probes.items():
    clone=Probe(dims[key[0]]);clone.load_state_dict(loaded_probe[str(key)]);xx=val_h[key[0]][key[1]][:2,1:,:].reshape(-1,dims[key[0]]);probe_replay=max(probe_replay,float((clone(xx)-pb(xx)).abs().max()))
  pp.unlink()
  for source,target in HELDOUT:
   for layer in LAYERS:
    sv=val_h[source][layer].reshape(-1,dims[source]);tv=val_h[target][layer].reshape(-1,dims[target])
    with torch.no_grad():
     factor_hidden=fac2(sv,source,target,layer);ind_hidden=ind2(sv,source,target,layer)
    hmask=mask_va.reshape(-1).bool();hmse_f=F.mse_loss(factor_hidden[hmask],tv[hmask]).item();hmse_i=F.mse_loss(ind_hidden[hmask],tv[hmask]).item()
    for family,(y,maskl) in val_labels.items():
     target_probe=probes[(target,layer,family)];rnd=shuffled[(target,layer,family)];yv=y.reshape(-1);mk=maskl.reshape(-1)
     native_ap,native_bal,npos,nobs=score(target_probe,tv,yv,mk)
     for condition,reprs,conn_bytes,extra in [('native',tv,0,0),('factorized',factor_hidden,fb,hmse_f),('independent',ind_hidden,ib,hmse_i),('shuffled_negative',factor_hidden,fb,hmse_f)]:
      probe=rnd if condition=='shuffled_negative' else target_probe
      ap,bal,_,_=score(probe,reprs,yv,mk)
      mac=0 if condition=='native' else (RANK*dims[source]+RANK*RANK+RANK*dims[target]+dims[target] if condition in ['factorized','shuffled_negative'] else dims[source]*dims[target]+dims[target])
      rows.append({'seed':seed,'source':source,'target':target,'layer':layer,'family':family,'condition':condition,'average_precision':ap,'balanced_accuracy':bal,'native_ap_retained':ap/max(native_ap,1e-8),'hidden_mse':extra if condition!='native' else 0.0,'connector_bytes':conn_bytes,'probe_bytes':probe_file_bytes,'full_deployment_bytes':basebytes+conn_bytes+probe_file_bytes+metadata_bytes,'mac_proxy':mac,'wall_seconds':tf+ti+tprobe})
  # exact connector replay
  rx=val_h[0][4][:2,1:,:].reshape(-1,dims[0]);err=float((fac2(rx,0,2,4)-fac(rx,0,2,4)).abs().max());err_ind=float((ind2(rx,0,2,4)-ind(rx,0,2,4)).abs().max());payload_notes.append({'seed':seed,'factor_roundtrip_max_abs':err,'independent_roundtrip_max_abs':err_ind,'factor_bytes':fb,'independent_bytes':ib,'probe_file_bytes':probe_file_bytes,'probe_roundtrip_max_abs':probe_replay,'metadata_bytes':metadata_bytes})
  fpath.unlink();ipath.unlink()
 out={'results':rows,'payload_checks':payload_notes,'animal_positives_train':train_pos,'animal_positives_validation':val_pos,'base_model_bytes':basebytes,'dataset_revision':DATA_REV,'train_rows':'256:512','validation_rows':'128:256','train_sequences':len(ids_tr),'validation_sequences':len(ids_va),'metadata_bytes':metadata_bytes,'torch':torch.__version__,'device':'cpu'}
 (ROOT/'source'/'results.json').write_text(json.dumps(out,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  fields=list(rows[0])+['status','notes'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({**r,'status':'SCREENING','notes':'Held-out TinyStories probe transfer; animal task is lexical only'})
 (ROOT/'source'/'data_manifest.json').write_text(json.dumps({'dataset_revision':DATA_REV,'train_file':str(trpath),'validation_file':str(vapath),'train_rows':'256:512','validation_rows':'128:256','animal_positive_count_train':train_pos,'animal_positive_count_validation':val_pos,'base_model_bytes':basebytes},indent=2)+'\n')
 print(json.dumps({'rows':len(rows),'payload_checks':payload_notes,'positive_counts':[train_pos,val_pos]},indent=2),flush=True)

if __name__=='__main__': main()
