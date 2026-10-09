import csv,hashlib,io,json,statistics,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';MODEL_ID='openai-community/gpt2';REV='607a30d783dfa663caf39e06633721c8d4cfcd7e';DEV=[51600,51601];FRESH=[51610,51611,51612];SEEDS=[0,1,2];T=16;LAYER=6;RANKS=[4,8,16]
KEYS=['apple','banana','cherry','lemon','orange','peach','pear','plum'];COLORS=['red','blue','green','yellow','purple','black','white','pink']
torch.set_num_threads(4)
def load():
 tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=REV);tok.padding_side='left';tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=REV,torch_dtype=torch.float32);model.eval();return model,tok

def examples(world,seed,n=T):
 g=torch.Generator().manual_seed(world*100003+seed*7919+59);out=[]
 for i in range(n):
  perm=torch.randperm(8,generator=g).tolist();mapping={KEYS[j]:COLORS[perm[j]] for j in range(8)}
  demos=''.join(f"If the object is {k}, its tag is {mapping[k]}.\n" for k in KEYS[:4])
  qs=[(k,mapping[k]) for k in KEYS[4:]];out.append((demos,qs,mapping))
 return out

def ids_batch(tok,texts):return tok(texts,return_tensors='pt',padding=True,add_special_tokens=False)
def hidden_delta(model,tok,demos):
 prompts=['Task mapping:\nWhat is the tag?','Task mapping:\n'+demos+'What is the tag?'];b=ids_batch(tok,prompts)
 with torch.no_grad():o=model(**b,output_hidden_states=True,return_dict=True)
 return (o.hidden_states[LAYER+1][1,-1]-o.hidden_states[LAYER+1][0,-1]).detach()

def logits(model,tok,prompts,vec=None):
 b=ids_batch(tok,prompts);handle=None
 if vec is not None:
  def hook(_mod,_inp,out):
   h=out[0] if isinstance(out,tuple) else out;h=h.clone();h[:,-1,:]+=vec.to(h.device).unsqueeze(0);return (h,)+out[1:] if isinstance(out,tuple) else h
  handle=model.transformer.h[LAYER].register_forward_hook(hook)
 try:
  with torch.no_grad():return model(**b,return_dict=True).logits[:,-1,:]
 finally:
  if handle is not None:handle.remove()

def fit_dev(model,tok):
 fs=[]
 for w in DEV:
  for s in SEEDS:
   for demos,_,_ in examples(w,s):fs.append(hidden_delta(model,tok,demos))
 x=torch.stack(fs);mean=x.mean(0);_,_,v=torch.linalg.svd(x-mean,full_matrices=False);basis=v[:max(RANKS)].T.contiguous();torch.save({'mean':mean,'basis':basis,'dev_vectors':x},ART/'development_basis.pt')

def payload(method,vecs,state,r=0):
 if method=='explicit':o={'method':method,'vectors':vecs.float().contiguous(),'tasks':len(vecs)}
 else:
  mean=state['mean'];basis=state['basis'][:,:r].contiguous();coords=(vecs-mean)@basis
  o={'method':method,'mean':mean.contiguous(),'basis':basis,'coords':coords.half().contiguous(),'rank':r,'tasks':len(vecs)}
 b=io.BytesIO();torch.save(o,b);return b.getvalue()

def evaluate_bank(model,tok,w,s,devstate):
 tasks=examples(w,s);vecs=torch.stack([hidden_delta(model,tok,d) for d,_,_ in tasks]);targets=[];query_prompts=[];icl_prompts=[];task_ix=[]
 for i,(d,qs,_) in enumerate(tasks):
  for k,c in qs:
   query_prompts.append(f"Task mapping:\nIf the object is {k}, its tag is");icl_prompts.append(f"Task mapping:\n{d}If the object is {k}, its tag is");targets.append(tok(' '+c,add_special_tokens=False)['input_ids'][0]);task_ix.append(i)
 target=torch.tensor(targets);task_ix=torch.tensor(task_ix);results=[]
 t0=time.perf_counter();direct=logits(model,tok,icl_prompts);elapsed_icl=time.perf_counter()-t0
 t0=time.perf_counter();base=logits(model,tok,query_prompts);elapsed_base=time.perf_counter()
 def acc(x):
  p=x.softmax(-1);return float((x.argmax(-1)==target).float().mean()),float(-p[torch.arange(len(target)),target].log().mean())
 for name,x,r in [('query_only',base,0),('direct_icl',direct,0)]:
  a,nll=acc(x);results.append({'method':name,'rank':r,'accuracy':a,'mean_target_nll':nll,'vector_nrmse':0.0,'inference_seconds':elapsed_icl if name=='direct_icl' else elapsed_base})
 configs=[('explicit',0,vecs)]
 for r in RANKS:
  b=devstate['basis'][:,:r];coord=(vecs-devstate['mean'])@b;rec=devstate['mean']+coord@b.T;configs.append(('mirror_pca',r,rec))
 for method,r,recon in configs:
  elapsed=0.;pieces=[]
  for i in range(len(tasks)):
   ix=(task_ix==i).nonzero().flatten();q=[query_prompts[j] for j in ix.tolist()];t0=time.perf_counter();z=logits(model,tok,q,recon[i]);elapsed+=time.perf_counter()-t0;pieces.append((ix,z))
  all_logits=torch.empty((len(target),model.config.vocab_size))
  for ix,z in pieces:all_logits[ix]=z
  a,nll=acc(all_logits);err=float((recon-vecs).norm()/(vecs.norm()+1e-12));results.append({'method':method,'rank':r,'accuracy':a,'mean_target_nll':nll,'vector_nrmse':err,'inference_seconds':elapsed})
 payloads=[]
 b=payload('explicit',vecs,devstate);payloads.append(('explicit',0,b))
 for r in RANKS:payloads.append(('mirror_pca',r,payload('mirror_pca',vecs,devstate,r)))
 return vecs,results,payloads

def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True);model,tok=load()
 if phase=='development':fit_dev(model,tok);print('development task vectors and PCA basis saved');return
 state=torch.load(ART/'development_basis.pt',map_location='cpu',weights_only=False);rows=[];base_model_dir=snapshot_download(MODEL_ID,revision=REV);base_names=['model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json'];base_bytes=sum((Path(base_model_dir)/n).stat().st_size for n in base_names if (Path(base_model_dir)/n).is_file())
 for w in FRESH:
  for s in SEEDS:
   vecs,metrics,blobs=evaluate_bank(model,tok,w,s,state);sizes={}
   for name,r,b in blobs:
    tag=f'{name}_R{r}';sizes[tag]=len(b);p=PAY/f'{w}_{s}_{tag}.pt';p.write_bytes(b);rows.append({'world':w,'seed':s,'method':name,'rank':r,'accuracy':next(x['accuracy'] for x in metrics if x['method']==name and x['rank']==r),'mean_target_nll':next(x['mean_target_nll'] for x in metrics if x['method']==name and x['rank']==r),'vector_nrmse':next(x['vector_nrmse'] for x in metrics if x['method']==name and x['rank']==r),'intervention_payload_bytes':len(b),'total_deployment_bytes':base_bytes+len(b),'base_model_bytes':base_bytes,'inference_seconds':next(x['inference_seconds'] for x in metrics if x['method']==name and x['rank']==r),'hash':hashlib.sha256(b).hexdigest(),'path':str(p.relative_to(ROOT.parents[2]))})
   for m in metrics:
    if m['method'] in ('query_only','direct_icl'):rows.append({'world':w,'seed':s,'method':m['method'],'rank':0,'accuracy':m['accuracy'],'mean_target_nll':m['mean_target_nll'],'vector_nrmse':0.0,'intervention_payload_bytes':0,'total_deployment_bytes':base_bytes,'base_model_bytes':base_bytes,'inference_seconds':m['inference_seconds'],'hash':'','path':''})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'rows':len(rows),'base_model_bytes':base_bytes}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
