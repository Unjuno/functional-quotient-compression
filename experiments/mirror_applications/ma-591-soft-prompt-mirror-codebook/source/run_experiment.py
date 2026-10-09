#!/usr/bin/env python3
"""MA-591 Pythia soft-prompt-bank compression screen."""
from __future__ import annotations
import argparse,hashlib,json,re,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from transformers import AutoTokenizer,GPTNeoXForCausalLM
REV='e93a9faa9c77e5d09219f6c868bfc7a1bd65593c';MODEL_SHA='3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd'
DATA_SHA={'train.txt':'9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f','valid.txt':'f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8','test.txt':'d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0'}
TASKS,PROMPT_LEN,HIDDEN,RANK,WIN=8,8,512,4,64
METHODS=('zero_prompt','full_prompt','mirror_rank4','native_rank4')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def articles(path,tok):
 out=[];title=None;body=[]
 for line in Path(path).read_text(encoding='utf8').splitlines():
  if line.startswith(' = ') and line.endswith(' = ') and not line.startswith(' = ='):
   if title:
    ids=np.asarray(tok.encode(title+'\n'+'\n'.join(body),add_special_tokens=False),np.int64)
    if len(ids)>=2048:out.append((title,ids))
   title=line.strip(' =');body=[]
  elif title:body.append(line)
 if title:
  ids=np.asarray(tok.encode(title+'\n'+'\n'.join(body),add_special_tokens=False),np.int64)
  if len(ids)>=2048:out.append((title,ids))
 return out
def initialize_prompt(rng):return torch.nn.Parameter(torch.tensor(rng.normal(0,0.02,(PROMPT_LEN,HIDDEN)),dtype=torch.float32))
def task_loss(model,embed,prompt,ids,train=True):
 x=torch.tensor(ids,dtype=torch.long);text=embed(x)[None];p=prompt[None];inp=torch.cat([p,text],dim=1)
 labels=torch.full((1,PROMPT_LEN+len(ids)),-100,dtype=torch.long);labels[0,PROMPT_LEN:]=x
 out=model(inputs_embeds=inp,labels=labels,use_cache=False,return_dict=True)
 return out.loss
def eval_nll(model,embed,prompt,ids):
 x=torch.tensor(ids,dtype=torch.long);text=embed(x)[None];inp=torch.cat([prompt[None],text],dim=1)
 labels=torch.full((1,PROMPT_LEN+len(ids)),-100,dtype=torch.long);labels[0,PROMPT_LEN+WIN:]=x[WIN:]
 with torch.inference_mode():out=model(inputs_embeds=inp,labels=labels,use_cache=False,return_dict=True)
 n=int((labels[:,1:]!=-100).sum());return float(out.loss) if n else float('nan')
def decode_task_prompt(mean16,basis16,codes16,task):
 return codes16[task].astype(np.float32)@basis16.astype(np.float32).T+mean16.astype(np.float32)
def serialize(path,**kw):np.savez(path,**kw);return path.stat().st_size
def run(seed,split,model_dir,data_dir,outdir):
 wall=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(seed);model_dir=Path(model_dir);data_dir=Path(data_dir);outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
 if sha(model_dir/'model.safetensors')!=MODEL_SHA:raise ValueError('model hash mismatch')
 for n,h in DATA_SHA.items():
  if sha(data_dir/n)!=h:raise ValueError('dataset hash mismatch '+n)
 tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True);t=time.perf_counter();model=GPTNeoXForCausalLM.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.float32).eval();load_s=time.perf_counter()-t
 for p in model.parameters():p.requires_grad_(False)
 rng=np.random.default_rng(seed);pool=articles(data_dir/'train.txt',tok)
 if len(pool)<TASKS:raise ValueError('not enough eligible article tasks')
 chosen=rng.choice(len(pool),size=TASKS,replace=False);tasks=[pool[int(i)] for i in chosen]
 embed=model.get_input_embeddings();prompts=[];task_train=[];task_eval=[];train_s=[];updates=0
 for ti,(title,ids) in enumerate(tasks):
  prompt=initialize_prompt(np.random.default_rng(seed*100+ti));opt=torch.optim.AdamW([prompt],lr=0.05);t0=time.perf_counter()
  for step in range(8):
   losses=[]
   for _ in range(4):
    start=int(rng.integers(0,1024-WIN));window=ids[start:start+WIN]
    losses.append(task_loss(model,embed,prompt,window))
   loss=torch.stack(losses).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step();updates+=1
  train_s.append(time.perf_counter()-t0);prompts.append(prompt.detach().numpy().copy())
  # Four non-overlapping evaluation contexts/targets in the disjoint article suffix.
  starts=[1024,1152,1280,1408]
  task_eval.append([ids[a-WIN:a+WIN].tolist() for a in starts])
  task_train.append({'title':title,'train_tokens':[0,1024],'heldout_tokens':[1024,1536]})
 prompt_bank=np.stack(prompts).astype(np.float32);mu=prompt_bank.reshape(TASKS,-1).mean(0,dtype=np.float64).astype(np.float32);centered=prompt_bank.reshape(TASKS,-1)-mu
 _,_,vt=np.linalg.svd(centered,full_matrices=False);basis=vt[:RANK].T.astype(np.float32);codes=centered@basis
 # Quantize each stored method exactly as its inference serializer.
 full16=prompt_bank.astype(np.float16);mean16=mu.astype(np.float16);basis16=basis.astype(np.float16);codes16=codes.astype(np.float16)
 compressed=np.stack([decode_task_prompt(mean16,basis16,codes16,i) for i in range(TASKS)]).reshape(TASKS,PROMPT_LEN,HIDDEN)
 state={}
 state['zero_prompt']=serialize(outdir/'zero_prompt_bank.npz',prompts=np.zeros_like(full16),task_titles=np.asarray([t for t,_ in tasks]),shape=np.asarray([TASKS,PROMPT_LEN,HIDDEN],np.int32),schema=np.asarray([591,1],np.int32))
 state['full_prompt']=serialize(outdir/'full_prompt_bank.npz',prompts=full16,task_titles=np.asarray([t for t,_ in tasks]),shape=np.asarray([TASKS,PROMPT_LEN,HIDDEN],np.int32),schema=np.asarray([591,2],np.int32))
 compfile=outdir/'mirror_rank4_bank.npz';state['mirror_rank4']=serialize(compfile,mean=mean16,basis=basis16,codes=codes16,task_titles=np.asarray([t for t,_ in tasks]),shape=np.asarray([TASKS,PROMPT_LEN,HIDDEN,RANK],np.int32),schema=np.asarray([591,3],np.int32))
 state['native_rank4']=serialize(outdir/'native_rank4_bank.npz',mean=mean16,basis=basis16,codes=codes16,task_titles=np.asarray([t for t,_ in tasks]),shape=np.asarray([TASKS,PROMPT_LEN,HIDDEN,RANK],np.int32),schema=np.asarray([591,3],np.int32))
 methods={m:[] for m in METHODS};runtime={m:[] for m in METHODS};reconstruct_s=[]
 for ti in range(TASKS):
  # Include code reconstruction wall time, then score four held-out windows.
  t0=time.perf_counter();decoded_task=decode_task_prompt(mean16,basis16,codes16,ti).reshape(PROMPT_LEN,HIDDEN);decode_sec=time.perf_counter()-t0
  pbank={'zero_prompt':np.zeros((PROMPT_LEN,HIDDEN),np.float32),'full_prompt':full16[ti].astype(np.float32),'mirror_rank4':decoded_task,'native_rank4':decoded_task};reconstruct_s.append(decode_sec)
  for method,prompt in pbank.items():
   p=torch.tensor(prompt,dtype=torch.float32);tin=time.perf_counter();losses=[eval_nll(model,embed,p,w) for w in task_eval[ti]]
   runtime[method].append(time.perf_counter()-tin);methods[method].append(float(np.mean(losses)))
 result={}
 for method,vals in methods.items():result[method]={'mean_nll':float(np.mean(vals)),'std_task_nll':float(np.std(vals,ddof=1)),'nll_delta_vs_full_prompt':float(np.mean(vals)-np.mean(methods['full_prompt'])),'serialized_prompt_bank_bytes':state[method],'per_task_prompt_tokens':PROMPT_LEN,'mean_reconstruction_seconds':float(np.mean(reconstruct_s)) if method in ('mirror_rank4','native_rank4') else 0.0,'mean_inference_seconds_per_task':float(np.mean(runtime[method])),'total_inference_seconds':float(np.sum(runtime[method]))}
 rep={'experiment_id':'MA-591','seed':seed,'split':split,'model_revision':REV,'model_sha256':MODEL_SHA,'dataset':'train.txt articles with within-article suffix split','dataset_sha256':DATA_SHA['train.txt'],'task_titles':[t for t,_ in tasks],'task_ranges':task_train,'task_count':TASKS,'prompt_length':PROMPT_LEN,'hidden_size':HIDDEN,'prompt_updates':updates,'steps_per_task':8,'windows_per_update':4,'methods':result,'actual_serialized_bytes':state,'compute':{'model_load_seconds':load_s,'mean_training_seconds_per_task':float(np.mean(train_s)),'total_prompt_training_seconds':float(np.sum(train_s)),'total_wall_seconds':time.perf_counter()-wall,'virtual_tokens_per_inference':PROMPT_LEN,'added_token_embedding_operations_per_example':PROMPT_LEN*HIDDEN},'note':'Model weights are frozen; only task prompts are trained. Compressed prompts are decoded from serialized FP16 PCA coordinates before evaluation.'}
 (outdir/'metrics.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n');print(json.dumps(rep,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.seed,a.split,a.model_dir,a.data_dir,a.out)
if __name__=='__main__':main()
