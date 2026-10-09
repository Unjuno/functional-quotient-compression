import csv,hashlib,io,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads'
MODEL='openai-community/gpt2';REV='607a30d783dfa663caf39e06633721c8d4cfcd7e';DEV=[51900,51901];FRESH=[51910,51911,51912];SEEDS=[0,1,2]
PROCS=['prepend','append','reverse','rotate'];DOMAINS={'fruit':['apple','banana','cherry','lemon'],'shape':['circle','square','triangle','diamond'],'color':['red','blue','green','yellow'],'vehicle':['car','truck','train','plane']};DN=list(DOMAINS);LAYER=6;torch.set_num_threads(4)
TASKS=[(p,d) for p in range(4) for d in range(4)]
def load():
 t=AutoTokenizer.from_pretrained(MODEL,revision=REV);t.padding_side='left';t.pad_token=t.eos_token
 m=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,torch_dtype=torch.float32);m.eval();return m,t
def demos(task,seed=0):
 p,d=task;dom=DN[d];w=DOMAINS[dom];pairs=[]
 for j,x in enumerate(w[:3]):
  if p==0:y=w[0]
  elif p==1:y=w[-1]
  elif p==2:y=w[3-j]
  else:y=w[(j+1)%4]
  pairs.append((x,y))
 g=torch.Generator().manual_seed(seed+100*p+d);order=torch.randperm(3,generator=g).tolist()
 return ''.join(f'For {dom}, {PROCS[p]} {pairs[i][0]} gives {pairs[i][1]}.\n' for i in order)
def query(task):
 p,d=task;dom=DN[d];return f'For {dom}, {PROCS[p]} {DOMAINS[dom][3]} gives'
def target(task):
 p,d=task;w=DOMAINS[DN[d]];return w[0] if p in (0,2,3) else w[-1]
def vec(model,tok,task,seed):
 a='Task mapping:\n';prompts=[a+'Apply:',a+demos(task,seed)+query(task)]
 b=tok(prompts,return_tensors='pt',padding=True,add_special_tokens=False)
 with torch.no_grad():o=model(**b,output_hidden_states=True,return_dict=True)
 return o.hidden_states[LAYER+1][1,-1]-o.hidden_states[LAYER+1][0,-1]
def ctx(model,tok,task,seed):
 b=tok('Task mapping:\n'+demos(task,seed)+query(task),return_tensors='pt',add_special_tokens=False)
 with torch.no_grad():o=model(**b,output_hidden_states=True,return_dict=True)
 return o.hidden_states[LAYER+1][0,-1]
def forward(model,tok,tasks,vectors=None,direct=False):
 texts=[('Task mapping:\n'+demos(t,0)+query(t)) if direct else ('Task mapping:\n'+query(t)) for t in tasks];b=tok(texts,return_tensors='pt',padding=True,add_special_tokens=False);h=None
 if vectors is not None:
  def hook(_m,_i,out):
   x=out[0] if isinstance(out,tuple) else out;x=x.clone();x[:,-1,:]+=vectors.to(x.device);return (x,)+out[1:] if isinstance(out,tuple) else x
  h=model.transformer.h[LAYER].register_forward_hook(hook)
 try:
  with torch.no_grad():return model(**b,return_dict=True).logits[:,-1,:]
 finally:
  if h:h.remove()
def dev():
 model,tok=load(); bad={}
 for d in DN:
  for w in DOMAINS[d]:
   if len(tok(' '+w,add_special_tokens=False)['input_ids'])!=1:bad[w]=tok(' '+w,add_special_tokens=False)['input_ids']
 assert not bad,bad
 train=[vec(model,tok,t,51900+i) for i,t in enumerate(TASKS)];protos=[ctx(model,tok,t,51900+i) for i,t in enumerate(TASKS)]
 sims=torch.nn.functional.normalize(torch.stack(protos),dim=1)@torch.nn.functional.normalize(torch.stack(protos),dim=1).T
 assert torch.diag(sims).min()>0.99
 (ART/'development_smoke.json').write_text(json.dumps({'tasks':len(TASKS),'hidden':768,'diag_sim_min':float(torch.diag(sims).min()),'router_prototype_bytes_fp16':len(payload(torch.stack(protos).half())),'fv_bank_bytes_fp32':len(payload(torch.stack(train).float()))},indent=2)+'\n')
 print('development routing/prototype smoke passed',float(torch.diag(sims).min()))
def payload(x):
 b=io.BytesIO();torch.save(x,b);return b.getvalue()
def fresh():
 model,tok=load();protos=torch.stack([ctx(model,tok,t,51900+i) for i,t in enumerate(TASKS)])
 fvs=torch.stack([vec(model,tok,t,51900+i) for i,t in enumerate(TASKS)])
 protobytes=payload(protos.half());fvbytes=payload(fvs.float());rows=[]
 base=Path(snapshot_download(MODEL,revision=REV,tqdm_class=None));names=['model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json'];basebytes=sum((base/n).stat().st_size for n in names if (base/n).is_file())
 for w in FRESH:
  for s in SEEDS:
   ts=TASKS.copy();g=torch.Generator().manual_seed(w*100003+s*7919);order=torch.randperm(16,generator=g).tolist();ts=[ts[i] for i in order];queries=torch.stack([ctx(model,tok,t,w+s+i) for i,t in enumerate(ts)])
   sim=torch.nn.functional.normalize(queries,dim=1)@torch.nn.functional.normalize(protos,dim=1).T;pred=sim.argmax(-1);truth=torch.tensor([TASKS.index(t) for t in ts]);racc=float((pred==truth).float().mean())
   ids=torch.tensor([TASKS.index(t) for t in ts]);oracle=fvs[ids];routed=fvs[pred];direct=forward(model,tok,ts,direct=True);basez=forward(model,tok,ts);oz=forward(model,tok,ts,oracle);rz=forward(model,tok,ts,routed);targetids=torch.tensor([tok(' '+target(t),add_special_tokens=False)['input_ids'][0] for t in ts])
   def metrics(z):
    p=z.softmax(-1);return float((z.argmax(-1)==targetids).float().mean()),float(-p[torch.arange(16),targetids].log().mean())
   for method,z,route in [('direct_icl',direct,None),('query_only',basez,None),('oracle_fv',oz,None),('routed_fv',rz,racc)]:
    acc,nll=metrics(z);b=payload({'fv_bank':fvs.float(),'router_prototypes':protos.half(),'task_label_table':TASKS,'layer':LAYER}) if method=='routed_fv' else b''
    path=''
    if b:
     path=PAY/f'{w}_{s}_router_fv.pt';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b);h=hashlib.sha256(b).hexdigest();rel=str(path.relative_to(ROOT.parents[2]));size=len(b)
    else:h='';rel='';size=0
    rows.append({'world':w,'seed':s,'method':method,'routing_accuracy':'' if route is None else route,'accuracy':acc,'mean_target_nll':nll,'payload_bytes':size,'base_model_bytes':basebytes,'total_deployment_bytes':size+basebytes,'hash':h,'path':rel})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'rows':len(rows),'base_model_bytes':basebytes}))
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['development','fresh'],required=True);x=a.parse_args();dev() if x.phase=='development' else fresh()
