import csv,hashlib,io,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';MODEL='openai-community/gpt2';REV='607a30d783dfa663caf39e06633721c8d4cfcd7e';FRESH=[52210,52211,52212];SEEDS=[0,1,2];PROCS=['prepend','append','reverse','rotate'];DOM={'fruit':['apple','banana','cherry','lemon'],'shape':['circle','square','triangle','diamond'],'color':['red','blue','green','yellow'],'vehicle':['car','truck','train','plane']};DN=list(DOM);TASKS=[(p,d) for p in range(4) for d in range(4)];LAYER=6;TURNS=8;torch.set_num_threads(4)
def load():
 t=AutoTokenizer.from_pretrained(MODEL,revision=REV);t.padding_side='left';t.pad_token=t.eos_token;m=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,torch_dtype=torch.float32);m.eval();return m,t
def demos(task,seed):
 p,d=task;dom=DN[d];w=DOM[dom];pairs=[]
 for j,x in enumerate(w[:3]):
  y=w[0] if p in (0,2) else w[-1] if p==1 else w[(j+1)%4];pairs.append((x,y))
 order=torch.randperm(3,generator=torch.Generator().manual_seed(seed+17*p+d)).tolist();return ''.join(f'For {dom}, {PROCS[p]} {pairs[i][0]} gives {pairs[i][1]}.\n' for i in order)
def query(task,turn):
 p,d=task;dom=DN[d];word=DOM[dom][turn%4];return f'For {dom}, {PROCS[p]} {word} gives'
def target(task,turn):
 p,d=task;w=DOM[DN[d]];j=turn%4
 if p==0:return w[0]
 if p==1:return w[-1]
 if p==2:return w[3-j]
 return w[(j+1)%4]
def getvec(m,t,task,seed):
 pref='Task mapping:\n';b=t([pref+'Apply:',pref+demos(task,seed)+query(task,0)],return_tensors='pt',padding=True,add_special_tokens=False)
 with torch.no_grad():o=m(**b,output_hidden_states=True,return_dict=True)
 return o.hidden_states[LAYER+1][1,-1]-o.hidden_states[LAYER+1][0,-1]
def blob(x):b=io.BytesIO();torch.save(x,b);return b.getvalue()
def context_tokens(t,tasks,seed,direct):
 texts=[]
 for task in tasks:
  for turn in range(TURNS):
   pre='Task mapping:\n'+demos(task,seed) if direct else 'Task mapping:\n'
   texts.append(pre+query(task,turn))
 ids=t(texts,add_special_tokens=False)['input_ids'];return ids
def run_dev():
 m,t=load();s=ROOT.parents[2]/'experiments/mirror_applications/ma-520-function-vector-mirror-distillation/artifacts/development_codes.pt';state=torch.load(s,map_location='cpu',weights_only=False)
 assert state['centers'].shape[0]==8 and state['basis'].shape[1]==4
 qids=context_tokens(t,TASKS,52200,False);dids=context_tokens(t,TASKS,52200,True)
 codeblob=blob({'centers':state['centers'].float(),'labels':TASKS,'layer':LAYER}); smoke={'tasks':len(TASKS),'turns':TURNS,'pca_rank':4,'pq_K':8,'init_context_tokens':sum(len(t('Task mapping:\n'+demos(q,0),add_special_tokens=False)['input_ids']) for q in TASKS),'query_tokens_total':sum(map(len,qids)),'repeat_icl_tokens_total':sum(map(len,dids)),'shared_codebook_bytes':len(codeblob)}
 (ART/'development_smoke.json').write_text(json.dumps(smoke,indent=2)+'\n');print(json.dumps(smoke))
def logits(m,t,prompts,vs=None):
 b=t(prompts,return_tensors='pt',padding=True,add_special_tokens=False);h=None
 if vs is not None:
  def hook(_m,_i,out):
   z=out[0] if isinstance(out,tuple) else out;z=z.clone();z[:,-1,:]+=vs.to(z.device);return (z,)+out[1:] if isinstance(out,tuple) else z
  h=m.transformer.h[LAYER].register_forward_hook(hook)
 try:
  with torch.no_grad():return m(**b,return_dict=True).logits[:,-1,:]
 finally:
  if h:h.remove()
def fresh():
 m,t=load();src=ROOT.parents[2]/'experiments/mirror_applications/ma-520-function-vector-mirror-distillation/artifacts/development_codes.pt';dev=torch.load(src,map_location='cpu',weights_only=False);rows=[]
 base=Path(snapshot_download(MODEL,revision=REV,tqdm_class=None));files=['model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json'];basebytes=sum((base/f).stat().st_size for f in files if (base/f).is_file())
 for world in FRESH:
  for seed in SEEDS:
   start=time.perf_counter();vs=torch.stack([getvec(m,t,q,world*31+seed+i) for i,q in enumerate(TASKS)]);extract=time.perf_counter()-start
   mean=dev['mean'];basis=dev['basis'];coeff=(vs-mean)@basis;pca=mean+coeff@basis.T;ids=torch.cdist(vs,dev['centers']).argmin(-1);pq=dev['centers'][ids]
   prompts=[]
   for task in TASKS:
    for turn in range(TURNS):prompts.append('Task mapping:\n'+query(task,turn))
   directprompts=[]
   for task in TASKS:
    for turn in range(TURNS):directprompts.append('Task mapping:\n'+demos(task,seed)+query(task,turn))
   targetids=torch.tensor([t(' '+target(task,turn),add_special_tokens=False)['input_ids'][0] for task in TASKS for turn in range(TURNS)])
   metrics=[]
   for name,z in [('repeat_icl',logits(m,t,directprompts)),('query_only',logits(m,t,prompts))]:
    prob=z.softmax(-1);metrics.append((name,z,float((z.argmax(-1)==targetids).float().mean()),float(-prob[torch.arange(len(targetids)),targetids].log().mean()),0.))
   state_methods=[('explicit_session_fv',vs,{'method':'explicit','vectors':vs.float().contiguous(),'task_labels':TASKS,'layer':LAYER}),('pca_session_code',pca,{'method':'pca','mean':mean,'basis':basis,'codes':coeff.half(),'task_labels':TASKS,'layer':LAYER}),('pq_session_code',pq,{'method':'pq','centers':dev['centers'].float(),'codes':ids.byte(),'task_labels':TASKS,'layer':LAYER})]
   for name,v,state in state_methods:
    start=time.perf_counter();z=logits(m,t,prompts,v.repeat_interleave(TURNS,dim=0));elapsed=time.perf_counter()-start;prob=z.softmax(-1);metrics.append((name,z,float((z.argmax(-1)==targetids).float().mean()),float(-prob[torch.arange(len(targetids)),targetids].log().mean()),elapsed))
   # Serialize each state bank once; charge all task/session codes and shared decoder/codebook.
   qctx=blob({'input_ids':context_tokens(t,TASKS,seed,False),'turns':TURNS});ictx=blob({'input_ids':context_tokens(t,TASKS,seed,True),'turns':TURNS})
   for name,z,acc,nll,elapsed in metrics:
    st=next((blob(state) for sn,v,state in state_methods if sn==name),b'')
    if st:
     f=PAY/f'{world}_{seed}_{name}.pt';f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(st);sha=hashlib.sha256(st).hexdigest();rel=str(f.relative_to(ROOT.parents[2]))
    else:sha='';rel=''
    ctxt=len(ictx) if name=='repeat_icl' else len(qctx)
    rows.append({'world':world,'seed':seed,'method':name,'accuracy':acc,'mean_target_nll':nll,'turns':TURNS,'session_state_bytes':len(st),'cumulative_context_bytes':ctxt,'total_state_plus_context_bytes':len(st)+ctxt,'cumulative_context_tokens':sum(map(len,context_tokens(t,TASKS,seed,name=='repeat_icl'))),'fv_extraction_seconds':extract,'session_inference_seconds':elapsed,'base_model_bytes':basebytes,'hash':sha,'path':rel})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'rows':len(rows),'base_model_bytes':basebytes}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);a=p.parse_args();run_dev() if a.phase=='development' else fresh()
