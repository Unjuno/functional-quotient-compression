import csv,hashlib,io,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';MODEL='openai-community/gpt2';REV='607a30d783dfa663caf39e06633721c8d4cfcd7e';DEV=[52000,52001];FRESH=[52010,52011,52012];SEEDS=[0,1,2];PROCS=['prepend','append','reverse','rotate'];DOM={'fruit':['apple','banana','cherry','lemon'],'shape':['circle','square','triangle','diamond'],'color':['red','blue','green','yellow'],'vehicle':['car','truck','train','plane']};DN=list(DOM);TASKS=[(p,d) for p in range(4) for d in range(4)];LAYER=6;K=8;R=4;torch.set_num_threads(4)
def load():
 t=AutoTokenizer.from_pretrained(MODEL,revision=REV);t.padding_side='left';t.pad_token=t.eos_token;m=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,torch_dtype=torch.float32);m.eval();return m,t
def demos(task,seed=0):
 p,d=task;dom=DN[d];w=DOM[dom];ps=[]
 for j,x in enumerate(w[:3]):
  y=w[0] if p==0 else w[-1] if p==1 else w[3-j] if p==2 else w[(j+1)%4];ps.append((x,y))
 order=torch.randperm(3,generator=torch.Generator().manual_seed(seed+100*p+d)).tolist();return ''.join(f'For {dom}, {PROCS[p]} {ps[i][0]} gives {ps[i][1]}.\n' for i in order)
def query(task):p,d=task;return f'For {DN[d]}, {PROCS[p]} {DOM[DN[d]][3]} gives'
def target(task):p,d=task;w=DOM[DN[d]];return w[0] if p in (0,2,3) else w[-1]
def getvec(m,t,task,seed):
 pref='Task mapping:\n';x=t([pref+'Apply:',pref+demos(task,seed)+query(task)],return_tensors='pt',padding=True,add_special_tokens=False)
 with torch.no_grad():o=m(**x,output_hidden_states=True,return_dict=True)
 return o.hidden_states[LAYER+1][1,-1]-o.hidden_states[LAYER+1][0,-1]
def logits(m,t,tasks,vs=None,direct=False):
 texts=[('Task mapping:\n'+demos(q,0)+query(q)) if direct else 'Task mapping:\n'+query(q) for q in tasks];x=t(texts,return_tensors='pt',padding=True,add_special_tokens=False);h=None
 if vs is not None:
  def hook(_m,_i,out):
   z=out[0] if isinstance(out,tuple) else out;z=z.clone();z[:,-1,:]+=vs.to(z.device);return (z,)+out[1:] if isinstance(out,tuple) else z
  h=m.transformer.h[LAYER].register_forward_hook(hook)
 try:
  with torch.no_grad():return m(**x,return_dict=True).logits[:,-1,:]
 finally:
  if h:h.remove()
def blob(x):b=io.BytesIO();torch.save(x,b);return b.getvalue()
def fitdev():
 m,t=load();x=torch.stack([getvec(m,t,q,52000+i) for i,q in enumerate(TASKS)]);mean=x.mean(0);_,_,v=torch.linalg.svd(x-mean,full_matrices=False);basis=v[:R].T.contiguous();coords=(x-mean)@basis;rec=mean+coords@basis.T
 # Deterministic K-means initialized from evenly spaced task vectors; fixed K=8.
 centers=x[torch.linspace(0,len(x)-1,K).long()].clone()
 for _ in range(20):
  ids=torch.cdist(x,centers).argmin(1);centers=torch.stack([x[ids==i].mean(0) if (ids==i).any() else centers[i] for i in range(K)])
 ids=torch.cdist(x,centers).argmin(1);pq=centers[ids]
 state={'mean':mean,'basis':basis,'centers':centers,'ids':ids};torch.save(state,ART/'development_codes.pt')
 (ART/'development_smoke.json').write_text(json.dumps({'tasks':len(TASKS),'rank':R,'K':K,'pca_nrmse':float((rec-x).norm()/x.norm()),'pq_nrmse':float((pq-x).norm()/x.norm()),'pca_payload_bytes':len(blob({'mean':mean,'basis':basis,'coords':coords.half()})),'pq_payload_bytes':len(blob({'centers':centers.half(),'ids':ids.byte()}))},indent=2)+'\n');print('development complete')
def fresh():
 m,t=load();state=torch.load(ART/'development_codes.pt',map_location='cpu',weights_only=False);rows=[];base=Path(snapshot_download(MODEL,revision=REV,tqdm_class=None));names=['model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json'];basebytes=sum((base/n).stat().st_size for n in names if (base/n).is_file())
 for w in FRESH:
  for s in SEEDS:
   vs=torch.stack([getvec(m,t,q,w*100+s) for q in TASKS]);mean=state['mean'];basis=state['basis'];coord=(vs-mean)@basis;rec=mean+coord@basis.T;ids=torch.cdist(vs,state['centers']).argmin(1);pq=state['centers'][ids]
   targetids=torch.tensor([t(' '+target(q),add_special_tokens=False)['input_ids'][0] for q in TASKS]);direct=logits(m,t,TASKS,direct=True);basez=logits(m,t,TASKS)
   def met(z):p=z.softmax(-1);return float((z.argmax(-1)==targetids).float().mean()),float(-p[torch.arange(len(TASKS)),targetids].log().mean())
   methods=[('direct_icl',direct,None,None),('query_only',basez,None,None)]
   for name,zv,code in [('explicit',vs,{'method':'explicit','vectors':vs.float().contiguous(),'task_labels':TASKS,'layer':LAYER}),('generic_pca',rec,{'mean':mean,'basis':basis,'coords':coord.half()}),('mirror_pq',pq,{'centers':state['centers'].half(),'ids':ids.byte()})]:
    out=logits(m,t,TASKS,zv);methods.append((name,out,zv,code))
   for name,z,zv,code in methods:
    acc,nll=met(z);b=blob(code) if code is not None else b'';path='';sha=''
    if b:path=PAY/f'{w}_{s}_{name}.pt';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b);sha=hashlib.sha256(b).hexdigest();rel=str(path.relative_to(ROOT.parents[2]))
    else:rel=''
    vnrm=float((zv-vs).norm()/vs.norm()) if zv is not None else 0.
    rows.append({'world':w,'seed':s,'method':name,'accuracy':acc,'mean_target_nll':nll,'vector_nrmse':vnrm,'payload_bytes':len(b),'base_model_bytes':basebytes,'total_deployment_bytes':len(b)+basebytes,'hash':sha,'path':rel})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'rows':len(rows),'base_model_bytes':basebytes}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);a=p.parse_args();fitdev() if a.phase=='development' else fresh()
