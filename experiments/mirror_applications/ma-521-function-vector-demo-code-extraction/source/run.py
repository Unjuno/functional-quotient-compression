import csv,hashlib,io,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';MODEL='openai-community/gpt2';REV='607a30d783dfa663caf39e06633721c8d4cfcd7e';DEV=[52100,52101];FRESH=[52110,52111,52112];SEEDS=[0,1,2];P=['prepend','append','reverse','rotate'];D={'fruit':['apple','banana','cherry','lemon'],'shape':['circle','square','triangle','diamond'],'color':['red','blue','green','yellow'],'vehicle':['car','truck','train','plane']};DN=list(D);TASKS=[(p,d) for p in range(4) for d in range(4)];LAYER=6;CODE=4;torch.set_num_threads(4)
def load():
 t=AutoTokenizer.from_pretrained(MODEL,revision=REV);t.padding_side='left';t.pad_token=t.eos_token;m=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,torch_dtype=torch.float32);m.eval();return m,t
def demo(task,seed):
 p,d=task;dom=DN[d];w=D[dom];lines=[]
 for j,x in enumerate(w[:3]):
  y=w[0] if p in (0,2) else w[-1] if p==1 else w[(j+1)%4];lines.append((x,y))
 order=torch.randperm(3,generator=torch.Generator().manual_seed(seed+101*p+d)).tolist();return ''.join(f'For {dom}, {P[p]} {lines[i][0]} gives {lines[i][1]}.\n' for i in order)
def query(task):p,d=task;return f'For {DN[d]}, {P[p]} {D[DN[d]][3]} gives'
def target(task):p,d=task;return D[DN[d]][0] if p in (0,2,3) else D[DN[d]][-1]
def hidden(m,t,task,seed):
 x=t('Task mapping:\n'+demo(task,seed)+query(task),return_tensors='pt',add_special_tokens=False)
 with torch.no_grad():o=m(**x,output_hidden_states=True,return_dict=True)
 return o.hidden_states[LAYER+1][0,-1]
def fv(m,t,task,seed):
 pre='Task mapping:\n';x=t([pre+'Apply:',pre+demo(task,seed)+query(task)],return_tensors='pt',padding=True,add_special_tokens=False)
 with torch.no_grad():o=m(**x,output_hidden_states=True,return_dict=True)
 return o.hidden_states[LAYER+1][1,-1]-o.hidden_states[LAYER+1][0,-1]
def forward(m,t,tasks,vecs=None,direct=False):
 txt=[('Task mapping:\n'+demo(q,0)+query(q)) if direct else 'Task mapping:\n'+query(q) for q in tasks];x=t(txt,return_tensors='pt',padding=True,add_special_tokens=False);h=None
 if vecs is not None:
  def hook(_m,_i,out):
   z=out[0] if isinstance(out,tuple) else out;z=z.clone();z[:,-1,:]+=vecs.to(z.device);return (z,)+out[1:] if isinstance(out,tuple) else z
  h=m.transformer.h[LAYER].register_forward_hook(hook)
 try:
  with torch.no_grad():return m(**x,return_dict=True).logits[:,-1,:]
 finally:
  if h:h.remove()
def blob(x):b=io.BytesIO();torch.save(x,b);return b.getvalue()
def train_dev():
 m,t=load();X=[];Y=[]
 for w in DEV:
  for s in SEEDS:
   for i,task in enumerate(TASKS):X.append(hidden(m,t,task,w+s+i));Y.append(i)
 X=torch.stack(X);Y=torch.tensor(Y);mu=X.mean(0);sd=X.std(0).clamp_min(1e-4);Z=(X-mu)/sd
 # Fixed 4-D linear bottleneck predicts task codebook coordinates; ridge fit on development only.
 codes=torch.eye(16)[:,:CODE];A=torch.linalg.solve(Z.T@Z+0.1*torch.eye(Z.shape[1]),Z.T@codes[Y]);bias=codes[Y].mean(0)
 # Use nearest code label at inference. Store only model state and code table.
 pred=(Z@A).argmax(-1);acc=float((pred==Y).float().mean())
 state={'mean':mu,'scale':sd,'encoder':A,'codes':codes,'bias':bias};torch.save(state,ART/'development_encoder.pt')
 (ART/'development_smoke.json').write_text(json.dumps({'examples':len(Y),'classes':16,'code_dim':CODE,'train_accuracy':acc,'router_bytes':len(blob(state))},indent=2)+'\n');print('development',acc)
def main_fresh():
 m,t=load();state=torch.load(ART/'development_encoder.pt',map_location='cpu',weights_only=False);fvs=torch.stack([fv(m,t,q,52100+i) for i,q in enumerate(TASKS)]);rows=[];base=Path(snapshot_download(MODEL,revision=REV,tqdm_class=None));files=['model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json'];basebytes=sum((base/n).stat().st_size for n in files if (base/n).is_file())
 for w in FRESH:
  for s in SEEDS:
   contexts=torch.stack([hidden(m,t,q,w+s+i) for i,q in enumerate(TASKS)]);z=(contexts-state['mean'])/state['scale'];code=z@state['encoder']+state['bias'];pred=code.argmax(-1);true=torch.arange(16);routeacc=float((pred==true).float().mean())
   direct=forward(m,t,TASKS,direct=True);queryz=forward(m,t,TASKS);oracle=forward(m,t,TASKS,fvs);routed=forward(m,t,TASKS,fvs[pred]);tids=torch.tensor([t(' '+target(q),add_special_tokens=False)['input_ids'][0] for q in TASKS])
   def metric(o):p=o.softmax(-1);return float((o.argmax(-1)==tids).float().mean()),float(-p[torch.arange(16),tids].log().mean())
   packed=blob({'encoder':state['encoder'].half(),'mean':state['mean'].half(),'scale':state['scale'].half(),'codes':state['codes'].half(),'fv_bank':fvs.float(),'task_labels':TASKS,'layer':LAYER});path=PAY/f'{w}_{s}_compiled.pt';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(packed);sha=hashlib.sha256(packed).hexdigest();rel=str(path.relative_to(ROOT.parents[2]))
   for name,out,route in [('direct_icl',direct,None),('query_only',queryz,None),('oracle_fv',oracle,None),('compiled_mirror',routed,routeacc)]:
    a,n=metric(out);rows.append({'world':w,'seed':s,'method':name,'code_accuracy':'' if route is None else route,'accuracy':a,'mean_target_nll':n,'context_tokens':0 if name=='compiled_mirror' else 1,'payload_bytes':len(packed) if name=='compiled_mirror' else 0,'base_model_bytes':basebytes,'total_deployment_bytes':basebytes+(len(packed) if name=='compiled_mirror' else 0),'hash':sha if name=='compiled_mirror' else '', 'path':rel if name=='compiled_mirror' else ''})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print({'rows':len(rows),'base_model_bytes':basebytes})
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['development','fresh'],required=True);x=a.parse_args();train_dev() if x.phase=='development' else main_fresh()
