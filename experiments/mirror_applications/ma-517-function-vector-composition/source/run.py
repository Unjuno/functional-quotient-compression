import csv,hashlib,io,json,statistics,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';MODEL='openai-community/gpt2';REV='607a30d783dfa663caf39e06633721c8d4cfcd7e';DEV=[51700,51701];FRESH=[51710,51711,51712];SEEDS=[0,1,2];T=16;KEYS=['apple','banana','cherry','lemon','orange','peach','pear','plum'];COLORS=['red','blue','green','yellow','purple','black','white','pink'];SHAPES=['circle','square','triangle','diamond','oval','star','heart','sphere'];torch.set_num_threads(4)
def load():
 tok=AutoTokenizer.from_pretrained(MODEL,revision=REV);tok.padding_side='left';tok.pad_token=tok.eos_token;model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,torch_dtype=torch.float32);model.eval();return model,tok
def tasks(w,s):
 g=torch.Generator().manual_seed(w*100003+s*7919+83);out=[]
 for _ in range(T):
  p1=torch.randperm(8,generator=g).tolist();p2=torch.randperm(8,generator=g).tolist();a={KEYS[i]:COLORS[p1[i]] for i in range(8)};b={COLORS[i]:SHAPES[p2[i]] for i in range(8)}
  d1=''.join(f"The fruit {k} has color {a[k]}.\n" for k in KEYS[:4]); mids=[a[k] for k in KEYS[4:]];d2=''.join(f"The color {c} has shape {b[c]}.\n" for c in mids);qs=[(k,b[a[k]]) for k in KEYS[4:]];out.append((d1,d2,qs))
 return out
def batch(tok,txt):return tok(txt,return_tensors='pt',padding=True,add_special_tokens=False)
def delta(model,tok,head,demos,layer):
 pref=f'{head} mapping:\n';prompts=[pref+'Apply:',pref+demos+'Apply:'];x=batch(tok,prompts)
 with torch.no_grad():o=model(**x,output_hidden_states=True,return_dict=True)
 return (o.hidden_states[layer+1][1,-1]-o.hidden_states[layer+1][0,-1]).detach()
def logits(model,tok,prompts,views=None):
 x=batch(tok,prompts);handles=[]
 for layer,v in (views or {}).items():
  def hook(_m,_i,out,v=v):
   h=out[0] if isinstance(out,tuple) else out;h=h.clone();h[:,-1,:]+=v.to(h.device).unsqueeze(0);return (h,)+out[1:] if isinstance(out,tuple) else h
  handles.append(model.transformer.h[int(layer)].register_forward_hook(hook))
 try:
  with torch.no_grad():return model(**x,return_dict=True).logits[:,-1,:]
 finally:
  for h in handles:h.remove()
def build_task_vectors(model,tok,d1,d2):
 return {'f1_4':delta(model,tok,'Fruit-color',d1,4),'f2_8':delta(model,tok,'Color-shape',d2,8),'f1_6':delta(model,tok,'Fruit-color',d1,6),'f2_6':delta(model,tok,'Color-shape',d2,6)}
def evaluate(model,tok,w,s):
 ts=tasks(w,s);vecs=[];queries=[];icl=[];targets=[];owners=[]
 t0=time.perf_counter()
 for i,(d1,d2,qs) in enumerate(ts):
  v=build_task_vectors(model,tok,d1,d2);vecs.append(v)
  for k,sh in qs:
   q=f'Given the fruit {k}, its shape is';queries.append(q);icl.append('Fruit-color mapping:\n'+d1+'Color-shape mapping:\n'+d2+q);targets.append(tok(' '+sh,add_special_tokens=False)['input_ids'][0]);owners.append(i)
 extract_sec=time.perf_counter()-t0;target=torch.tensor(targets);owners=torch.tensor(owners);metrics=[];payloads=[]
 def assess(method,rank,views_by_task):
  outs=[];tstart=time.perf_counter()
  for i in range(T):
   ix=(owners==i).nonzero().flatten();outs.append((ix,logits(model,tok,[queries[j] for j in ix.tolist()],views_by_task[i])))
  z=torch.empty((len(target),model.config.vocab_size))
  for ix,val in outs:z[ix]=val
  p=z.softmax(-1);acc=float((z.argmax(-1)==target).float().mean());nll=float(-p[torch.arange(len(target)),target].log().mean());metrics.append({'method':method,'rank':rank,'accuracy':acc,'mean_target_nll':nll,'inference_seconds':time.perf_counter()-tstart})
 t0=time.perf_counter();direct=logits(model,tok,icl);direct_sec=time.perf_counter()-t0;t0=time.perf_counter();base=logits(model,tok,queries);base_sec=time.perf_counter()-t0
 for name,z,elapsed in [('query_only',base,base_sec),('direct_icl',direct,direct_sec)]:
  p=z.softmax(-1);metrics.append({'method':name,'rank':0,'accuracy':float((z.argmax(-1)==target).float().mean()),'mean_target_nll':float(-p[torch.arange(len(target)),target].log().mean()),'inference_seconds':elapsed})
 for name in ['raw_sum','factorized','f1_only','f2_only']:
  views=[]
  for v in vecs:
   if name=='raw_sum':views.append({6:v['f1_6']+v['f2_6']})
   elif name=='factorized':views.append({4:v['f1_4'],8:v['f2_8']})
   elif name=='f1_only':views.append({6:v['f1_6']})
   else:views.append({6:v['f2_6']})
  assess(name,0,views)
 # Explicit payloads charge both task vectors for composition; one vector for ablation.
 pairs={m:[] for m in ['raw_sum','factorized','f1_only','f2_only']}
 for v in vecs:
  pairs['raw_sum'].append(torch.stack([v['f1_6'],v['f2_6']]));pairs['factorized'].append(torch.stack([v['f1_4'],v['f2_8']]));pairs['f1_only'].append(v['f1_6']);pairs['f2_only'].append(v['f2_6'])
 for m,vs in pairs.items():
  o={'method':m,'vectors':torch.stack(vs).contiguous(),'layers':[6,6] if m=='raw_sum' else [4,8] if m=='factorized' else [6]};b=io.BytesIO();torch.save(o,b);blob=b.getvalue();payloads.append((m,blob))
 return metrics,payloads,extract_sec,vecs

def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True);model,tok=load()
 if phase=='development':
  sample=tasks(DEV[0],SEEDS[0])[0];d1,d2,qs=sample;vs=build_task_vectors(model,tok,d1,d2)
  assert all(len(tok(' '+shape,add_special_tokens=False)['input_ids'])==1 for _,shape in qs)
  (ART/'development_smoke.json').write_text(json.dumps({'task_count':1,'function_vector_dim':int(vs['f1_4'].numel()),'layers':[4,6,8],'target_tokens_single_token':True},indent=2)+'\n')
  print('development mapping/vector smoke passed; no model or layer selection');return
 base_dir=Path(snapshot_download(MODEL,revision=REV,tqdm_class=None));names=['model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json'];base_bytes=sum((base_dir/n).stat().st_size for n in names if (base_dir/n).is_file());rows=[]
 for w in FRESH:
  for s in SEEDS:
   ms,ps,extract,vecs=evaluate(model,tok,w,s);pvt=torch.stack([v['f1_6']+v['f2_6'] for v in vecs]);sim=torch.nn.functional.normalize(pvt,dim=1)@torch.nn.functional.normalize(pvt,dim=1).T;mean_sim=float((sim.sum()-T)/(T*(T-1)))
   for m,b in ps:
    path=PAY/f'{w}_{s}_{m}.pt';path.write_bytes(b);met=next(x for x in ms if x['method']==m);rows.append({'world':w,'seed':s,'method':m,'rank':0,'accuracy':met['accuracy'],'mean_target_nll':met['mean_target_nll'],'mean_intertask_vector_cosine':mean_sim,'vector_extraction_seconds_per_bank':extract,'inference_seconds':met['inference_seconds'],'payload_bytes':len(b),'base_model_bytes':base_bytes,'total_deployment_bytes':base_bytes+len(b),'hash':hashlib.sha256(b).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
   for m in ms:
    if m['method'] in ('query_only','direct_icl'):rows.append({'world':w,'seed':s,'method':m['method'],'rank':0,'accuracy':m['accuracy'],'mean_target_nll':m['mean_target_nll'],'mean_intertask_vector_cosine':mean_sim,'vector_extraction_seconds_per_bank':0.,'inference_seconds':0.,'payload_bytes':0,'base_model_bytes':base_bytes,'total_deployment_bytes':base_bytes,'hash':'','path':''})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'rows':len(rows),'base_model_bytes':base_bytes}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
