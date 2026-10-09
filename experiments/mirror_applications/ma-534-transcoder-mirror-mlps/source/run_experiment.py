#!/usr/bin/env python3
"""Run frozen MA-534 support-adapted logical MLP experiment."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time, zipfile
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma534_harness',Path(__file__).with_name('shared_fv_harness.py'))
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
WIDTH=2048; TOPK=32; POOL=16; FIT=np.arange(12); HELD=np.arange(12,16)

def sparse_encode(x,E,eb):
 z=np.maximum(x@E.T+eb,0)
 k=min(TOPK,z.shape[-1]);ix=np.argpartition(z,-k,axis=-1)[:,-k:]
 out=np.zeros_like(z,dtype=np.float32);np.put_along_axis(out,ix,np.take_along_axis(z,ix,axis=-1),axis=-1)
 return out

def decode(z,D,db): return z@D+db

def apply_view(z,code,kind):
 q=z[..., :POOL].copy()
 if kind=='givens':
  for p,a in enumerate(code):
   i=2*p;j=i+1;c=np.cos(a);s=np.sin(a);u=q[...,i].copy();v=q[...,j].copy()
   q[...,i]=c*u-s*v;q[...,j]=s*u+c*v
 elif kind=='pairwise':
  for p,g in enumerate(code): q[...,2*p:2*p+2]*=g
 elif kind=='elementwise': q*=code
 else: raise ValueError(kind)
 return q

def delta_from_code(x,E,eb,D,db,code,kind):
 z=sparse_encode(np.asarray(x,dtype=np.float32),E,eb)[...,:POOL]
 return decode(apply_view(z,code,kind),D,db)

def capture(model,tok,torchmod,text):
 layer=model.gpt_neox.layers[3]; ids=h.enc(tok,text); got={}
 def xhook(mod,ins,out): got['x']=(out[0] if isinstance(out,tuple) else out).detach()[0,-1].cpu().numpy().astype(np.float32)
 def yhook(mod,ins,out): got['y']=(out[0] if isinstance(out,tuple) else out).detach()[0,-1].cpu().numpy().astype(np.float32)
 a=layer.post_attention_layernorm.register_forward_hook(xhook);b=layer.mlp.register_forward_hook(yhook)
 try:
  with torchmod.no_grad():model(input_ids=torchmod.tensor([ids],dtype=torchmod.long),use_cache=False)
 finally:a.remove();b.remove()
 return got['x'],got['y'],len(ids)

def capture_basis_data(model,tok,torchmod,manifest):
 xs=[];ys=[];n=0
 for tid in FIT:
  support=manifest[int(tid)]['support']
  for x,_ in support:
   demos=[p for p in support if p[0]!=x]
   ids=h.enc(tok,h.prompt(x,demos));got={};layer=model.gpt_neox.layers[3]
   def xhook(mod,ins,out):got['x']=(out[0] if isinstance(out,tuple) else out).detach()[0].cpu().numpy().astype(np.float32)
   def yhook(mod,ins,out):got['y']=(out[0] if isinstance(out,tuple) else out).detach()[0].cpu().numpy().astype(np.float32)
   a=layer.post_attention_layernorm.register_forward_hook(xhook);b=layer.mlp.register_forward_hook(yhook)
   try:
    with torchmod.no_grad():model(input_ids=torchmod.tensor([ids],dtype=torchmod.long),use_cache=False)
   finally:a.remove();b.remove()
   xs.append(got['x']);ys.append(got['y']);n+=len(ids)
 return np.concatenate(xs),np.concatenate(ys),n

def train_basis(X,Y):
 torch.manual_seed(0);x=torch.tensor(X);y=torch.tensor(Y);enc=torch.nn.Linear(512,WIDTH);dec=torch.nn.Linear(WIDTH,512)
 opt=torch.optim.Adam([*enc.parameters(),*dec.parameters()],lr=.001,weight_decay=0);t=time.perf_counter()
 for _ in range(1000):
  ix=torch.randint(len(x),(128,));xb=x[ix];yb=y[ix];opt.zero_grad();z=torch.relu(enc(xb));v,ii=torch.topk(z,TOPK,dim=-1,sorted=False);z=torch.zeros_like(z).scatter(-1,ii,v);pred=dec(z)
  loss=((pred-yb)**2).mean()+.001*z.abs().mean();loss.backward();opt.step()
 e=enc.weight.detach().cpu().numpy().astype(np.float32);eb=enc.bias.detach().cpu().numpy().astype(np.float32);d=dec.weight.detach().cpu().numpy().T.astype(np.float32);db=dec.bias.detach().cpu().numpy().astype(np.float32)
 return e,eb,d,db,time.perf_counter()-t

def fit_roles(examples,E,eb,D,db,kind,updates=500):
 outputs=[];codes=[];seconds=[];losses=[]
 for role,items in enumerate(examples):
  x=torch.tensor(np.stack([q[0] for q in items]),dtype=torch.float32);y=torch.tensor(np.stack([q[1] for q in items]),dtype=torch.float32)
  Et=torch.tensor(E);ebt=torch.tensor(eb);Dt=torch.tensor(D.T.copy());dbt=torch.tensor(db)
  with torch.no_grad():
   # Exact deployment slice: selected 16 encoder rows only, with direct ReLU activations.
   z=torch.relu(torch.nn.functional.linear(x,Et,ebt))
  shape=8 if kind in ('givens','pairwise') else 16
  raw=torch.nn.Parameter(torch.zeros(shape));opt=torch.optim.Adam([raw],lr=.01);start=time.perf_counter()
  for _ in range(updates):
   opt.zero_grad()
   code=raw if kind!='givens' else torch.tanh(raw)*np.pi
   q=z.clone()
   if kind=='givens':
    for p in range(8):
     a=code[p];i=2*p;j=i+1;u=q[:,i].clone();v=q[:,j].clone();q[:,i]=torch.cos(a)*u-torch.sin(a)*v;q[:,j]=torch.sin(a)*u+torch.cos(a)*v
   elif kind=='pairwise':
    for p in range(8):q[:,2*p:2*p+2]*=code[p]
   else:q*=code
   pred=torch.nn.functional.linear(q,Dt,dbt);loss=((pred-y)**2).mean();loss.backward();opt.step()
  c=(torch.tanh(raw)*np.pi if kind=='givens' else raw).detach().numpy().astype(np.float32)
  codes.append(c);outputs.append((z.numpy(),y.numpy()));seconds.append(time.perf_counter()-start);losses.append(float(loss.item()))
 return np.stack(codes),outputs,seconds,losses

def score_one(model,tok,torchmod,query,candidates,role,method,params):
 prefix=h.prompt(query);pids=h.enc(tok,prefix);seqs=[];targets=[]
 for answer in candidates:
  full=h.enc(tok,prefix+' '+answer)
  if full[:len(pids)]!=pids:raise ValueError('candidate tokenization changed frozen prompt prefix')
  seqs.append(full);targets.append(full[len(pids):])
 mx=max(map(len,seqs));pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
 ids=torch.full((len(seqs),mx),pad,dtype=torch.long);mask=torch.zeros_like(ids)
 for i,s in enumerate(seqs):ids[i,:len(s)]=torch.tensor(s);mask[i,:len(s)]=1
 layer=model.gpt_neox.layers[3].mlp
 if method=='explicit_mean':means=params['means']
 else:E,eb,D,db,codes=params
 def hook(mod,ins,out):
  xx=ins[0];yy=out[0] if isinstance(out,tuple) else out;yy=yy.clone();pos=len(pids)-1
  if method=='none':delta=np.zeros((1,yy.shape[-1]),dtype=np.float32)
  elif method=='explicit_mean':delta=means[role]
  elif method=='unmodulated':delta=decode(sparse_encode(xx[:,pos,:].detach().cpu().numpy(),E,eb),D,db)
  else:delta=delta_from_code(xx[:,pos,:].detach().cpu().numpy(),E,eb,D,db,codes[role],method)
  yy[:,pos,:]+=torch.as_tensor(delta,dtype=yy.dtype,device=yy.device)
  return (yy,)+tuple(out[1:]) if isinstance(out,tuple) else yy
 handle=layer.register_forward_hook(hook)
 try:
  with torchmod.no_grad():logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
 finally:handle.remove()
 lp=torch.log_softmax(logits,dim=-1);scores=[]
 for i,ans in enumerate(targets):scores.append(sum(float(lp[i,len(pids)+j-1,tokid].item()) for j,tokid in enumerate(ans)))
 return np.asarray(scores),len(seqs),sum(map(len,seqs))

def evaluate_method(model,tok,torchmod,manifest,method,params):
 gold=[];correct=0;total=0;margins=[];seqs=0;tokens=0;bytask=[]
 for tid in HELD:
  ev=manifest[int(tid)]['evaluation'];cand=sorted({y for _,y in ev});ys=[];cs=[];ms=[]
  for x,y in ev:
   scores,n,ntok=score_one(model,tok,torchmod,x,cand,int(tid),method,params);gi=cand.index(y);ys.append(scores[gi]);cs.append(int(np.argmax(scores)==gi));ms.append(scores[gi]-np.max(np.delete(scores,gi)));seqs+=n;tokens+=ntok
  correct+=sum(cs);total+=len(ev);gold+=ys;margins+=ms;bytask.append({'task_id':int(tid),'accuracy':float(np.mean(cs)),'gold_logprob':float(np.mean(ys)),'margin':float(np.mean(ms))})
 return {'heldout_accuracy':correct/total,'mean_gold_candidate_logprob':float(np.mean(gold)),'mean_gold_vs_best_negative_margin':float(np.mean(margins)),'queries':total,'candidate_sequences':seqs,'candidate_input_tokens':tokens,'by_task':bytask}

def choose_pool(targets,D):
 fit=np.concatenate([np.stack([v for v in row]) for row in targets[:12]],axis=0);norm=np.linalg.norm(D,axis=1);dn=D/np.maximum(norm[:,None],1e-12);r=fit.copy();used=np.zeros(len(D),bool);ids=[]
 for _ in range(POOL):
  c=np.abs(r@dn.T).sum(0);c[used]=-np.inf;j=int(c.argmax());ids.append(j);used[j]=True;coef=(r@dn[j])/max(float(norm[j]),1e-12);r-=coef[:,None]*D[j]
 return np.array(ids,dtype=np.int16)

def payload(path,arrays):
 np.savez(path,**arrays)
 with zipfile.ZipFile(path) as z:
  assert all(q.compress_type==zipfile.ZIP_STORED for q in z.infolist())
 return Path(path).stat().st_size

def run(seed,model_dir,outdir):
 run_started=time.perf_counter();out=Path(outdir);out.mkdir(parents=True,exist_ok=True);model,tok,tm=h.load_model(model_dir)
 manifest=[]
 for i,(name,_) in enumerate(h.TASKS):
  sup,ev=h.split_task(i,seed);manifest.append({'task_id':i,'name':name,'support':sup,'evaluation':ev})
 t=time.perf_counter();X,Y,tokens_basis=capture_basis_data(model,tok,tm,manifest);capture_s=time.perf_counter()-t
 E,eb,Dfull,db,fit_s=train_basis(X,Y)
 # Fit targets and query inputs for every role, including held-out roles; query examples remain unused here.
 targets=[[] for _ in range(16)];xs=[[] for _ in range(16)];forward_tokens=0
 for tid,task in enumerate(manifest):
  for x,y in task['support']:
   bare_x,bare_y,n1=capture(model,tok,tm,h.prompt(x));demos=[p for p in task['support'] if p[0]!=x];full_x,full_y,n2=capture(model,tok,tm,h.prompt(x,demos));xs[tid].append(bare_x);targets[tid].append(full_y-bare_y);forward_tokens+=n1+n2
 pool=choose_pool(targets,Dfull);E=E[pool];eb=eb[pool];D=Dfull[pool]
 means=np.stack([np.mean(np.stack(t),axis=0) for t in targets]).astype(np.float32)
 fitdata={}
 for kind in ('givens','pairwise','elementwise'):
  codes,_,secs,losses=fit_roles([[ (xs[i][j],targets[i][j]) for j in range(8)] for i in range(16)],E,eb,D,db,kind)
  fitdata[kind]=(codes,secs,losses)
 model_bytes=sum((Path(model_dir)/n).stat().st_size for n in ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json'])
 meta={'schema':np.array([534,WIDTH,TOPK,POOL],np.int32),'pool_global_ids':pool,'task_ids':np.arange(16,dtype=np.int16),'encoder_rows':E,'encoder_bias':eb,'decoder_rows':D,'decoder_bias':db,'model_revision':np.frombuffer(h.REVISION.encode(),dtype='S40'),'model_sha256':np.frombuffer(h.MODEL_SHA.encode(),dtype='S64')}
 explicit_b=payload(out/'explicit_mean_deltas.npz',{'mean_deltas':means,'task_ids':np.arange(16,dtype=np.int16),'schema':np.array([534,512,16],np.int32)})
 rows=[]
 for method in ('none','explicit_mean','unmodulated','pairwise','elementwise','givens'):
  codes=fitdata[method][0] if method in fitdata else np.zeros((16,8),np.float32)
  if method=='none': b=0;params=(E,eb,D,db,codes);evalmethod='none'
  elif method=='explicit_mean':b=explicit_b;params={'means':means,'unused':meta};evalmethod='explicit_mean'
  elif method=='unmodulated':b=payload(out/(method+'.npz'),{**meta});params=(E,eb,D,db,codes);evalmethod='unmodulated'
  else:
   c=codes
   arr={'codes':c,**meta}
   b=payload(out/(method+'.npz'),arr);params=(E,eb,D,db,c);evalmethod=method
  eval_started=time.perf_counter()
  if method=='none':
   params=(E,eb,D,db,codes);m=evaluate_method(model,tok,tm,manifest,'none',params)
  else:m=evaluate_method(model,tok,tm,manifest,evalmethod,params)
  eval_seconds=time.perf_counter()-eval_started
  charged_basis=(E.nbytes+eb.nbytes+D.nbytes+db.nbytes if method not in ('none','explicit_mean') else 0)
  codebytes=(b-charged_basis if b else 0)
  rows.append({'seed':seed,'method':method,'payload_bytes':b,'basis_bytes':int(charged_basis),'code_bytes':int(codebytes),'model_bytes':model_bytes,'total_deployment_bytes':model_bytes+b,'explicit_total_bytes':model_bytes+explicit_b,'support_examples':128,'support_capture_tokens':forward_tokens,'transcoder_training_tokens':tokens_basis,'transcoder_optimizer_updates':1000,'role_optimizer_updates':8000 if method in fitdata else 0,'active_feature_compute_proxy':int((32*512*2048*1000*3 if method not in ('none','explicit_mean') else 0)+(16*8*512*2048*500 if method in fitdata else 0)),'transcoder_capture_seconds':capture_s,'transcoder_fit_seconds':fit_s,'role_fit_seconds':fitdata.get(method,(None,[0]*16,None))[1],'evaluation_seconds':eval_seconds,'metrics':m})
 report={'experiment_id':'MA-534','seed':seed,'total_wall_seconds':time.perf_counter()-run_started,'revision':h.REVISION,'model_sha256':h.MODEL_SHA,'transcoder_encoder_sha256':hashlib.sha256(E.tobytes()+eb.tobytes()).hexdigest(),'transcoder_decoder_sha256':hashlib.sha256(Dfull.tobytes()+db.tobytes()).hexdigest(),'pool_ids':pool.tolist(),'basis_training_tokens':tokens_basis,'transcoder_fvu':None,'rows':rows}
 (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');(out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'seed':seed,'pool':pool.tolist(),'rows':[{'method':r['method'],'bytes':r['payload_bytes'],'acc':r['metrics']['heldout_accuracy'],'gold':r['metrics']['mean_gold_candidate_logprob']} for r in rows]},indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.seed,a.model_dir,a.out)
