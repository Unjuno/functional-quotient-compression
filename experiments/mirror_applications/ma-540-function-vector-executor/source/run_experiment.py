#!/usr/bin/env python3
"""MA-540: tied sequential function-vector execution on affine bit permutations."""
from __future__ import annotations
import argparse,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];N=16;R=24;D=16;H=32

def apply_rule(x,perm,mask):
 bits=[(int(x)>>i)&1 for i in range(4)];out=0
 for j,src in enumerate(perm):out|=(bits[src]^((mask>>j)&1))<<j
 return out

def make_world(seed):
 rng=np.random.default_rng(seed);candidates=[(p,m) for p in __import__('itertools').permutations(range(4)) for m in range(16)];ix=rng.choice(len(candidates),R,replace=False);ops=[candidates[i] for i in ix];rules=np.array([tuple(op[0])+(int(op[1]),) for op in ops],np.int16);table=np.array([[apply_rule(x,*op) for x in range(N)] for op in ops],np.int64)
 groups=[(i,j) for i in range(R) for j in range(i+1,R)];rng.shuffle(groups);nh=max(1,round(.2*len(groups)));held=set(groups[:nh]);train=[];test=[]
 for i in range(R):
  for j in range(R):
   if i==j:continue
   (test if (min(i,j),max(i,j)) in held else train).append((i,j))
 return table,np.asarray(train,np.int16),np.asarray(test,np.int16),rules

class StepNet(nn.Module):
 def __init__(self):
  super().__init__();self.state=nn.Embedding(N,D);self.op=nn.Embedding(R,D);self.head=nn.Sequential(nn.Linear(D,H),nn.Tanh(),nn.Linear(H,N))
 def preactivation(self,x,op):return self.state(x.long())+self.op(op.long())
 def forward_ids(self,x,op):return self.head(torch.tanh(self.preactivation(x,op)))
 def extract_fv(self,support_states):
  # support states are [R,8]; each delta is h(s,op)-h(s,null-op)
  with torch.no_grad():
   ss=torch.tensor(support_states,dtype=torch.long);ids=torch.arange(R)[:,None].expand(-1,ss.shape[1]);return (self.preactivation(ss,ids)-self.state(ss)).mean(1)
 def forward_fv(self,x,code):return self.head(torch.tanh(self.state(x.long())+code))

class EndpointNet(nn.Module):
 def __init__(self):
  super().__init__();self.state=nn.Embedding(N,D);self.op=nn.Embedding(R,D);self.net=nn.Sequential(nn.Linear(2*D,H),nn.Tanh(),nn.Linear(H,N))
 def forward(self,x,a,b):return self.net(torch.cat((self.state(x.long()),self.op(a.long())+self.op(b.long())),dim=-1))

def make_pair_examples(table,pairs):
 xs=[];ys=[];aa=[];bb=[]
 for a,b in pairs:
  for x in range(N):xs.append(x);ys.append(table[b,table[a,x]]);aa.append(a);bb.append(b)
 return np.asarray(xs),np.asarray(aa),np.asarray(bb),np.asarray(ys)
def train_atom(table,updates,lr,seed):
 torch.manual_seed(seed);m=StepNet();opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);xs=np.tile(np.arange(N),R);ops=np.repeat(np.arange(R),N);ys=table.reshape(-1);g=np.random.default_rng(seed+19);start=time.perf_counter()
 for _ in range(updates):
  ix=g.integers(len(xs),size=96);x=torch.tensor(xs[ix]);o=torch.tensor(ops[ix]);y=torch.tensor(ys[ix]);loss=F.cross_entropy(m.forward_ids(x,o),y);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-start

def train_endpoint(table,pairs,updates,lr,seed):
 torch.manual_seed(seed);m=EndpointNet();opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);x,a,b,y=make_pair_examples(table,pairs);g=np.random.default_rng(seed+29);start=time.perf_counter()
 for _ in range(updates):
  ix=g.integers(len(x),size=96);tx=torch.tensor(x[ix]);ta=torch.tensor(a[ix]);tb=torch.tensor(b[ix]);ty=torch.tensor(y[ix]);loss=F.cross_entropy(m(tx,ta,tb),ty);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-start

def evaluate_seq(net,table,pairs,support_states,kind):
 states=np.arange(N);correct=atomic=0;total=0;rev_ok=0;rows=[];net.eval();fv=net.extract_fv(support_states)
 with torch.no_grad():
  for a,b in pairs:
   x=torch.tensor(states);t1=table[a,states];t2=table[b,t1]
   if kind=='fv':z=net.forward_fv(x,fv[a]);p1=z.argmax(-1);z2=net.forward_fv(p1,fv[b]);pred=z2.argmax(-1)
   else:p1=net.forward_ids(x,torch.full_like(x,a)).argmax(-1);pred=net.forward_ids(p1,torch.full_like(p1,b)).argmax(-1)
   ok=pred.numpy()==t2;correct+=int(ok.sum());total+=N;rows.append(float(ok.mean()))
  for a,b in pairs:
   x=torch.tensor(states);t1=table[b,states];target=table[a,t1]
   if kind=='fv':p=net.forward_fv(x,fv[b]).argmax(-1);pred=net.forward_fv(p,fv[a]).argmax(-1)
   else:p=net.forward_ids(x,torch.full_like(x,b)).argmax(-1);pred=net.forward_ids(p,torch.full_like(p,a)).argmax(-1)
   rev_ok+=int((pred.numpy()==target).sum())
 return {'ordered_pair_exact_accuracy':correct/total,'reverse_order_exact_accuracy':rev_ok/total,'heldout_ordered_pairs':len(pairs),'evaluated_states':total}

def evaluate_endpoint(m,table,pairs):
 x,a,b,y=make_pair_examples(table,pairs);m.eval();ok=rev=0
 with torch.no_grad():
  for lo in range(0,len(x),512):
   xx=torch.tensor(x[lo:lo+512]);aa=torch.tensor(a[lo:lo+512]);bb=torch.tensor(b[lo:lo+512]);yy=y[lo:lo+512];pred=m(xx,aa,bb).argmax(-1).numpy();ok+=int((pred==yy).sum())
   revpred=m(xx,bb,aa).argmax(-1).numpy();# Reversed target is second op then first
   revtarget=np.array([table[aa[k],table[bb[k],xx[k]]] for k in range(len(xx))]);rev+=int((revpred==revtarget).sum())
 return {'ordered_pair_exact_accuracy':ok/len(x),'reverse_order_exact_accuracy':rev/len(x),'heldout_ordered_pairs':len(pairs),'evaluated_states':len(x)}

def save_npz(path,arrays):
 np.savez(path,**arrays)
 with zipfile.ZipFile(path) as z:assert all(i.compress_type==zipfile.ZIP_STORED for i in z.infolist())
 return Path(path).stat().st_size
def save_step_payload(path,net,support,table,mode):
 sd={k:v.detach().cpu().numpy() for k,v in net.state_dict().items()}
 if mode=='fv':
  fv=net.extract_fv(support);sd={'state.weight':sd['state.weight'],'head.0.weight':sd['head.0.weight'],'head.0.bias':sd['head.0.bias'],'head.2.weight':sd['head.2.weight'],'head.2.bias':sd['head.2.bias'],'function_vectors':fv.numpy()}
 sd.update({'support_states':support.astype(np.uint8),'schema':np.array([540,2,N,D],np.int32)})
 return save_npz(path,sd)
def run(seed,outdir,settings):
 out=Path(outdir);out.mkdir(parents=True,exist_ok=True);table,train_pairs,test_pairs,rules=make_world(seed);rng=np.random.default_rng(seed+101);support=np.stack([rng.choice(N,8,replace=False) for _ in range(R)]);np.savez(out/'world.npz',table=table.astype(np.uint8),rules=rules.astype(np.uint8),support_states=support.astype(np.uint8),train_pairs=train_pairs,test_pairs=test_pairs)
 rows=[]
 for updates,lr in settings:
  net,atom_s=train_atom(table,updates,lr,seed+updates+int(lr*100000));endpoint,endpoint_s=train_endpoint(table,train_pairs,updates,lr,seed+updates+int(lr*100000)+301)
  # Verify FV extraction exactly substitutes for ordinary op embeddings before scoring.
  fv=net.extract_fv(support);xx=torch.arange(N).repeat(R);oo=torch.arange(R).repeat_interleave(N);base=torch.max(torch.abs(net.forward_ids(xx,oo)-net.forward_fv(xx,fv[oo]))).item()
  controls={'fv_tied_sequential':('fv',net,atom_s),'native_operator_id_tied':('ids',net,atom_s),'external_two_call':('ids',net,atom_s)}
  for name,(mode,model,sec) in controls.items():
   metrics=evaluate_seq(model,table,test_pairs,support,mode);payload=save_step_payload(out/f'{name}_u{updates}_lr{lr}.npz',model,support,table,mode);rows.append({'world':seed,'method':name,'updates':updates,'lr':lr,'payload_bytes':payload,'steps_per_pair':2,'fit_seconds':sec,'compute_proxy':int(updates*96*R*N*D*3),'metrics':metrics,'fv_id_max_logit_difference':base if mode=='fv' else None})
  # Two untied copies, initialized from the same atomic solution, represent step-specific storage.
  states={k:v.detach().cpu().numpy() for k,v in net.state_dict().items()};untied_bytes=save_npz(out/f'untied_two_step_u{updates}_lr{lr}.npz',{'step1_'+k:v for k,v in states.items()}|{'step2_'+k:v for k,v in states.items()}|{'schema':np.array([540,2,N,D],np.int32)});um=evaluate_seq(net,table,test_pairs,support,'ids');rows.append({'world':seed,'method':'untied_two_step','updates':updates,'lr':lr,'payload_bytes':untied_bytes,'steps_per_pair':2,'fit_seconds':2*atom_s,'compute_proxy':2*updates*96*R*N*D*3,'metrics':um,'fv_id_max_logit_difference':None})
  em=evaluate_endpoint(endpoint,table,test_pairs);epbytes=save_npz(out/f'endpoint_sum_u{updates}_lr{lr}.npz',{**{k:v.detach().cpu().numpy() for k,v in endpoint.state_dict().items()},'schema':np.array([540,1,N,D],np.int32)});rows.append({'world':seed,'method':'endpoint_sum','updates':updates,'lr':lr,'payload_bytes':epbytes,'steps_per_pair':1,'fit_seconds':endpoint_s,'compute_proxy':int(updates*96*R*R*N*D*3),'metrics':em,'fv_id_max_logit_difference':None})
 # Exact structured table upper, charged as a complete 24x16 map bank.
 upper=save_npz(out/'exact_affine_table_upper.npz',{'function_tables':table.astype(np.uint8),'rules':rules.astype(np.uint8),'schema':np.array([540,R,N],np.int32)});rows.append({'world':seed,'method':'exact_affine_table_upper','updates':0,'lr':0,'payload_bytes':upper,'steps_per_pair':2,'fit_seconds':0,'compute_proxy':0,'metrics':{'ordered_pair_exact_accuracy':1.,'reverse_order_exact_accuracy':1.,'heldout_ordered_pairs':len(test_pairs),'evaluated_states':len(test_pairs)*N},'fv_id_max_logit_difference':None})
 (out/'metrics.json').write_text(json.dumps({'experiment_id':'MA-540','world':seed,'settings':settings,'atomic_functions':R,'train_pairs':len(train_pairs),'heldout_pairs':len(test_pairs),'rows':rows},indent=2,sort_keys=True)+'\n');print(json.dumps({'world':seed,'metrics':[{k:r[k] for k in ('method','updates','lr','payload_bytes','metrics','fv_id_max_logit_difference')} for r in rows]},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--settings',default='800:0.001,800:0.003,1600:0.001,1600:0.003');a=p.parse_args();run(a.seed,a.out,[(int(x.split(':')[0]),float(x.split(':')[1])) for x in a.settings.split(',')])
