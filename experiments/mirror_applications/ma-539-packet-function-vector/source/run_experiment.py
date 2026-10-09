#!/usr/bin/env python3
"""Frozen MA-539 synthetic packet function-vector screen."""
from __future__ import annotations
import argparse,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];SLOTS=4;STATES=32;FUNCTIONS=16;WIDTH=16;HIDDEN=32
METHODS=('none','support_extracted_fv','generic_packet_latent','ptp_slot_codes')

def make_world(seed):
 rng=np.random.default_rng(seed);maps=np.stack([rng.permutation(STATES) for _ in range(FUNCTIONS)]).astype(np.int64);sup=[];que=[]
 for i in range(FUNCTIONS):
  order=rng.permutation(STATES);a,b=order[:8],order[8:];sup.append(np.stack((a,maps[i,a]),-1));que.append(np.stack((b,maps[i,b]),-1))
 return maps,np.stack(sup),np.stack(que)
def packets(pairs,seed):
 rng=np.random.default_rng(seed);xs=[];ys=[]
 for row in pairs:
  order=rng.permutation(len(row));x,y=row[order,0].reshape(-1,SLOTS),row[order,1].reshape(-1,SLOTS);xs.append(x);ys.append(y)
 return np.stack(xs),np.stack(ys)
class PacketNet(nn.Module):
 def __init__(self,method):
  super().__init__();self.method=method;self.xemb=nn.Embedding(STATES,WIDTH);self.yemb=nn.Embedding(STATES,WIDTH);self.phase=nn.Parameter(torch.randn(SLOTS,WIDTH)*.02)
  if method=='support_extracted_fv':self.pair_encoder=nn.Linear(2*WIDTH,WIDTH)
  if method=='generic_packet_latent':self.codes=nn.Parameter(torch.randn(FUNCTIONS,WIDTH)*.02)
  if method=='ptp_slot_codes':self.slot_codes=nn.Parameter(torch.randn(FUNCTIONS,SLOTS,WIDTH)*.02)
  self.inp=nn.Linear(2*WIDTH,HIDDEN);self.mid=nn.Linear(HIDDEN,HIDDEN);self.out=nn.Linear(HIDDEN,STATES)
 def function_codes(self,support):
  if self.method=='support_extracted_fv':
   x=self.xemb(support[:,:,0].long());y=self.yemb(support[:,:,1].long());return torch.tanh(self.pair_encoder(torch.cat((x,y),-1))).mean(1)
  if self.method=='generic_packet_latent':return self.codes
  if self.method=='ptp_slot_codes':return self.slot_codes
  return torch.zeros(FUNCTIONS,WIDTH,device=self.phase.device)
 def forward(self,fn,states,support):
  code=self.function_codes(support)[fn];x=self.xemb(states.long())
  if code.ndim==3:c=code
  else:c=code[:,None,:].expand(-1,SLOTS,-1)
  c=c+self.phase[None];h=torch.tanh(self.inp(torch.cat((x,c),-1)));return self.out(F.gelu(self.mid(h)))
def batch_at(x,y,seed,step,batch=64):
 rng=np.random.default_rng(seed+step*7919);f=rng.integers(FUNCTIONS,size=batch);p=rng.integers(x.shape[1],size=batch);return f,x[f,p],y[f,p]
def train(method,support,x,y,lr,updates,seed):
 torch.manual_seed(seed);m=PacketNet(method);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);sup=torch.tensor(support,dtype=torch.long);start=time.perf_counter();m.train()
 for step in range(updates):
  f,xx,yy=batch_at(x,y,seed,step);f=torch.tensor(f);xx=torch.tensor(xx);yy=torch.tensor(yy);logits=m(f,xx,sup);loss=F.cross_entropy(logits.reshape(-1,STATES),yy.reshape(-1));opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-start
def evaluate(m,support,x,y,maps):
 sup=torch.tensor(support,dtype=torch.long);gold=[];ok=joint=valid=0;npacks=0;ntok=0;m.eval()
 with torch.no_grad():
  for fi in range(FUNCTIONS):
   xx=torch.tensor(x[fi],dtype=torch.long);yy=torch.tensor(y[fi],dtype=torch.long);f=torch.full((len(xx),),fi);logits=m(f,xx,sup);lp=F.log_softmax(logits,-1);gold.extend(lp.gather(-1,yy[...,None]).squeeze(-1).flatten().tolist());pred=logits.argmax(-1);match=pred.eq(yy);ok+=int(match.sum());joint+=int(match.all(-1).sum());valid+=int((torch.from_numpy(maps[fi])[xx]==pred).all(-1).sum());npacks+=len(xx);ntok+=yy.numel()
 return {'joint_packet_accuracy':joint/npacks,'token_accuracy':ok/ntok,'token_nll':-float(np.mean(gold)),'valid_path_rate':valid/npacks,'packets':npacks,'tokens':ntok}
def macs(method,updates):
 per=SLOTS*(2*WIDTH*HIDDEN+HIDDEN*HIDDEN+HIDDEN*STATES)*3;v=updates*64*per+96*per
 if method=='support_extracted_fv':v+=updates*64*8*2*WIDTH*WIDTH
 return int(v)
def serialize(path,method,model,support):
 arr={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}
 if method=='support_extracted_fv':
  with torch.no_grad():arr['extracted_function_vectors']=model.function_codes(torch.tensor(support,dtype=torch.long)).cpu().numpy()
 arr['function_ids']=np.arange(FUNCTIONS,dtype=np.uint8);arr['schema']=np.array([539,SLOTS,STATES,WIDTH],np.int32);np.savez(path,**arr)
 with zipfile.ZipFile(path) as z:assert all(q.compress_type==zipfile.ZIP_STORED for q in z.infolist())
 return Path(path).stat().st_size
def run(world,outdir,settings):
 out=Path(outdir);out.mkdir(parents=True,exist_ok=True);maps,support,query=make_world(world);tx,ty=packets(support,world+101);qx,qy=packets(query,world+303);rows=[]
 np.savez(out/'world_split.npz',support=support,query=query,maps_sha256=np.frombuffer(hashlib.sha256(maps.tobytes()).hexdigest().encode(),dtype='S64'))
 for updates,lr in settings:
  for idx,method in enumerate(METHODS):
   model,elapsed=train(method,support,tx,ty,lr,updates,world+idx*313+updates);ev0=time.perf_counter();metric=evaluate(model,support,qx,qy,maps);eval_s=time.perf_counter()-ev0;size=serialize(out/f'{method}_u{updates}_lr{lr}.npz',method,model,support);rows.append({'world':world,'method':method,'updates':updates,'learning_rate':lr,'payload_bytes':size,'address_bits_per_packet':(0 if method=='none' else (16 if method=='ptp_slot_codes' else 4)),'fit_seconds':elapsed,'evaluation_seconds':eval_s,'total_seconds':elapsed+eval_s,'active_compute_proxy':macs(method,updates),'metrics':metric})
 upper=out/'independent_function_table.npz';np.savez(upper,function_tables=maps.astype(np.uint8),schema=np.array([539,FUNCTIONS,STATES],np.int32));rows.append({'world':world,'method':'independent_function_table','updates':0,'learning_rate':0,'payload_bytes':upper.stat().st_size,'address_bits_per_packet':4,'fit_seconds':0,'evaluation_seconds':0,'total_seconds':0,'active_compute_proxy':0,'metrics':{'joint_packet_accuracy':1.,'token_accuracy':1.,'token_nll':0.,'valid_path_rate':1.,'packets':96,'tokens':384}})
 report={'experiment_id':'MA-539','world':world,'functions':FUNCTIONS,'support_examples_per_function':8,'query_examples_per_function':24,'packets':96,'settings':settings,'rows':rows};(out/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'world':world,'rows':[{'method':r['method'],'updates':r['updates'],'lr':r['learning_rate'],'bytes':r['payload_bytes'],'address_bits':r['address_bits_per_packet'],'metrics':r['metrics']} for r in rows]},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--world',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--settings',default='800:0.001,800:0.003,1200:0.001,1200:0.003');a=p.parse_args();run(a.world,a.out,[(int(x.split(':')[0]),float(x.split(':')[1])) for x in a.settings.split(',')])
