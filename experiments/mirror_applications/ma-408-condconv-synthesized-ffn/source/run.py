#!/usr/bin/env python3
import argparse,hashlib,io,json,math,random,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';METHODS=['shared','condconv','output_mix','rank1','independent'];DEV=[40800,40801];FRESH=[40810,40811,40812];SEEDS=[0,1,2];LRS=[.003,.01];UPDATES,BATCH=200,128;DIN,HID,CLS,CTX=32,48,10,4
def fixseed(s):random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
def data(world):
 g=torch.Generator().manual_seed(world);x=torch.randn(4096,DIN,generator=g);a1=torch.randn(2,DIN,HID,generator=g)/math.sqrt(DIN);a2=torch.randn(2,HID,CLS,generator=g)/math.sqrt(HID);coef=torch.softmax(torch.randn(CTX,2,generator=g)*1.1,dim=-1);cid=torch.arange(4096)%CTX;y=[]
 for i in range(len(x)):
  c=int(cid[i]);logits=[F.relu(x[i]@a1[k])@a2[k] for k in range(2)];y.append(int((coef[c,:,None]*torch.stack(logits)).sum(0).argmax()))
 return x,cid,torch.tensor(y)
class Model(nn.Module):
 def __init__(self,m):
  super().__init__();self.m=m
  if m=='independent':self.w1=nn.Parameter(torch.randn(CTX,DIN,HID)*.12);self.w2=nn.Parameter(torch.randn(CTX,HID,CLS)*.12)
  elif m in ('condconv','output_mix'):self.w1=nn.Parameter(torch.randn(2,DIN,HID)*.12);self.w2=nn.Parameter(torch.randn(2,HID,CLS)*.12)
  else:self.w1=nn.Parameter(torch.randn(DIN,HID)*.12);self.w2=nn.Parameter(torch.randn(HID,CLS)*.12)
  if m=='condconv':self.code=nn.Parameter(torch.zeros(CTX,2))
  if m=='output_mix':self.gate=nn.Parameter(torch.zeros(CTX,2))
  if m=='rank1':self.a=nn.Parameter(torch.randn(CTX,DIN,1)*.01);self.b=nn.Parameter(torch.randn(CTX,1,HID)*.01)
 def forward(self,x,c):
  if self.m=='independent':return torch.bmm(F.relu(torch.bmm(x[:,None,:],self.w1[c]).squeeze(1))[:,None,:],self.w2[c]).squeeze(1)
  if self.m=='condconv':
   out=x.new_zeros((x.size(0),CLS))
   for ctx in range(CTX):
    ix=(c==ctx).nonzero().squeeze(-1)
    if ix.numel()==0:continue
    co=self.code[ctx].softmax(-1);w1=torch.einsum('k,kdh->dh',co,self.w1);w2=torch.einsum('k,kho->ho',co,self.w2);h=F.relu(x[ix]@w1);out=out.index_copy(0,ix,h@w2)
   return out
  if self.m=='output_mix':
   z=torch.stack([F.relu(x@self.w1[k])@self.w2[k] for k in range(2)],1);return (z*self.gate[c].softmax(-1)[:,:,None]).sum(1)
  w=self.w1+(self.a[c]@self.b[c] if self.m=='rank1' else 0);h=F.relu(torch.bmm(x[:,None,:],w).squeeze(1) if w.ndim==3 else x@w);return h@self.w2
def serialize(m):
 b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},b,_use_new_zipfile_serialization=True);return b.getvalue()
def go(m,w,s,lr,split):
 fixseed(w*100+s);x,c,y=data(w);model=Model(m);opt=torch.optim.AdamW(model.parameters(),lr=lr);st=time.perf_counter()
 for _ in range(UPDATES):
  ix=torch.randint(0,3072,(BATCH,));loss=F.cross_entropy(model(x[ix],c[ix]),y[ix]);opt.zero_grad();loss.backward();opt.step()
 wall=time.perf_counter()-st
 with torch.no_grad():
  z=model(x[3072:],c[3072:]);acc=(z.argmax(-1)==y[3072:]).float().mean().item();ce=F.cross_entropy(z,y[3072:]).item();per=[(z[c[3072:]==k].argmax(-1)==y[3072:][c[3072:]==k]).float().mean().item() for k in range(CTX)]
 raw=serialize(model);name=f'{split}_{w}_{s}_{m}_lr{lr:g}.pt';(OUT/'payloads'/name).write_bytes(raw);cl=Model(m);cl.load_state_dict(torch.load(io.BytesIO(raw),weights_only=True));rb=serialize(cl);assert raw==rb
 one=2*DIN*HID+2*HID*CLS;synth_per_context=2*(DIN*HID+HID*CLS);synth_total=synth_per_context*CTX;mac=2*one if m=='output_mix' else one+(synth_total/3072 if m=='condconv' else DIN*HID if m=='rank1' else 0)
 return {'split':split,'world':w,'seed':s,'method':m,'lr':lr,'payload_file':name,'serialized_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'roundtrip_sha256':hashlib.sha256(rb).hexdigest(),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy_per_example':mac,'weight_synthesis_mac_proxy_total':synth_total if m=='condconv' else 0,'wall_time_s':round(wall,6),'accuracy':acc,'cross_entropy':ce,'per_context_accuracy':per}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);(OUT/'payloads').mkdir(exist_ok=True);rows=[];sel={}
 if a.phase=='dev':
  for m in METHODS:
   score={lr:[] for lr in LRS}
   for lr in LRS:
    for w in DEV:
     for s in SEEDS:r=go(m,w,s,lr,'development');rows.append(r);score[lr].append(r['accuracy'])
   sel[m]=max(LRS,key=lambda lr:sum(score[lr])/len(score[lr]))
  (OUT/'development_selection.json').write_text(json.dumps(sel,indent=2)+'\n');mode='w'
 else:
  sel=json.loads((OUT/'development_selection.json').read_text())
  for m in METHODS:
   for w in FRESH:
    for s in SEEDS:rows.append(go(m,w,s,sel[m],'fresh'))
  mode='a'
 with (OUT/'runs.jsonl').open(mode) as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
 print(json.dumps({'phase':a.phase,'rows':len(rows),'selection':sel},indent=2))
if __name__=='__main__':main()
