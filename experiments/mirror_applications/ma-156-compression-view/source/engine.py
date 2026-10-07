"""Actual-byte weight reconstruction screen for MA-156."""
import io,math,struct,time
from dataclasses import dataclass
import torch

D,ROLES=16,4
ANGLES=torch.tensor([-.60,-.20,.20,.60])
METHODS=('int4_independent','int4_tied','int4_mirror','int4_shared_rank2','int4_qer_rank2','fp32_untied')

def rotation(theta):
 c,s=torch.cos(theta),torch.sin(theta);r=torch.eye(D);r[0,0],r[0,1]=c,-s;r[1,0],r[1,1]=s,c;return r

def pack_int4(q):
 flat=q.to(torch.int16).reshape(-1).clamp(-7,7);nibs=torch.where(flat<0,flat+16,flat).to(torch.uint8)
 if nibs.numel()%2:nibs=torch.cat((nibs,torch.zeros(1,dtype=torch.uint8)))
 return (nibs[0::2] | (nibs[1::2]<<4)).numpy().tobytes()

def unpack_int4(raw,count):
 data=torch.tensor(list(raw),dtype=torch.uint8);v=torch.empty(data.numel()*2,dtype=torch.int16)
 v[0::2]=(data&15).to(torch.int16);v[1::2]=(data>>4).to(torch.int16)
 v=torch.where(v>=8,v-16,v);return v[:count].to(torch.int8)

def quant4(w):
 scale=float(w.abs().max().clamp_min(1e-8)/7);q=torch.round(w/scale).clamp(-7,7).to(torch.int8);return q,scale,q.float()*scale

def rank_residual(a,rank=2):
 u,s,v=torch.linalg.svd(a,full_matrices=False);root=s[:rank].clamp_min(0).sqrt()
 left=(u[:,:rank]*root).half();right=(root[:,None]*v[:rank]).half();return left,right,left.float()@right.float()

def serialize(method,objs):
 out=bytearray(b'MA156V1');name=method.encode();out.extend(struct.pack('<B',len(name)));out.extend(name);out.extend(struct.pack('<HHB',D,ROLES,4))
 for kind,value in objs:
  k=kind.encode();out.extend(struct.pack('<B',len(k)));out.extend(k)
  if kind=='q4':
   q,scale=value;raw=pack_int4(q);out.extend(struct.pack('<fI',scale,len(raw)));out.extend(raw)
  elif kind=='f32':
   raw=value.contiguous().float().numpy().tobytes();out.extend(struct.pack('<I',len(raw)));out.extend(raw)
  elif kind=='f16':
   raw=value.contiguous().half().numpy().tobytes();out.extend(struct.pack('<I',len(raw)));out.extend(raw)
 return bytes(out)

def deserialize(payload):
 view=memoryview(payload);off=7
 n=view[off];off+=1;method=bytes(view[off:off+n]).decode();off+=n
 d,roles,bits=struct.unpack_from('<HHB',view,off);off+=5;objects=[]
 while off<len(view):
  n=view[off];off+=1;kind=bytes(view[off:off+n]).decode();off+=n
  if kind=='q4':
   scale,size=struct.unpack_from('<fI',view,off);off+=8;raw=bytes(view[off:off+size]);off+=size
   q=unpack_int4(raw,d*d).reshape(d,d);objects.append((kind,q,scale))
  elif kind in ('f32','f16'):
   size=struct.unpack_from('<I',view,off)[0];off+=4;raw=bytes(view[off:off+size]);off+=size
   dtype=torch.float32 if kind=='f32' else torch.float16
   left_factor=(len(objects)%3==1) if method=='int4_qer_rank2' else (len(objects)%2==1)
   shape=(roles,) if kind=='f32' and method=='int4_mirror' else (roles,d,d) if kind=='f32' else (d,2) if left_factor else (2,d)
   objects.append((kind,torch.frombuffer(bytearray(raw),dtype=dtype).clone().reshape(shape)))
 if method=='fp32_untied':return objects[0][1]
 if method=='int4_independent':return torch.stack([q.float()*s for _,q,s in objects])
 if method=='int4_qer_rank2':
  mats=[]
  for i in range(0,len(objects),3):
   _,q,s=objects[i];l=objects[i+1][1].float();r=objects[i+2][1].float();mats.append(q.float()*s+l@r)
  return torch.stack(mats)
 _,q,s=objects[0];base=q.float()*s
 if method=='int4_tied':return base[None].expand(roles,-1,-1)
 if method=='int4_mirror':
  angles=objects[1][1];return torch.stack([rotation(a)@base@rotation(-a) for a in angles])
 mats=[]
 for i in range(1,len(objects),2):mats.append(base+objects[i][1].float()@objects[i+1][1].float())
 return torch.stack(mats)

def make_world(seed,aligned):
 g=torch.Generator().manual_seed(seed);base=torch.randn(D,D,generator=g)/math.sqrt(D)
 if aligned: targets=torch.stack([rotation(a)@base@rotation(-a) for a in ANGLES])
 else:
  targets=torch.randn(ROLES,D,D,generator=g)/math.sqrt(D);base=targets.mean(dim=0)
 x=torch.randn(512,D,generator=g);xt=torch.randn(256,D,generator=g)
 return targets,base,x,xt

def build(method,targets,base):
 objs=[]
 if method=='fp32_untied':
  objs=[('f32',targets)];return targets,serialize(method,objs)
 if method=='int4_independent':
  mats=[]
  for w in targets:
   q,s,dq=quant4(w);mats.append(dq);objs.append(('q4',(q,s)))
  return torch.stack(mats),serialize(method,objs)
 if method=='int4_qer_rank2':
  mats=[]
  for w in targets:
   q,s,dq=quant4(w);l,r,c=rank_residual(w-dq);mats.append(dq+c);objs.extend([('q4',(q,s)),('f16',l),('f16',r)])
  return torch.stack(mats),serialize(method,objs)
 q,s,qb=quant4(base);objs.append(('q4',(q,s)))
 if method=='int4_tied':
  mats=qb[None].expand(ROLES,-1,-1)
 elif method=='int4_mirror':
  objs.append(('f32',ANGLES));mats=torch.stack([rotation(a)@qb@rotation(-a) for a in ANGLES])
 elif method=='int4_shared_rank2':
  out=[]
  for w in targets:
   l,r,c=rank_residual(w-qb);objs.extend([('f16',l),('f16',r)]);out.append(qb+c)
  mats=torch.stack(out)
 return mats,serialize(method,objs)

@dataclass
class Result:
 method:str;seed:int;condition:str;relative_fro:float;activation_mse:float;payload_bytes:int;decode_macs:int;wall_s:float

def run_one(method,seed,condition):
 targets,base,x,xt=make_world(seed,condition=='aligned');start=time.perf_counter();decoded,payload=build(method,targets,base);decode_wall=time.perf_counter()-start
 with torch.no_grad():
  err=decoded-targets;rel=float(err.norm()/targets.norm());act=float(torch.einsum('dij,nj->dni',err,xt).square().mean())
  for _ in range(2):torch.einsum('dij,nj->dni',decoded,x)
  t=time.perf_counter()
  for _ in range(20):torch.einsum('dij,nj->dni',decoded,x)
  wall=decode_wall+(time.perf_counter()-t)/20
 macs=ROLES*D*D if method=='int4_independent' else D*D if method=='int4_tied' else 0
 if method=='int4_mirror':macs=D*D+ROLES*2*D**3
 if method=='int4_shared_rank2':macs=D*D+ROLES*2*D*D*2
 if method=='int4_qer_rank2':macs=ROLES*(D*D+2*D*D*2)
 return Result(method,seed,condition,rel,act,len(payload),macs,wall)

def run(seed,condition):return [run_one(m,seed,condition) for m in METHODS]
