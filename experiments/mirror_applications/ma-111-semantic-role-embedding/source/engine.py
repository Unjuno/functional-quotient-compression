import io, math, struct, time
import torch
import torch.nn.functional as F
from torch import nn

V,D,C,ROLES,RANK=32,16,8,4,2
METHODS=('hard_tie','role_add','role_film','input_lora_rank2','head_lora_rank2','full_role_head','vsa_sign','full_role_map','mirror_givens','exact_teacher')
TEMPERATURE=2.0

def rotate_pair(x,a):
 c,s=torch.cos(a),torch.sin(a);u,v=x[...,0],x[...,1]
 return torch.cat((torch.stack((c*u-s*v,s*u+c*v),-1),x[...,2:]),-1)

def world(seed,condition):
 g=torch.Generator().manual_seed(seed);emb=torch.randn(V,D,generator=g)/math.sqrt(D);readout=torch.randn(D,C,generator=g)/math.sqrt(D);bias=torch.randn(C,generator=g)*.03
 angles=torch.zeros(ROLES);angles[1:]=torch.rand(ROLES-1,generator=g)*.9-.45;maps=torch.eye(D).repeat(ROLES,1,1)
 if condition=='independent':
  for r in range(1,ROLES):
   q,z=torch.linalg.qr(torch.randn(D,D,generator=g));maps[r]=q*torch.sign(torch.diag(z)).unsqueeze(0)
 elif condition!='aligned':raise ValueError(condition)
 return {'seed':seed,'condition':condition,'emb':emb,'readout':readout,'bias':bias,'angles':angles,'maps':maps}

def teacher_logits(x,w,role):
 h=x if role==0 else rotate_pair(x,w['angles'][role]) if w['condition']=='aligned' else x@w['maps'][role]
 return h@w['readout']+w['bias']

class Student:
 def __init__(self,method,w,seed,lr):
  torch.manual_seed(seed);self.method=method;self.emb=w['emb'].clone();self.readout=w['readout'].clone();self.bias=w['bias'].clone()
  self.angles={};self.add={};self.film={};self.input_lora={};self.head_lora={};self.full_head={};self.full={};self.optimizers={};self.wall={};self.examples={};self.updates={};self.seen=[0]
  g=torch.Generator().manual_seed(11100);self.vsa_codes=torch.ones(ROLES,D)
  for r in range(1,ROLES):self.vsa_codes[r]=torch.where(torch.randint(0,2,(D,),generator=g).bool(),1.,-1.)
 def set_teacher_reference(self,w):self.teacher_condition=w['condition'];self.teacher_angles=w['angles'].clone();self.teacher_maps=w['maps'].clone()
 def params(self,r):
  m=self.method
  if m=='mirror_givens':self.angles[r]=nn.Parameter(torch.zeros(()));return [self.angles[r]]
  if m=='role_add':self.add[r]=nn.Parameter(torch.zeros(D));return [self.add[r]]
  if m=='role_film':
   a=nn.Parameter(torch.ones(D));b=nn.Parameter(torch.zeros(D));self.film[r]=(a,b);return [a,b]
  if m=='full_role_head':self.full_head[r]=nn.Parameter(torch.zeros(D,C));return [self.full_head[r]]
  if m in ('input_lora_rank2','head_lora_rank2'):
   out=D if m=='input_lora_rank2' else C;a=nn.Parameter(torch.randn(D,RANK)/math.sqrt(D));b=nn.Parameter(torch.zeros(RANK,out))
   (self.input_lora if m=='input_lora_rank2' else self.head_lora)[r]=(a,b);return [a,b]
  if m=='full_role_map':self.full[r]=nn.Parameter(torch.eye(D));return [self.full[r]]
  return []
 def forward(self,x,r):
  if self.method=='exact_teacher' and r>0:
   h=rotate_pair(x,self.teacher_angles[r]) if self.teacher_condition=='aligned' else x@self.teacher_maps[r]
   return h@self.readout+self.bias
  h=x;m=self.method
  if m=='mirror_givens' and r in self.angles:h=rotate_pair(h,self.angles[r])
  elif m=='role_add' and r in self.add:h=h+self.add[r]
  elif m=='role_film' and r in self.film:a,b=self.film[r];h=h*a+b
  elif m=='input_lora_rank2' and r in self.input_lora:a,b=self.input_lora[r];h=h+(h@a)@b
  elif m=='full_role_map' and r in self.full:h=h@self.full[r]
  elif m=='vsa_sign':h=h*self.vsa_codes[r]
  if m=='head_lora_rank2' and r in self.head_lora:a,b=self.head_lora[r];return h@self.readout+(h@a)@b+self.bias
  if m=='full_role_head' and r in self.full_head:return h@self.readout+h@self.full_head[r]+self.bias
  return h@self.readout+self.bias
 def acquire(self,r,x,logits,lr,updates=300):
  if r==0 or self.method in ('hard_tie','vsa_sign','exact_teacher'):
   if r not in self.seen:self.seen.append(r)
   self.examples[r]=0;self.updates[r]=0;self.wall[r]=0.;return
  ps=self.params(r);opt=torch.optim.AdamW(ps,lr=lr,weight_decay=1e-4);self.optimizers[r]=opt;target=F.softmax(logits/TEMPERATURE,-1);start=time.perf_counter()
  for _ in range(updates):
   loss=F.kl_div(F.log_softmax(self.forward(x,r)/TEMPERATURE,-1),target,reduction='batchmean')*TEMPERATURE**2;opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  self.wall[r]=time.perf_counter()-start;self.examples[r]=updates*len(x);self.updates[r]=updates
  if r not in self.seen:self.seen.append(r)
 def records(self,base_only=False):
  r=[('base.embedding',self.emb),('base.readout',self.readout),('base.bias',self.bias),('role_ids',torch.tensor([0] if base_only else sorted(self.seen),dtype=torch.uint8))]
  if base_only:return r
  m=self.method
  if m=='mirror_givens':r += [(f'angle.{i}',v.detach()) for i,v in sorted(self.angles.items())]
  if m=='role_add':r += [(f'add.{i}',v.detach()) for i,v in sorted(self.add.items())]
  if m=='role_film':
   for i,(a,b) in sorted(self.film.items()):r += [(f'film.{i}.scale',a.detach()),(f'film.{i}.shift',b.detach())]
  if m in ('input_lora_rank2','head_lora_rank2'):
   obj=self.input_lora if m=='input_lora_rank2' else self.head_lora;name='input' if m=='input_lora_rank2' else 'head'
   for i,(a,b) in sorted(obj.items()):r += [(f'{name}.{i}.A',a.detach()),(f'{name}.{i}.B',b.detach())]
  if m=='full_role_head':r += [(f'full_head.{i}',v.detach()) for i,v in sorted(self.full_head.items())]
  if m=='full_role_map':r += [(f'full.{i}',v.detach()) for i,v in sorted(self.full.items())]
  if m=='vsa_sign':r += [('vsa.role_codes',self.vsa_codes)]
  if m=='exact_teacher':
   if self.teacher_condition=='aligned':r += [('teacher.angles',self.teacher_angles)]
   else:r += [('teacher.maps',self.teacher_maps)]
  return r
 def serialize(self,base_only=False):return serialize_records(self.method,self.records(base_only))
 def base_bytes(self):return len(self.serialize(True))
 def resume_bytes(self):
  f=io.BytesIO();torch.save({'state':self.records(),'optimizers':{i:o.state_dict() for i,o in self.optimizers.items()}},f);return len(f.getvalue())

def serialize_records(method,records):
 method=method.encode('ascii');out=bytearray(b'MA111I1\0');out.extend(struct.pack('<BHHHH',len(method),len(records),V,D,C));out.extend(method)
 for name,t in records:
  name=name.encode('ascii');t=t.detach().contiguous().cpu();tid=1 if t.dtype==torch.uint8 else 0
  if not tid:t=t.float()
  raw=t.numpy().tobytes();out.extend(struct.pack('<H',len(name)));out.extend(name);out.extend(struct.pack('<BB',tid,t.ndim))
  if t.ndim:out.extend(struct.pack('<'+'H'*t.ndim,*t.shape))
  out.extend(struct.pack('<I',len(raw)));out.extend(raw)
 return bytes(out)

def deserialize_records(payload):
 v=memoryview(payload)
 if bytes(v[:8])!=b'MA111I1\0':raise ValueError('bad inference magic')
 ml,n,vocab,d,c=struct.unpack_from('<BHHHH',v,8)
 if (vocab,d,c)!=(V,D,C):raise ValueError('shape mismatch')
 off=17;m=bytes(v[off:off+ml]).decode('ascii');off+=ml;rec=[]
 for _ in range(n):
  nl=struct.unpack_from('<H',v,off)[0];off+=2;name=bytes(v[off:off+nl]).decode('ascii');off+=nl
  tid,nd=struct.unpack_from('<BB',v,off);off+=2;shape=struct.unpack_from('<'+'H'*nd,v,off) if nd else ();off+=2*nd
  size=struct.unpack_from('<I',v,off)[0];off+=4;raw=bytes(v[off:off+size]);off+=size;dtype=torch.uint8 if tid==1 else torch.float32
  rec.append((name,torch.frombuffer(bytearray(raw),dtype=dtype).clone().reshape(shape)))
 if off!=len(v):raise ValueError('trailing bytes')
 return m,rec

def make_batch(w,seed,n,train):
 g=torch.Generator().manual_seed(seed);lo,hi=(0,24) if train else (24,32);ids=torch.randint(lo,hi,(n,),generator=g);x=w['emb'][ids]+torch.randn(n,D,generator=g)*.03;return x,ids

def ece(conf,correct,bins=10):
 edges=torch.linspace(0,1,bins+1);total=0.
 for i in range(bins):
  mask=(conf>=edges[i])&(conf<(edges[i+1]) if i<bins-1 else conf<=edges[i+1])
  if mask.any():total+=float(mask.float().mean())*abs(float(conf[mask].mean()-correct[mask].float().mean()))
 return total

def evaluate(s,w,seed,upto):
 rows=[]
 with torch.no_grad():
  for r in range(upto+1):
   x,_=make_batch(w,seed+r*17,1024,False);tl=teacher_logits(x,w,r);sl=s.forward(x,r);tp=F.softmax(tl/TEMPERATURE,-1);sp=F.softmax(sl/TEMPERATURE,-1)
   nll=float(-(tp*sp.clamp_min(1e-12).log()).sum(-1).mean())
   kl=float((tp*(tp.clamp_min(1e-12).log()-sp.clamp_min(1e-12).log())).sum(-1).mean().clamp_min(0));t=tl.argmax(-1);p=sl.argmax(-1);conf=F.softmax(sl,-1).max(-1).values
   rows.append({'nll':nll,'kl':kl,'agreement':float((t==p).float().mean()),'ece':ece(conf,t==p)})
 return rows

def mac_proxy(method,n,condition):
 extra={'mirror_givens':8,'role_add':D,'role_film':2*D,'input_lora_rank2':2*D*RANK,'head_lora_rank2':RANK*(D+C),'full_role_head':D*C,'full_role_map':D*D,'vsa_sign':D}.get(method,0)
 if method=='exact_teacher':extra=8 if condition=='aligned' else D*D
 return n*(D*C+extra)

def benchmark_inference(s,seed,repeats=50):
 w=world(seed,'aligned');x,_=make_batch(w,seed+1,2048,False)
 with torch.no_grad():
  for _ in range(5):s.forward(x,ROLES-1)
  t=time.perf_counter()
  for _ in range(repeats):s.forward(x,ROLES-1)
 elapsed=time.perf_counter()-t
 return {'inference_batch_size':len(x),'inference_repeats':repeats,'inference_wall_time_s':elapsed,'inference_examples_per_s':len(x)*repeats/elapsed}
