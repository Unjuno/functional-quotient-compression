import io,math,struct,time
import torch
from torch import nn

D,O,TASKS,RANK=16,8,5,2
METHODS=('mirror_code','lora_rank2','hypernet_rank2','sequential_finetune','hard_tie','independent_full')

def serialize_records(method,records):
 method=method.encode('ascii');out=bytearray(b'MA186I1\0');out.extend(struct.pack('<BHHHHH',len(method),len(records),D,O,TASKS,RANK));out.extend(method)
 for name,t in records:
  name=name.encode('ascii');t=t.detach().contiguous().cpu()
  if t.dtype==torch.uint8:type_id=1;raw=t.numpy().tobytes()
  else:type_id=0;t=t.float();raw=t.numpy().tobytes()
  out.extend(struct.pack('<H',len(name)));out.extend(name);out.extend(struct.pack('<BB',type_id,t.ndim))
  if t.ndim:out.extend(struct.pack('<'+'H'*t.ndim,*t.shape))
  out.extend(struct.pack('<I',len(raw)));out.extend(raw)
 return bytes(out)

def rotate_first_pair(x,theta):
 c,s=torch.cos(theta),torch.sin(theta);a,b=x[...,0],x[...,1]
 return torch.cat((torch.stack((c*a-s*b,s*a+c*b),-1),x[...,2:]),dim=-1)

def world(seed,condition):
 g=torch.Generator().manual_seed(seed);w0=torch.randn(D,O,generator=g)/math.sqrt(D);b0=torch.randn(O,generator=g)*.05
 if condition=='aligned':
  angles=torch.zeros(TASKS);angles[1:]=torch.rand(TASKS-1,generator=g)*.8-.4
  weights=torch.stack([w0]*TASKS);biases=b0.expand(TASKS,-1).clone()
 else:
  angles=torch.zeros(TASKS);weights=torch.stack([w0]+[torch.randn(D,O,generator=g)/math.sqrt(D) for _ in range(TASKS-1)]);biases=torch.stack([b0]+[torch.randn(O,generator=g)*.05 for _ in range(TASKS-1)])
 return {'seed':seed,'condition':condition,'teacher_w':weights,'teacher_b':biases,'angles':angles}

def teacher(x,task,w):
 if w['condition']=='aligned':return rotate_first_pair(x,w['angles'][task])@w['teacher_w'][task]+w['teacher_b'][task]
 return x@w['teacher_w'][task]+w['teacher_b'][task]

def pretrain_base(w,seed,lr,updates=600,batch=64):
 torch.manual_seed(seed);weight=nn.Parameter(torch.randn(D,O)*.08);bias=nn.Parameter(torch.zeros(O));opt=torch.optim.AdamW([weight,bias],lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(seed+77);start=time.perf_counter()
 for _ in range(updates):
  x=torch.randn(batch,D,generator=g);y=teacher(x,0,w);loss=((x@weight+bias-y)**2).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return weight.detach(),bias.detach(),opt.state_dict(),time.perf_counter()-start,updates*batch

class Hyper(nn.Module):
 def __init__(self):
  super().__init__();self.embedding=nn.Embedding(TASKS,8);self.proj=nn.Linear(8,RANK*(D+O))
  nn.init.normal_(self.embedding.weight,std=.02);nn.init.zeros_(self.proj.weight);nn.init.zeros_(self.proj.bias)
 def factors(self,task):
  v=self.proj(self.embedding(torch.tensor(task)));a=v[:D*RANK].reshape(D,RANK);b=v[D*RANK:].reshape(RANK,O);return a,b

class Learner:
 def __init__(self,method,base_w,base_b,seed,pretrain_opt=None,lr=.003):
  torch.manual_seed(seed);self.method=method;self.base_w=nn.Parameter(base_w.clone(),requires_grad=method=='sequential_finetune');self.base_b=nn.Parameter(base_b.clone(),requires_grad=method=='sequential_finetune');self.seed=seed;self.codes={};self.adapters={};self.private={};self.opts={};self.shared_opt=None;self.hyper=Hyper() if method=='hypernet_rank2' else None
  if method=='sequential_finetune':
   self.shared_opt=torch.optim.AdamW([self.base_w,self.base_b],lr=lr,weight_decay=1e-4)
   if pretrain_opt is not None:self.shared_opt.load_state_dict(pretrain_opt)
  if self.hyper is not None:self.shared_opt=torch.optim.AdamW(self.hyper.parameters(),lr=lr,weight_decay=1e-4)
  self.seen=[0];self.wall=0.;self.updates=0;self.examples=0

 def forward(self,x,task):
  if self.method=='sequential_finetune':return x@self.base_w+self.base_b
  if self.method=='mirror_code' and task in self.codes:return rotate_first_pair(x,self.codes[task])@self.base_w+self.base_b
  if self.method=='lora_rank2' and task in self.adapters:
   a,b=self.adapters[task];return x@self.base_w+(x@a)@b+self.base_b
  if self.method=='hypernet_rank2' and task>0:
   a,b=self.hyper.factors(task);return x@self.base_w+(x@a)@b+self.base_b
  if self.method=='independent_full' and task in self.private:
   a,b=self.private[task];return x@a+b
  return x@self.base_w+self.base_b

 def acquire(self,task,w,lr,updates=300,batch=64):
  if task==0 or self.method=='hard_tie':self.seen.append(task);return
  pars=[]
  if self.method=='mirror_code':
   self.codes[task]=nn.Parameter(torch.zeros(()));pars=[self.codes[task]];opt=torch.optim.AdamW(pars,lr=lr,weight_decay=0)
  elif self.method=='lora_rank2':
   torch.manual_seed(self.seed+task*97);a=nn.Parameter(torch.zeros(D,RANK));b=nn.Parameter(torch.randn(RANK,O)*.02);self.adapters[task]=(a,b);opt=torch.optim.AdamW([a,b],lr=lr,weight_decay=1e-4)
  elif self.method=='hypernet_rank2':opt=self.shared_opt
  elif self.method=='sequential_finetune':opt=self.shared_opt
  elif self.method=='independent_full':
   a=nn.Parameter(self.base_w.detach().clone());b=nn.Parameter(self.base_b.detach().clone());self.private[task]=(a,b);opt=torch.optim.AdamW([a,b],lr=lr,weight_decay=1e-4)
  else:raise ValueError(self.method)
  if self.method in ('mirror_code','lora_rank2','independent_full'):self.opts[task]=opt
  else:self.opts[0]=opt
  g=torch.Generator().manual_seed(self.seed+task*1009+int(lr*1e5));start=time.perf_counter()
  for _ in range(updates):
   x=torch.randn(batch,D,generator=g);y=teacher(x,task,w);loss=(self.forward(x,task)-y).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  self.wall+=time.perf_counter()-start;self.updates+=updates;self.examples+=updates*batch;self.seen.append(task)

 def inference_state(self):
  state={'method':self.method,'base_w':self.base_w.detach(),'base_b':self.base_b.detach(),'task_ids':torch.arange(max(self.seen)+1,dtype=torch.uint8),'codes':{k:v.detach() for k,v in self.codes.items()},'adapters':{k:(a.detach(),b.detach()) for k,(a,b) in self.adapters.items()},'private':{k:(a.detach(),b.detach()) for k,(a,b) in self.private.items()}}
  if self.hyper is not None:state['hyper']=self.hyper.state_dict()
  return state

 def serialize_inference(self):
  state=self.inference_state();records=[('base_w',state['base_w']),('base_b',state['base_b']),('task_ids',state['task_ids'])]
  records.extend((f'codes.{k}',v) for k,v in sorted(state['codes'].items()))
  for k,(a,b) in sorted(state['adapters'].items()):records.extend([(f'adapters.{k}.A',a),(f'adapters.{k}.B',b)])
  for k,(a,b) in sorted(state['private'].items()):records.extend([(f'private.{k}.W',a),(f'private.{k}.b',b)])
  if self.hyper is not None:records.extend((f'hyper.{k}',v) for k,v in sorted(state['hyper'].items()))
  return serialize_records(self.method,records)

 @staticmethod
 def from_inference(payload):
  view=memoryview(payload);assert bytes(view[:8])==b'MA186I1\0';off=8
  ml,n,D0,O0,T0,R0=struct.unpack_from('<BHHHHH',view,off);off+=11;method=bytes(view[off:off+ml]).decode('ascii');off+=ml
  assert (D0,O0,T0,R0)==(D,O,TASKS,RANK);records={}
  for _ in range(n):
   nl=struct.unpack_from('<H',view,off)[0];off+=2;name=bytes(view[off:off+nl]).decode('ascii');off+=nl;type_id,nd=struct.unpack_from('<BB',view,off);off+=2
   shape=struct.unpack_from('<'+'H'*nd,view,off) if nd else ();off+=2*nd
   size=struct.unpack_from('<I',view,off)[0];off+=4;raw=bytes(view[off:off+size]);off+=size;dtype=torch.uint8 if type_id==1 else torch.float32
   records[name]=torch.frombuffer(bytearray(raw),dtype=dtype).clone().reshape(shape)
  if off!=len(view):raise ValueError('trailing inference payload bytes')
  m=Learner(method,records['base_w'],records['base_b'],0)
  m.codes={int(k.split('.')[1]):nn.Parameter(v.clone()) for k,v in records.items() if k.startswith('codes.')}
  for k,v in records.items():
   parts=k.split('.')
   if len(parts)==3 and parts[0]=='adapters':
    task=int(parts[1]);m.adapters.setdefault(task,[None,None])[0 if parts[2]=='A' else 1]=nn.Parameter(v.clone())
   if len(parts)==3 and parts[0]=='private':
    task=int(parts[1]);m.private.setdefault(task,[None,None])[0 if parts[2]=='W' else 1]=nn.Parameter(v.clone())
  m.adapters={k:tuple(v) for k,v in m.adapters.items()};m.private={k:tuple(v) for k,v in m.private.items()}
  if m.hyper is not None:m.hyper.load_state_dict({k[len('hyper.'):]:v for k,v in records.items() if k.startswith('hyper.')})
  m.seen=list(records['task_ids'].tolist());return m

 def inference_bytes(self):return len(self.serialize_inference())

 def base_payload_bytes(self):
  records=[('base_w',self.base_w.detach()),('base_b',self.base_b.detach()),('task_ids',torch.tensor([0],dtype=torch.uint8))]
  return len(serialize_records(self.method,records))

 def resume_base_bytes(self):
  base={'method':self.method,'base_w':self.base_w.detach(),'base_b':self.base_b.detach(),'task_ids':torch.tensor([0],dtype=torch.uint8),'codes':{},'adapters':{},'private':{}}
  if self.hyper is not None:base['hyper']=self.hyper.state_dict()
  f=io.BytesIO();torch.save({'state':base,'optimizers':{}},f);return len(f.getvalue())

 def resume_bytes(self):
  optimizers={k:o.state_dict() for k,o in self.opts.items()}
  f=io.BytesIO();torch.save({'state':self.inference_state(),'optimizers':optimizers},f);return len(f.getvalue())

def evaluate(learner,w,seed):
 g=torch.Generator().manual_seed(seed);x=torch.randn(1024,D,generator=g);out=[]
 with torch.no_grad():
  for t in range(TASKS):out.append(float((learner.forward(x,t)-teacher(x,t,w)).square().mean()))
 return out

def mac_proxy(method,examples):
 base=examples*D*O
 if method=='mirror_code':return base+examples*D
 if method=='lora_rank2':return base+examples*RANK*(D+O)
 if method=='hypernet_rank2':return base+examples*RANK*(D+O)
 if method=='independent_full':return base
 return base
