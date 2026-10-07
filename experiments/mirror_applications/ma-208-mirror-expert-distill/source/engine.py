import io
import math
import struct
import time

import torch
import torch.nn.functional as F
from torch import nn

D, H, C, TASKS, RANK = 12, 16, 4, 5, 2
METHODS = ('hard_tie', 'sequential_shared', 'mirror_view', 'multihead', 'lora_rank2', 'hypernet_rank2', 'independent_full')
TEMPERATURE = 2.0


def rotate_pair(x, angle):
    c, s = torch.cos(angle), torch.sin(angle); a, b = x[..., 0], x[..., 1]
    return torch.cat((torch.stack((c*a-s*b, s*a+c*b), -1), x[..., 2:]), dim=-1)


def world(seed, condition):
    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(D, H, generator=g) / math.sqrt(D); b1 = torch.randn(H, generator=g) * .05
    w2 = torch.randn(H, C, generator=g) / math.sqrt(H); b2 = torch.randn(C, generator=g) * .05
    angles = torch.zeros(TASKS); angles[1:] = torch.rand(TASKS-1, generator=g) * 1.2 - .6
    teachers = []
    for t in range(TASKS):
        if condition == 'aligned': teachers.append((w1.clone(), b1.clone(), w2.clone(), b2.clone()))
        elif t == 0: teachers.append((w1.clone(), b1.clone(), w2.clone(), b2.clone()))
        else:
            teachers.append((torch.randn(D, H, generator=g)/math.sqrt(D), torch.randn(H, generator=g)*.05,
                             torch.randn(H, C, generator=g)/math.sqrt(H), torch.randn(C, generator=g)*.05))
    if condition not in ('aligned', 'independent'): raise ValueError(condition)
    return {'seed': seed, 'condition': condition, 'base': (w1, b1, w2, b2), 'angles': angles, 'teachers': teachers}


def teacher_logits(x, w, task):
    w1, b1, w2, b2 = w['teachers'][task]; h = F.relu(x @ w1 + b1)
    if w['condition'] == 'aligned' and task > 0: h = rotate_pair(h, w['angles'][task])
    return h @ w2 + b2


def effective_teacher_state(w, task):
    """Return ordinary MLP weights equivalent to the teacher's task function."""
    w1, b1, w2, b2 = [z.detach().clone() for z in w['teachers'][task]]
    if w['condition'] == 'aligned' and task > 0:
        a = w['angles'][task]
        c, s = torch.cos(a), torch.sin(a)
        # row-vector hidden rotation means W2' = R^T W2
        w2[:2] = torch.stack((c * w2[0] + s * w2[1], -s * w2[0] + c * w2[1]))
    return w1, b1, w2, b2


class Hyper(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding = nn.Embedding(TASKS, 8); self.proj = nn.Linear(8, RANK*(H+C))
        nn.init.normal_(self.embedding.weight, std=.1); nn.init.normal_(self.proj.weight, std=.02); nn.init.zeros_(self.proj.bias)

    def factors(self, task):
        v = self.proj(self.embedding(torch.tensor(task))); a=v[:H*RANK].reshape(H,RANK); b=v[H*RANK:].reshape(RANK,C); return a,b


class Student:
    def __init__(self, method, base, seed, lr):
        torch.manual_seed(seed); self.method=method; self.base_w1,self.base_b1,self.base_w2,self.base_b2=[x.detach().clone() for x in base]
        self.angles={}; self.heads={}; self.lora={}; self.private={}; self.hyper=Hyper() if method=='hypernet_rank2' else None
        self.sequential_w2=nn.Parameter(self.base_w2.clone()) if method=='sequential_shared' else None
        self.sequential_b2=nn.Parameter(self.base_b2.clone()) if method=='sequential_shared' else None
        self.seen=[0]; self.optimizers={}; self.wall={}; self.examples={}; self.updates={}
        self.shared_optimizer=None
        if self.hyper: self.shared_optimizer=torch.optim.AdamW(self.hyper.parameters(),lr=lr,weight_decay=1e-4)
        if method=='sequential_shared': self.shared_optimizer=torch.optim.AdamW([self.sequential_w2,self.sequential_b2],lr=lr,weight_decay=1e-4)

    def forward(self,x,task):
        if self.method=='independent_full' and task in self.private:
            w1,b1,w2,b2=self.private[task]; return F.relu(x@w1+b1)@w2+b2
        h=F.relu(x@self.base_w1+self.base_b1)
        if self.method=='mirror_view' and task in self.angles: h=rotate_pair(h,self.angles[task])
        if self.method=='sequential_shared': return h@self.sequential_w2+self.sequential_b2
        if self.method=='multihead' and task in self.heads:
            w,b=self.heads[task]; return h@w+b
        if self.method=='lora_rank2' and task in self.lora:
            a,b=self.lora[task]; return h@self.base_w2+(h@a)@b+self.base_b2
        if self.method=='hypernet_rank2' and task>0:
            a,b=self.hyper.factors(task); return h@self.base_w2+(h@a)@b+self.base_b2
        return h@self.base_w2+self.base_b2

    def acquire(self,task,x,logits,lr,updates=300,teacher_state=None):
        if self.method=='hard_tie':
            if task not in self.seen: self.seen.append(task)
            return
        if self.method=='independent_full':
            self.private[task]=tuple(nn.Parameter(z.detach().clone(),requires_grad=False) for z in teacher_state)
            if task not in self.seen: self.seen.append(task)
            self.examples[task]=0; self.updates[task]=0; self.wall[task]=0.; return
        if self.method=='mirror_view':
            angle=nn.Parameter(torch.zeros(())); self.angles[task]=angle; params=[angle]
        elif self.method=='multihead':
            w=nn.Parameter(self.base_w2.clone()); b=nn.Parameter(self.base_b2.clone()); self.heads[task]=(w,b); params=[w,b]
        elif self.method=='lora_rank2':
            a=nn.Parameter(torch.randn(H,RANK)/math.sqrt(H)); b=nn.Parameter(torch.zeros(RANK,C)); self.lora[task]=(a,b); params=[a,b]
        elif self.method=='hypernet_rank2': params=list(self.hyper.parameters())
        elif self.method=='sequential_shared': params=[self.sequential_w2,self.sequential_b2]
        else: raise ValueError(self.method)
        opt=self.shared_optimizer if self.method in ('hypernet_rank2','sequential_shared') else torch.optim.AdamW(params,lr=lr,weight_decay=1e-4)
        if self.method not in ('hypernet_rank2','sequential_shared'): self.optimizers[task]=opt
        start=time.perf_counter(); target=F.softmax(logits/TEMPERATURE,dim=-1)
        for _ in range(updates):
            loss=F.kl_div(F.log_softmax(self.forward(x,task)/TEMPERATURE,dim=-1),target,reduction='batchmean')*(TEMPERATURE**2)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        self.wall[task]=time.perf_counter()-start; self.examples[task]=updates*len(x); self.updates[task]=updates
        if task not in self.seen: self.seen.append(task)

    def records(self,base_only=False):
        rec=[('base.W1',self.base_w1),('base.b1',self.base_b1),('base.W2',self.base_w2),('base.b2',self.base_b2),('task_ids',torch.tensor([0] if base_only else self.seen,dtype=torch.uint8))]
        if not base_only:
            if self.method=='mirror_view': rec.extend((f'angle.{t}',a.detach()) for t,a in sorted(self.angles.items()))
            if self.method=='multihead':
                for t,(w,b) in sorted(self.heads.items()): rec.extend([(f'head.{t}.W',w.detach()),(f'head.{t}.b',b.detach())])
            if self.method=='lora_rank2':
                for t,(a,b) in sorted(self.lora.items()): rec.extend([(f'lora.{t}.A',a.detach()),(f'lora.{t}.B',b.detach())])
            if self.method=='sequential_shared': rec.extend([('sequential.W2',self.sequential_w2.detach()),('sequential.b2',self.sequential_b2.detach())])
            if self.method=='independent_full':
                for t,(w1,b1,w2,b2) in sorted(self.private.items()): rec.extend([(f'private.{t}.W1',w1.detach()),(f'private.{t}.b1',b1.detach()),(f'private.{t}.W2',w2.detach()),(f'private.{t}.b2',b2.detach())])
            if self.hyper: rec.extend((f'hyper.{k}',v.detach()) for k,v in sorted(self.hyper.state_dict().items()))
        return rec

    def serialize(self): return serialize_records(self.method,self.records())
    def inference_bytes(self): return len(self.serialize())
    def base_bytes(self): return len(serialize_records(self.method,self.records(base_only=True)))
    def resume_bytes(self):
        opts={t:o.state_dict() for t,o in self.optimizers.items()}
        if self.shared_optimizer: opts['shared']=self.shared_optimizer.state_dict()
        f=io.BytesIO(); torch.save({'state':self.records(),'optimizers':opts},f); return len(f.getvalue())
    def resume_base_bytes(self):
        f=io.BytesIO(); torch.save({'state':self.records(base_only=True),'optimizers':{}},f); return len(f.getvalue())


def serialize_records(method,records):
    method=method.encode('ascii'); out=bytearray(b'MA208I1\0'); out.extend(struct.pack('<BHHHH',len(method),len(records),D,H,C)); out.extend(method)
    for name,t in records:
        name=name.encode('ascii'); t=t.detach().contiguous().cpu(); type_id=1 if t.dtype==torch.uint8 else 0
        if not type_id: t=t.float()
        raw=t.numpy().tobytes(); out.extend(struct.pack('<H',len(name))); out.extend(name); out.extend(struct.pack('<BB',type_id,t.ndim))
        if t.ndim: out.extend(struct.pack('<'+'H'*t.ndim,*t.shape))
        out.extend(struct.pack('<I',len(raw))); out.extend(raw)
    return bytes(out)


def deserialize_records(payload):
    view=memoryview(payload)
    if bytes(view[:8])!=b'MA208I1\0': raise ValueError('bad inference magic')
    ml,n,d,h,c=struct.unpack_from('<BHHHH',view,8)
    if (d,h,c)!=(D,H,C): raise ValueError('shape mismatch')
    off=17; method=bytes(view[off:off+ml]).decode('ascii'); off+=ml; records=[]
    for _ in range(n):
        nl=struct.unpack_from('<H',view,off)[0];off+=2;name=bytes(view[off:off+nl]).decode('ascii');off+=nl
        type_id,nd=struct.unpack_from('<BB',view,off);off+=2;shape=struct.unpack_from('<'+'H'*nd,view,off) if nd else ();off+=2*nd
        size=struct.unpack_from('<I',view,off)[0];off+=4;raw=bytes(view[off:off+size]);off+=size
        dtype=torch.uint8 if type_id==1 else torch.float32;records.append((name,torch.frombuffer(bytearray(raw),dtype=dtype).clone().reshape(shape)))
    if off!=len(view):raise ValueError('trailing bytes')
    return method,records


def ece(conf,correct,bins=10):
    edges=torch.linspace(0,1,bins+1);total=0.
    for i in range(bins):
        mask=(conf>=edges[i])&(conf<(edges[i+1]) if i<bins-1 else conf<=edges[i+1])
        if mask.any(): total+=float(mask.float().mean())*abs(float(conf[mask].mean()-correct[mask].float().mean()))
    return total


def evaluate(student,w,seed,upto):
    rows=[];g=torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for task in range(upto+1):
            x=torch.randn(2048,D,generator=g); tlog=teacher_logits(x,w,task); slog=student.forward(x,task)
            tp=F.softmax(tlog/TEMPERATURE,dim=-1); sp=F.softmax(slog/TEMPERATURE,dim=-1)
            kl=float((tp*(tp.clamp_min(1e-12).log()-sp.clamp_min(1e-12).log())).sum(-1).mean())
            tl=tlog.argmax(-1); sl=slog.argmax(-1); conf=F.softmax(slog,dim=-1).max(-1).values
            rows.append({'kl':kl,'agreement':float((tl==sl).float().mean()),'ece':ece(conf,tl==sl)})
    return rows


def mac_proxy(method,examples):
    base=D*H+H*C
    if method=='mirror_view':base+=H+4
    elif method=='multihead':base+=H*C+C
    elif method=='lora_rank2':base+=RANK*(H+C)
    elif method=='hypernet_rank2':base+=RANK*(H+C)+8*RANK*(H+C)
    elif method=='independent_full':base=D*H+H*C
    return examples*base
