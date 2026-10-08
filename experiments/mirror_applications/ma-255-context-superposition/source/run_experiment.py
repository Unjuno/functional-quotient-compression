"""Synthetic parameter-superposition versus learned Mirror task contexts."""
from __future__ import annotations
import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn

T=4;D=32

def make_world(seed):
    g=torch.Generator().manual_seed(seed)
    teachers=torch.randn(T,D,generator=g); teachers=teachers/teachers.norm(dim=-1,keepdim=True)*3.0
    data=[]
    for split,n,off in [('train',1024,100),('validation',2048,300),('test',4096,500)]:
        gg=torch.Generator().manual_seed(seed+off)
        x=torch.randn(T,n,D,generator=gg); y=((x*teachers[:,None,:]).sum(-1)>0).float()
        data.append((x,y))
    return teachers,dict(zip(('train','validation','test'),data))

def fit_independent(seed,train_x,train_y):
    torch.manual_seed(seed); ws=[]; wall=0.0
    for t in range(T):
        w=nn.Parameter(torch.zeros(D)); opt=torch.optim.Adam([w],lr=.02); start=time.perf_counter()
        g=torch.Generator().manual_seed(seed+1000+t)
        for _ in range(500):
            ix=torch.randint(train_x.shape[1],(128,),generator=g)
            logits=train_x[t,ix]@w; loss=nn.functional.binary_cross_entropy_with_logits(logits,train_y[t,ix])
            opt.zero_grad();loss.backward();opt.step()
        wall+=time.perf_counter()-start;ws.append(w.detach().clone())
    return torch.stack(ws),wall

def rotate(x,theta):
    q=x.reshape(*x.shape[:-1],D//2,2); c=theta.cos();s=theta.sin()
    a=q[...,0];b=q[...,1]
    return torch.stack((c*a-s*b,s*a+c*b),-1).reshape_as(x)

def build_phase(weights,theta): return rotate(weights,theta[:,None]).sum(0)
def retrieve_phase(shared,theta): return rotate(shared.expand(T,-1),-theta[:,None])
def evaluate(weights,x,y):
    with torch.no_grad():
        p=torch.sigmoid(torch.einsum('tnd,td->tn',x,weights)); pred=(p>=.5).float();
        acc=(pred==y).float().mean(-1)
        loss=nn.functional.binary_cross_entropy(p,y,reduction='none').mean(-1)
    return {'mean_accuracy':float(acc.mean()),'per_task_accuracy':acc.tolist(),'mean_bce':float(loss.mean()),'per_task_bce':loss.tolist()}

def optimize_phase(weights,x,y,seed,steps=300):
    torch.manual_seed(seed); theta=nn.Parameter(torch.rand(T)*6.2831853);opt=torch.optim.Adam([theta],lr=.01);g=torch.Generator().manual_seed(seed+44)
    start=time.perf_counter()
    for _ in range(steps):
        shared=build_phase(weights,theta); recovered=retrieve_phase(shared,theta)
        losses=[]
        for t in range(T):
            ix=torch.randint(x.shape[1],(128,),generator=g)
            losses.append(nn.functional.binary_cross_entropy_with_logits(x[t,ix]@recovered[t],y[t,ix]))
        loss=torch.stack(losses).mean();opt.zero_grad();loss.backward();opt.step()
    return theta.detach(),time.perf_counter()-start

def optimize_dense(weights,x,y,seed,steps=300):
    torch.manual_seed(seed); codes=nn.Parameter(torch.ones(T,D)+.02*torch.randn(T,D));opt=torch.optim.Adam([codes],lr=.01);g=torch.Generator().manual_seed(seed+54);start=time.perf_counter()
    for _ in range(steps):
        shared=(codes*weights).sum(0); recovered=codes*shared
        losses=[]
        for t in range(T):
            ix=torch.randint(x.shape[1],(128,),generator=g)
            losses.append(nn.functional.binary_cross_entropy_with_logits(x[t,ix]@recovered[t],y[t,ix]))
        loss=torch.stack(losses).mean();opt.zero_grad();loss.backward();opt.step()
    return codes.detach(),time.perf_counter()-start

def archive(path,method,shared,codes,seed):
    arrays={'shared':shared.detach().cpu().numpy().astype(np.float16)}
    if method=='psp_sign': arrays['context_bits']=np.packbits((np.asarray(codes)>0).astype(np.uint8).reshape(-1))
    elif method in ('mirror_fixed','mirror_learned'): arrays['angles']=np.asarray(codes,dtype=np.float16)
    elif method=='dense_learned': arrays['dense_codes']=np.asarray(codes,dtype=np.float16)
    meta={'experiment_id':'MA-255','method':method,'task_count':T,'input_dim':D,'code_seed':seed if method=='psp_sign' else None,'decode':'sign multiply' if method=='psp_sign' else ('pairwise orthogonal rotation inverse' if method.startswith('mirror') else 'diagonal multiply')}
    bmeta=json.dumps(meta,sort_keys=True).encode();path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('metadata.json',bmeta)
        for name,a in arrays.items():
            b=io.BytesIO();np.save(b,a,allow_pickle=False);z.writestr(f'arrays/{name}.npy',b.getvalue())
    blob=path.read_bytes();return len(blob),hashlib.sha256(blob).hexdigest()

def run(seed,out):
    teachers,data=make_world(seed);tx,ty=data['train']; w,train_wall=fit_independent(seed,tx,ty)
    phases=torch.rand(T,generator=torch.Generator().manual_seed(seed+777))*6.2831853
    phase_code,phase_wall=optimize_phase(w,tx,ty,seed+900)
    dense_code,dense_wall=optimize_dense(w,tx,ty,seed+901)
    # Fixed random-sign PSP context; signs are explicit paid inference state.
    g=torch.Generator().manual_seed(seed+333); bits=torch.randint(0,2,(T,D),generator=g); signs=bits.float()*2-1
    pspshared=(signs*w).sum(0); psprec=signs*pspshared
    phasefixed=retrieve_phase(build_phase(w,phases),phases)
    phaselearned=retrieve_phase(build_phase(w,phase_code),phase_code)
    dshared=(dense_code*w).sum(0); drec=dense_code*dshared
    # One hard-shared classifier trained jointly without task identity.
    tied=nn.Parameter(torch.zeros(D));opt=torch.optim.Adam([tied],lr=.02);g=torch.Generator().manual_seed(seed+222);tied_start=time.perf_counter()
    for _ in range(500):
        ti=torch.randint(T,(128,),generator=g);ix=torch.randint(tx.shape[1],(128,),generator=g)
        loss=nn.functional.binary_cross_entropy_with_logits((tx[ti,ix]*tied).sum(-1),ty[ti,ix]);opt.zero_grad();loss.backward();opt.step()
    tied_wall=time.perf_counter()-tied_start
    methods={'independent':(w,w),'tied':(tied.detach(),tied.detach().expand(T,-1)),'psp_sign':(pspshared,psprec),'mirror_fixed':(build_phase(w,phases),phasefixed),'mirror_learned':(build_phase(w,phase_code),phaselearned),'dense_learned':(dshared,drec)}
    result={'experiment_id':'MA-255','seed':seed,'train_examples':T*tx.shape[1],'base_updates':T*500,'mirror_code_updates':300,'training_wall_seconds':{'independent':train_wall,'mirror_phase':phase_wall,'dense_code':dense_wall,'tied':tied_wall},'methods':{}}
    for name,(shared,recovered) in methods.items():
        m={'quality':evaluate(recovered,*data['test']),'validation':evaluate(recovered,*data['validation'])}
        codes={'psp_sign':signs.numpy(),'mirror_fixed':phases.numpy(),'mirror_learned':phase_code.numpy(),'dense_learned':dense_code.numpy()}.get(name)
        path=out/f'{seed}_{name}.zip';size,digest=archive(path,name,shared,codes,seed)
        m.update({'actual_payload_bytes':size,'payload_sha256':digest,'shared_tensor_elements':int(shared.numel()),'view_arithmetic_proxy':D if name.startswith('mirror') else (D if name=='psp_sign' else (D if name=='dense_learned' else 0))})
        result['methods'][name]=m
    out.mkdir(parents=True,exist_ok=True);(out/f'{seed}_result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();print(json.dumps(run(a.seed,a.out),indent=2))
if __name__=='__main__':main()
