"""Factorized two-coordinate composition for held-out synthetic task combinations."""
from __future__ import annotations
import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn

D=32;PAIRS=((0,0),(1,0),(0,1),(1,1));OBS=(0,1,2);HOLD=3

def rotate(x,theta):
    if x.ndim>1 and theta.ndim==1: theta=theta.unsqueeze(-1)
    q=x.reshape(*x.shape[:-1],D//2,2);c=theta.cos();s=theta.sin();a=q[...,0];b=q[...,1]
    return torch.stack((c*a-s*b,s*a+c*b),-1).reshape_as(x)

def make_world(seed):
    g=torch.Generator().manual_seed(seed);base=torch.randn(D,generator=g);base=base/base.norm()*3
    factors=torch.rand(2,generator=g)*.8+.4;true_angles=torch.tensor([a*factors[0]+b*factors[1] for a,b in PAIRS])
    data={}
    for split,n,off in [('train',1024,100),('validation',2048,300),('test',4096,500)]:
        gg=torch.Generator().manual_seed(seed+off);xs=[];ys=[]
        for t in range(4):
            x=torch.randn(n,D,generator=gg);wt=rotate(base,true_angles[t]);y=((x@wt)>0).float();xs.append(x);ys.append(y)
        data[split]=(torch.stack(xs),torch.stack(ys))
    return base,factors,true_angles,data

def eval_weights(weights,x,y):
    with torch.no_grad():
        logits=torch.einsum('tnd,td->tn',x,weights);acc=((logits>0)==(y>0)).float().mean(-1)
        bce=nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none').mean(-1)
    return {'mean_accuracy':float(acc.mean()),'per_task_accuracy':acc.tolist(),'mean_bce':float(bce.mean()),'per_task_bce':bce.tolist()}

def fit_independent(seed,x,y):
    ws=[];wall=0.
    for t in range(4):
        torch.manual_seed(seed+t);w=nn.Parameter(torch.zeros(D));opt=torch.optim.Adam([w],lr=.015);g=torch.Generator().manual_seed(seed+100+t);st=time.perf_counter()
        for _ in range(1200):
            ix=torch.randint(x.shape[1],(128,),generator=g);loss=nn.functional.binary_cross_entropy_with_logits(x[t,ix]@w,y[t,ix]);opt.zero_grad();loss.backward();opt.step()
        wall+=time.perf_counter()-st;ws.append(w.detach())
    return torch.stack(ws),wall

def fit_shared(seed,x,y,factorized):
    torch.manual_seed(seed+200);w=nn.Parameter(torch.randn(D)*.01)
    code=nn.Parameter(torch.zeros(2 if factorized else 3));opt=torch.optim.Adam([w,code],lr=.015);g=torch.Generator().manual_seed(seed+300);st=time.perf_counter()
    for step in range(1200):
        task=OBS[step%3];a,b=PAIRS[task]
        if factorized: theta=a*code[0]+b*code[1]
        else: theta=code[task]
        ix=torch.randint(x.shape[1],(128,),generator=g);logits=x[task,ix]@rotate(w,theta)
        loss=nn.functional.binary_cross_entropy_with_logits(logits,y[task,ix]);opt.zero_grad();loss.backward();opt.step()
    wall=time.perf_counter()-st
    if factorized: angles=torch.tensor([0.,code[0].item(),code[1].item(),(code[0]+code[1]).item()])
    else: angles=torch.tensor([code[0].item(),code[1].item(),code[2].item(),(code[1]+code[2]-code[0]).item()])
    return w.detach(),angles,wall

def fit_tied(seed,x,y):
    torch.manual_seed(seed+400);w=nn.Parameter(torch.zeros(D));opt=torch.optim.Adam([w],lr=.015);g=torch.Generator().manual_seed(seed+401);st=time.perf_counter()
    for step in range(1200):
        t=OBS[step%3];ix=torch.randint(x.shape[1],(128,),generator=g);loss=nn.functional.binary_cross_entropy_with_logits(x[t,ix]@w,y[t,ix]);opt.zero_grad();loss.backward();opt.step()
    return w.detach(),time.perf_counter()-st

def archive(path,method,shared,codes,seed):
    arr={'shared':np.asarray(shared.detach().cpu(),dtype=np.float16)}
    if method=='psp_sign':arr['context_bits']=np.packbits((np.asarray(codes)>0).astype(np.uint8).reshape(-1))
    elif codes is not None:arr['context_codes']=np.asarray(codes,dtype=np.float16)
    meta={'experiment_id':'MA-257','method':method,'dimension':D,'seed':seed,'composition':'theta11=theta10+theta01-theta00' if method=='additive_task_angles' else 'theta(a,b)=a*thetaA+b*thetaB' if method=='factorized_mirror' else 'none'}
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('metadata.json',json.dumps(meta,sort_keys=True).encode())
        for n,a in arr.items():b=io.BytesIO();np.save(b,a,allow_pickle=False);z.writestr(f'arrays/{n}.npy',b.getvalue())
    blob=path.read_bytes();return len(blob),hashlib.sha256(blob).hexdigest()

def run(seed,out):
    _,true_factors,teacher_angles,data=make_world(seed);tx,ty=data['train'];ind,ind_wall=fit_independent(seed,tx,ty)
    factors,fa,fa_wall=fit_shared(seed,tx,ty,True);additive,aa,aa_wall=fit_shared(seed,tx,ty,False);tied,tied_wall=fit_tied(seed,tx,ty)
    # Native PA16 PSP control on the three observed independently fitted models.
    g=torch.Generator().manual_seed(seed+501);bits=torch.randint(0,2,(3,D),generator=g);signs=bits.float()*2-1;pspshared=(signs*ind[:3]).sum(0);pspdecoded=signs*pspshared
    methods={
      'independent_oracle':(ind,None,ind),
      'tied':(tied,None,tied.expand(4,-1)),
      'psp_sign':(pspshared,signs.numpy(),torch.cat((pspdecoded,torch.zeros(1,D)),0)),
      'factorized_mirror':(factors,fa[1:3].numpy(),rotate(factors.expand(4,-1),fa)),
      'additive_task_angles':(additive,aa[:3].numpy(),rotate(additive.expand(4,-1),aa))}
    result={'experiment_id':'MA-257','seed':seed,'observed_tasks':list(OBS),'heldout_task':HOLD,'true_factor_angles':true_factors.tolist(),'teacher_task_angles':teacher_angles.tolist(),'train_examples':3*tx.shape[1],'optimizer_updates':{'independent':4800,'shared_factorized':1200,'shared_additive':1200,'tied':1200},'wall_seconds':{'independent':ind_wall,'factorized':fa_wall,'additive':aa_wall,'tied':tied_wall},'methods':{}}
    for method,(shared,codes,recovered) in methods.items():
        m={'quality_all_tasks':None if method=='psp_sign' else eval_weights(recovered,*data['test']),'quality_observed_tasks':eval_weights(recovered[:3],(data['test'][0])[:3],(data['test'][1])[:3])}
        codearg=codes[:3] if method=='additive_task_angles' else codes
        path=out/f'{seed}_{method}.zip';size,digest=archive(path,method,shared,codearg,seed)
        m.update({'actual_payload_bytes':size,'payload_sha256':digest})
        if method=='psp_sign':m['heldout_supported']=False
        result['methods'][method]=m
    out.mkdir(parents=True,exist_ok=True);(out/f'{seed}_result.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();print(json.dumps(run(a.seed,a.out),indent=2))
if __name__=='__main__':main()
