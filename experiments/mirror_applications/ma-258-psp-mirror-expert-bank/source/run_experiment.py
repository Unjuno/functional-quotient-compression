"""Clean MA-258 rerun: compare expert bank sharing, PSP, SVD and Mirror orbit codes."""
from __future__ import annotations
import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
import torch

E=8;R=16;P=R*R

def rotate(v,theta):
    if v.ndim>1 and theta.ndim==1:theta=theta.unsqueeze(-1)
    q=v.reshape(*v.shape[:-1],P//2,2);c=theta.cos();s=theta.sin();a=q[...,0];b=q[...,1]
    return torch.stack((c*a-s*b,s*a+c*b),-1).reshape_as(v)

def make_world(seed,regime):
    g=torch.Generator().manual_seed(seed)
    if regime=='aligned':
        base=torch.randn(P,generator=g);angles=torch.rand(E,generator=g)*1.4-.7;weights=rotate(base.expand(E,-1),angles)
    else:
        weights=torch.randn(E,P,generator=g);weights=weights/weights.norm(dim=-1,keepdim=True)*P**.5;angles=None
    return weights.reshape(E,R,R),angles

def fit_mirror(weights,seed):
    torch.manual_seed(seed);base=torch.nn.Parameter(weights.mean(0).reshape(-1).clone());angles=torch.nn.Parameter(torch.zeros(E));opt=torch.optim.Adam([base,angles],lr=.01);start=time.perf_counter()
    target=weights.reshape(E,P)
    for _ in range(2000):
        recon=rotate(base.expand(E,-1),angles);loss=(recon-target).square().mean();opt.zero_grad();loss.backward();opt.step()
    return base.detach(),angles.detach(),time.perf_counter()-start

def fit_svd(weights):
    flat=weights.reshape(E,P);mean=flat.mean(0);center=flat-mean;_,_,vh=torch.linalg.svd(center,full_matrices=False);basis=vh[:2];coeff=center@basis.T;return mean,basis,coeff

def archive(path,method,objects,seed,regime):
    arr={k:np.asarray(v.detach().cpu() if torch.is_tensor(v) else v,dtype=np.float16) for k,v in objects.items() if k!='context_bits'}
    if 'context_bits' in objects:arr['context_bits']=np.asarray(objects['context_bits'],dtype=np.uint8)
    meta={'experiment_id':'MA-258','method':method,'regime':regime,'seed':seed,'experts':E,'weight_shape':[R,R],'dtype':'FP16','decode':{'independent':'direct expert weight bank','tied':'repeat common matrix','psp':'unpack signs and multiply superposed vector','svd_rank2':'mean + coefficient times shared basis','mirror_orbit':'rotate shared matrix by stored expert angle'}[method]}
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('metadata.json',json.dumps(meta,sort_keys=True).encode())
        for k,a in arr.items():b=io.BytesIO();np.save(b,a,allow_pickle=False);z.writestr(f'arrays/{k}.npy',b.getvalue())
    raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()

def decode(path):
    with zipfile.ZipFile(path) as z:
        meta=json.loads(z.read('metadata.json'));a={Path(n).stem:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')}
    m=meta['method']
    if m=='independent':rec=torch.from_numpy(a['weights'].astype(np.float32)).reshape(E,P)
    elif m=='tied':rec=torch.from_numpy(a['weight'].astype(np.float32)).reshape(1,P).expand(E,-1)
    elif m=='psp':
        signs=torch.from_numpy((np.unpackbits(a['context_bits'])[:E*P].reshape(E,P).astype(np.float32)*2)-1);shared=torch.from_numpy(a['superposed'].astype(np.float32));rec=signs*shared
    elif m=='svd_rank2':
        mean=torch.from_numpy(a['mean'].astype(np.float32));basis=torch.from_numpy(a['basis'].astype(np.float32));coeff=torch.from_numpy(a['coeff'].astype(np.float32));rec=mean+coeff@basis
    else:
        base=torch.from_numpy(a['base'].astype(np.float32));angles=torch.from_numpy(a['angles'].astype(np.float32));rec=rotate(base.expand(E,-1),angles)
    return meta,a,rec.reshape(E,R,R)

def score(recon,target,seed):
    diff=(recon-target).square().mean((1,2));den=target.square().mean((1,2)).clamp_min(1e-12);rel=diff/den
    g=torch.Generator().manual_seed(seed+8000);x=torch.randn(E,512,R,generator=g);yp=torch.einsum('eni,eoi->eno',x,recon);yt=torch.einsum('eni,eoi->eno',x,target);outmse=(yp-yt).square().mean((1,2))
    return {'mean_normalized_mse':float(rel.mean()),'max_expert_normalized_mse':float(rel.max()),'per_expert_normalized_mse':rel.tolist(),'mean_random_input_output_mse':float(outmse.mean()),'per_expert_output_mse':outmse.tolist()}

def run(seed,regime,split,out):
    weights,teacher_angles=make_world(seed,regime);base,angles,fit_wall=fit_mirror(weights,seed+900)
    mean,basis,coeff=fit_svd(weights);g=torch.Generator().manual_seed(seed+333);bits=torch.randint(0,2,(E,P),generator=g);signs=bits.float()*2-1;sup=(signs*weights.reshape(E,P)).sum(0)
    objects={
      'independent':{'weights':weights},
      'tied':{'weight':weights.mean(0)},
      'psp':{'superposed':sup,'context_bits':np.packbits(bits.numpy().reshape(-1))},
      'svd_rank2':{'mean':mean,'basis':basis,'coeff':coeff},
      'mirror_orbit':{'base':base,'angles':angles}}
    names={'psp':'psp','svd_rank2':'svd_rank2','mirror_orbit':'mirror_orbit','independent':'independent','tied':'tied'};methods={}
    for name,obj in objects.items():
        path=out/f'{regime}_{seed}_{name}.zip';size,digest=archive(path,name,obj,seed,regime);_,_,decoded=decode(path);q=score(decoded,weights,seed)
        methods[name]={'quality':q,'actual_payload_bytes':size,'payload_sha256':digest,'reconstructed_parameters':int(decoded.numel())}
    result={'experiment_id':'MA-258','seed':seed,'split':split,'regime':regime,'experts':E,'matrix_shape':[R,R],'weight_bank_parameters':E*P,'examples_used_for_code_fit':0,'mirror_optimizer_updates':2000,'mirror_fit_wall_seconds':fit_wall,'test_examples_per_expert':512,'teacher_angles':teacher_angles.tolist() if teacher_angles is not None else None,'methods':methods,'scope_note':'oracle weight-space compression; no task-example learning or Transformer/MoE benchmark'}
    out.mkdir(parents=True,exist_ok=True);(out/f'{regime}_{seed}_result.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--regime',choices=('aligned','unrelated'),required=True);ap.add_argument('--split',choices=('development','fresh'),required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();print(json.dumps(run(a.seed,a.regime,a.split,a.out),indent=2))
if __name__=='__main__':main()
