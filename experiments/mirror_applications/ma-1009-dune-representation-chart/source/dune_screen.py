#!/usr/bin/env python3
"""Independent small DUNE-z equation implementation for synthetic MA-1009."""
import copy, hashlib, io, json, math, pathlib, time
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
N=32; C=2; CG_STEPS=0; UNROLL=3
_PROJECTION_CACHE={}
_PROJECTION_QR_SECONDS=0.0

def fftc(x): return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x,dim=(-2,-1)),norm='ortho'),dim=(-2,-1))
def ifftc(x): return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x,dim=(-2,-1)),norm='ortho'),dim=(-2,-1))
def chcomplex(x): return torch.view_as_complex(x.permute(0,2,3,1).contiguous())
def complexch(x): return torch.view_as_real(x).permute(0,3,1,2).contiguous()

def make_case(seed,scanner='A'):
    rng=np.random.default_rng(seed); yy,xx=np.mgrid[-1:1:complex(N),-1:1:complex(N)]; image=np.zeros((N,N),np.float32)
    for _ in range(7):
        cx,cy=rng.uniform(-.45,.45,2);a,b=rng.uniform(.1,.4),rng.uniform(.1,.33);ang=rng.uniform(0,math.pi)
        x=(xx-cx)*math.cos(ang)+(yy-cy)*math.sin(ang);y=-(xx-cx)*math.sin(ang)+(yy-cy)*math.cos(ang)
        image += rng.uniform(.1,1)*(((x/a)**2+(y/b)**2)<1)
    image=np.maximum(image,0);image/=max(float(image.max()),1e-8);phi=.08*np.sin(np.pi*xx)*np.cos(np.pi*yy)
    im=np.stack((image*np.cos(phi),image*np.sin(phi)),-1).astype(np.float32)
    shift=0.0 if scanner=='A' else .57;sens=[]
    for c in range(C):
        ang=2*np.pi*c/C+shift;amp=1+.22*np.cos(np.pi*xx*np.cos(ang)+np.pi*yy*np.sin(ang)+shift)
        phase=.35*(xx*np.cos(ang)+yy*np.sin(ang))+(0 if scanner=='A' else .65*np.sin(np.pi*yy+ang))
        sens.append(amp*np.exp(1j*phase))
    sens=torch.tensor(np.asarray(sens),dtype=torch.complex64);sens=sens/torch.sqrt((sens.abs()**2).sum(0,keepdim=True))
    target=complexch(torch.view_as_complex(torch.tensor(im)[None]))
    line=torch.zeros(N,dtype=torch.bool);line[::4]=True;line[N//2-4:N//2+4]=True;mask=line[None,None,:,None].expand(1,1,N,N)
    k=fftc(sens[None]*chcomplex(target)[:,None])*mask
    zf=complexch((sens.conj()[None]*ifftc(k)).sum(1))
    return {'zf':zf,'target':target,'kspace':k,'mask':mask,'sens':sens[None],'scanner':scanner,'seed':seed}

def forward(x,sens,mask): return fftc(sens*chcomplex(x)[:,None])*mask
def adjoint(k,sens): return complexch((sens.conj()*ifftc(k)).sum(1))

def exact_projection(x,k,sens,mask,steps=CG_STEPS):
    """Orthogonal image-space projection onto A(x)=y using a cached QR solve."""
    dtype=x.dtype
    global _PROJECTION_QR_SECONDS
    sk=sens.detach().cpu().contiguous();mk=mask.detach().cpu().contiguous()
    key=hashlib.sha256(sk.numpy().tobytes()+mk.numpy().tobytes()).hexdigest()
    if key not in _PROJECTION_CACHE:
        start=time.perf_counter();n=N*N
        basis=torch.eye(n,dtype=torch.complex128).reshape(n,N,N)
        xb=complexch(basis);sb=sens.detach().to(torch.complex128).expand(n,-1,-1,-1)
        kb=fftc(sb*chcomplex(xb)[:,None])*mask
        selected=mask[0,0].reshape(-1).nonzero().reshape(-1)
        row_idx=torch.cat([selected+c*N*N for c in range(C)])
        A=kb.reshape(n,-1)[:,row_idx].T.contiguous()
        Q,R=torch.linalg.qr(A.mH,mode='reduced')
        _PROJECTION_CACHE[key]=(A,Q,R,row_idx)
        _PROJECTION_QR_SECONDS+=time.perf_counter()-start
    A,Q,R,row_idx=_PROJECTION_CACHE[key]
    xin=chcomplex(x).reshape(-1).to(torch.complex128);y=k.detach().reshape(-1)[row_idx].to(torch.complex128)
    residual=y-A@xin
    u=torch.linalg.solve_triangular(R.mH,residual[:,None],upper=False)
    out=complexch((xin+Q@u[:,0]).reshape(1,N,N))
    return out.to(dtype),k

class DUNE(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Conv2d(2,4,1);self.decoder=nn.Conv2d(4,2,1)
        self.r1=nn.Conv2d(4,8,3,padding=1);self.r2=nn.Conv2d(8,4,3,padding=1)
        self.mirror_angle=nn.Parameter(torch.zeros(()));self.film_scale=nn.Parameter(torch.zeros(4));self.film_bias=nn.Parameter(torch.zeros(4));self.mode='native'
        self.eta=.12;self.reg_scale=.025
    def _transform(self,z,inverse=False):
        a=-self.mirror_angle if inverse else self.mirror_angle;c=torch.cos(a);s=torch.sin(a);parts=[]
        for j in (0,2): parts.extend((c*z[:,j:j+1]-s*z[:,j+1:j+2],s*z[:,j:j+1]+c*z[:,j+1:j+2]))
        return torch.cat(parts,1)
    def regularizer(self,z):
        if self.mode=='mirror': q=self._transform(z)
        elif self.mode=='film': q=z*(1+self.film_scale[None,:,None,None])+self.film_bias[None,:,None,None]
        else: q=z
        r=self.r2(torch.nn.functional.leaky_relu(self.r1(q),.1))
        if self.mode=='mirror': r=self._transform(r,inverse=True)
        return r
    def reconstruct(self,case,project=True):
        z=self.encoder(case['zf'])
        with torch.enable_grad():
            for _ in range(UNROLL):
                z=z.requires_grad_(True);x=self.decoder(z)
                gx=adjoint(forward(x,case['sens'],case['mask'])-case['kspace'],case['sens'])
                vjp=torch.autograd.grad(x,z,grad_outputs=gx,create_graph=self.training,retain_graph=True)[0]
                z=z-self.eta*vjp-self.reg_scale*self.regularizer(z)
            x=self.decoder(z)
        if project: x,_=exact_projection(x,case['kspace'],case['sens'],case['mask'])
        return x
    def forward(self,case,project=True): return self.reconstruct(case,project)

def fit(model,data,updates,adapt=None):
    model.train()
    for p in model.parameters():p.requires_grad_(False)
    if adapt=='mirror':model.mirror_angle.requires_grad_(True)
    elif adapt=='film':model.film_scale.requires_grad_(True);model.film_bias.requires_grad_(True)
    elif adapt in (None,'full'):
        for p in model.parameters():p.requires_grad_(True)
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=1e-3);start=time.perf_counter();loss_last=0
    for step in range(updates):
        q=data[step%len(data)];pred=model(q,project=False);loss=(pred-q['target']).square().mean();opt.zero_grad();loss.backward();opt.step();loss_last=float(loss.detach())
    return {'updates':updates,'examples_seen':updates,'trainable_parameters':sum(p.numel() for p in params),'wall_seconds':time.perf_counter()-start,'last_preprojection_mse':loss_last}

@torch.no_grad()
def metrics(model,data):
    model.eval();rows=[]
    # Build/cache the exact operator QR before isolated per-case latency timing.
    if data:
        warm=model(data[0],project=False)
        exact_projection(warm,data[0]['kspace'],data[0]['sens'],data[0]['mask'])
    start=time.perf_counter()
    for q in data:
        t=time.perf_counter();pre=model(q,project=False);out,kprojected=exact_projection(pre,q['kspace'],q['sens'],q['mask']);elapsed=time.perf_counter()-t
        diff=out-q['target'];nmse=float((diff.square().sum()/(q['target'].square().sum()+1e-12)))
        targ=chcomplex(q['target']);pred=chcomplex(out);mse=float((pred.abs()-targ.abs()).square().mean());psnr=float(20*torch.log10(targ.abs().max()/math.sqrt(max(mse,1e-12))))
        a=pred.abs().flatten();b=targ.abs().flatten();ma=a.mean();mb=b.mean();va=((a-ma)**2).mean();vb=((b-mb)**2).mean();cov=((a-ma)*(b-mb)).mean();ssim=float(((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma**2+mb**2+.01**2)*(va+vb+.03**2)))
        residual=float(((kprojected-q['kspace'])*q['mask']).norm()/(q['kspace'].norm()+1e-12))
        reencoded=float(((forward(out,q['sens'],q['mask'])-q['kspace'])*q['mask']).norm()/(q['kspace'].norm()+1e-12))
        rows.append({'nmse':nmse,'psnr_db':psnr,'global_ssim':ssim,'sample_residual':residual,'image_reencoded_residual':reencoded,'seconds':elapsed})
    return {k:float(np.mean([x[k] for x in rows])) for k in rows[0]}|{'n':len(rows),'wall_total_seconds':time.perf_counter()-start,'projection_qr_seconds_total':_PROJECTION_QR_SECONDS}

def payload(model,kind):
    state={k:v.detach().cpu() for k,v in model.state_dict().items() if k not in ('mirror_angle','film_scale','film_bias')}
    if kind=='independent_pair':state={k:v.detach().cpu() for k,v in model.state_dict().items() if k not in ('mirror_angle','film_scale','film_bias')}
    code=None
    if kind=='mirror':code=model.mirror_angle.detach().cpu()
    elif kind=='film':code={'scale':model.film_scale.detach().cpu(),'bias':model.film_bias.detach().cpu()}
    obj={'shared_model':state,'scanner_b_code':code,'operator_metadata':'normalized analytic 2-coil maps supplied per scan; 4x Cartesian mask; centered orthonormal FFT','architecture':'learned 2-to-4 1x1 encoder; shared 4->8->4 latent regularizer; 4-to-2 1x1 decoder; 3 steps; exact decoder VJP; QR orthogonal image-space projection','dtype':'float32'}
    b=io.BytesIO();torch.save(obj,b);raw=b.getvalue();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def main():
    torch.manual_seed(1009001);np.random.seed(1009001)
    src=[make_case(1009001+i,'A') for i in range(24)];support=[make_case(1009101+i,'B') for i in range(8)];dev=[make_case(1009201+i,'B') for i in range(16)]
    base=DUNE();pre=fit(base,src,48);native=copy.deepcopy(base)
    mirror=copy.deepcopy(base);mirror.mode='mirror';fit_m=fit(mirror,support,48,'mirror')
    film=copy.deepcopy(base);film.mode='film';fit_f=fit(film,support,48,'film')
    upper=copy.deepcopy(base);upper.mode='native';fit_u=fit(upper,support,48,'full')
    models={'native_shared_dune':native,'ordinary_latent_film':film,'latent_givens_mirror':mirror,'independent_scanner_b_finetune':upper}
    scores={k:metrics(v,dev) for k,v in models.items()};payloads={k:payload(v,'mirror' if k=='latent_givens_mirror' else 'film' if k=='ordinary_latent_film' else 'shared' if k=='native_shared_dune' else 'independent_pair') for k,v in models.items()}
    base_payload=payload(native,'shared');shared_state={k:v.detach().cpu() for k,v in native.state_dict().items() if k not in ('mirror_angle','film_scale','film_bias')};meta='2 scanner domains; analytic sensitivity maps are supplied per scan; 4x Cartesian mask; centered orthonormal FFT';arch='DUNE-z: 2-to-4 encoder; shared 4-8-4 latent prior; 4-to-2 decoder; 3 exact-VJP steps; QR image projection'
    def deploy(scanner_models,scanner_codes):
        obj={'scanner_models':scanner_models,'scanner_codes':scanner_codes,'operator_metadata':meta,'architecture':arch,'dtype':'float32'}
        bio=io.BytesIO();torch.save(obj,bio);return {'bytes':len(bio.getvalue()),'sha256':hashlib.sha256(bio.getvalue()).hexdigest()}
    hard_deploy=deploy({'shared':shared_state},{'B':None})
    independent_deploy=deploy({'A':shared_state,'B':{k:v.detach().cpu() for k,v in upper.state_dict().items() if k not in ('mirror_angle','film_scale','film_bias')}},{'B':None})
    mirror_deploy=deploy({'shared':shared_state},{'B':mirror.mirror_angle.detach().cpu()})
    film_deploy=deploy({'shared':shared_state},{'B':{'scale':film.film_scale.detach().cpu(),'bias':film.film_bias.detach().cpu()}})
    hard_total=hard_deploy['bytes'];ind_total=independent_deploy['bytes'];mirror_total=mirror_deploy['bytes'];film_total=film_deploy['bytes']
    A,Q,R,row_idx=next(iter(_PROJECTION_CACHE.values()));qr_state_bytes=sum(t.numel()*t.element_size() for t in (A,Q,R,row_idx))
    out={'protocol':'MA-1009 amendment 07; underdetermined two-coil synthetic DUNE-z screen','official_repo_commit':'8e62d1b9ca193060d09f5ac4e866dc4a6af28347','official_repo_code_status':'README-only, no license/checkpoint observed; independent implementation','pretraining':pre,'adaptations':{'mirror':fit_m,'film':fit_f,'full_finetune':fit_u},'metrics':scores,'payloads':payloads,'deployment_payloads':{'shared_source_model_bytes':base_payload['bytes'],'source_only_two_scanner_hard_shared_bytes':hard_total,'source_plus_scanner_b_mirror_code_bytes':mirror_total,'source_plus_scanner_b_film_bytes':film_total,'source_plus_independent_scanner_b_model_bytes':ind_total,'mirror_saving_vs_independent_fraction':1-mirror_total/ind_total,'film_saving_vs_independent_fraction':1-film_total/ind_total,'payload_hashes':{'hard_shared':hard_deploy['sha256'],'mirror':mirror_deploy['sha256'],'film':film_deploy['sha256'],'independent':independent_deploy['sha256']}},'compute_proxy':{'unroll_steps':UNROLL,'final_cg_steps':CG_STEPS,'cnn_macs_per_step_per_case':9*(4*8+8*4)*N*N,'fft_forward_calls_per_case':UNROLL+2,'fft_adjoint_calls_per_case':UNROLL+1,'projection':'orthogonal image-space projection via cached QR; common to all methods','projection_qr_build_wall_seconds':_PROJECTION_QR_SECONDS,'projection_qr_ephemeral_state_bytes':qr_state_bytes},'seed_protocol':{'source_seeds':list(range(1009001,1009025)),'support_seeds':list(range(1009101,1009109)),'dev_seeds':list(range(1009201,1009217)),'fresh_seeds':[1009301,1009307,1009313],'fresh_opened':False},'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    rd=pathlib.Path(__file__).parents[1]/'runs';torch.save({'native':native.state_dict(),'mirror':mirror.state_dict(),'film':film.state_dict(),'upper':upper.state_dict()},rd/'development_models.pt');out['checkpoint_sha256']=hashlib.sha256((rd/'development_models.pt').read_bytes()).hexdigest();(rd/'development.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
