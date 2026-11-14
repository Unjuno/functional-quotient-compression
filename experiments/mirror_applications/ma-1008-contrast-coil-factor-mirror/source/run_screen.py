#!/usr/bin/env python3
"""Synthetic, physics-explicit MA-1008 development screen (CPU PyTorch)."""
import copy, hashlib, io, json, math, pathlib, time
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
DEVICE = 'cpu'
SIZE = 32
COILS = 4
MASK = None

def case(seed, contrast, scanner):
    """Analytic complex phantom and explicit multicoil Cartesian acquisition."""
    rng=np.random.default_rng(seed); yy,xx=np.mgrid[-1:1:complex(SIZE),-1:1:complex(SIZE)]
    base=np.zeros((SIZE,SIZE),np.float32)
    for _ in range(6):
        cx,cy=rng.uniform(-.45,.45,2); a,b=rng.uniform(.12,.4),rng.uniform(.1,.32); ang=rng.uniform(0,math.pi)
        x=(xx-cx)*math.cos(ang)+(yy-cy)*math.sin(ang); y=-(xx-cx)*math.sin(ang)+(yy-cy)*math.cos(ang)
        base += rng.uniform(.15,1.0)*(((x/a)**2+(y/b)**2)<1)
    base=np.maximum(base,0); base/=max(float(base.max()),1e-6)
    if contrast:
        # Two distinct, deterministic contrast transfer functions.
        tissue=np.exp(-((xx+.18-.12*contrast)**2+(yy-.12)**2)/.18)
        gamma=(.72,.88)[contrast-1]
        image=np.clip((1-.22*contrast)*base + .22*contrast*base**gamma + .10*contrast*tissue*base,0,1).astype(np.float32)
    else: image=base.astype(np.float32)
    phi=.08*np.sin(np.pi*xx)*np.cos(np.pi*yy)
    im=np.stack([image*np.cos(phi),image*np.sin(phi)],-1).astype(np.float32)
    sens=[]; shift=.57*scanner
    for c in range(COILS):
        ang=2*np.pi*c/COILS+shift
        amp=1+.22*np.cos(np.pi*xx*np.cos(ang)+np.pi*yy*np.sin(ang)+shift)
        phase=.35*(xx*np.cos(ang)+yy*np.sin(ang))+.65*scanner*np.sin(np.pi*yy+ang)
        sens.append(np.stack([amp*np.cos(phase),amp*np.sin(phase)],-1))
    sens=np.asarray(sens,np.float32); denom=np.sqrt(np.sum(np.sum(sens*sens,-1),0))
    sens=sens/np.maximum(denom[None,:,:,None],1e-6)
    it=torch.from_numpy(im)[None,None]; st=torch.from_numpy(sens)[None]
    coil=torch.view_as_complex(st.contiguous())*torch.view_as_complex(it.contiguous())
    ks=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(coil,dim=(-2,-1)),norm='ortho'),dim=(-2,-1))
    mask=torch.zeros((SIZE,SIZE),dtype=torch.bool); mask[::4,:]=True; mask[SIZE//2-4:SIZE//2+4,:]=True
    km=ks*mask
    # Same explicit SENSE adjoint input for every candidate.
    zf=(torch.view_as_complex(st).conj()*torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(km,dim=(-2,-1)),norm='ortho'),dim=(-2,-1))).sum(1)
    inp=torch.view_as_real(zf)[0].permute(2,0,1).contiguous()
    target=torch.from_numpy(im).permute(2,0,1).contiguous()
    return {'x':inp,'y':target,'ks':ks[0],'mask':mask,'sens':torch.view_as_complex(st[0]),'contrast':contrast,'scanner':scanner,'seed':seed}

class Prior(nn.Module):
    def __init__(self):
        super().__init__(); self.c1=nn.Conv2d(2,12,3,padding=1); self.c2=nn.Conv2d(12,12,3,padding=1); self.c3=nn.Conv2d(12,2,3,padding=1)
        self.mode='hard'; self.factor=nn.Parameter(torch.zeros(5)); self.direct=nn.Parameter(torch.zeros(1))
    def forward(self,x,contrast=0,scanner=0):
        h=self.c1(x)
        if self.mode in ('mirror','direct','pair_mirror'):
            if self.mode=='mirror': a=self.factor[0]+(self.factor[1] if contrast==1 else self.factor[2] if contrast==2 else 0)+(self.factor[3] if scanner==1 else self.factor[4] if scanner==2 else 0)
            else: a=self.direct[0]
            u,v=h[:,0:1],h[:,1:2]; c=torch.cos(a); s=torch.sin(a)
            h=torch.cat((c*u-s*v,s*u+c*v,h[:,2:]),1)
        elif self.mode=='scalar': h=h*(1+self.factor[0]+(self.factor[1] if contrast==1 else self.factor[2] if contrast==2 else 0)+(self.factor[3] if scanner==1 else self.factor[4] if scanner==2 else 0))
        elif self.mode=='pair_scalar': h=h*(1+self.direct[0])
        h=torch.relu(h); h=torch.relu(self.c2(h)); return x+self.c3(h)

def train(model, data, updates, lr=1e-3, adapt=None):
    model.train()
    if adapt:
        for p in model.parameters(): p.requires_grad_(False)
        model.factor.requires_grad_(adapt=='factor')
        model.direct.requires_grad_(adapt=='direct')
        if adapt=='full':
            for p in model.parameters(): p.requires_grad_(True)
    ps=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(ps,lr=lr); t=time.perf_counter(); n=0
    for step in range(updates):
        q=data[step%len(data)]; pred=model(q['x'][None],q['contrast'],q['scanner']); loss=(pred-q['y'][None]).square().mean()
        opt.zero_grad(); loss.backward(); opt.step(); n+=1
    return {'updates':updates,'examples_seen':n,'trainable_parameters':sum(p.numel() for p in ps),'wall_seconds':time.perf_counter()-t}

@torch.no_grad()
def evaluate(model,data):
    model.eval(); nm=[]; ps=[]; coil=[]; wall=[]
    for q in data:
        t=time.perf_counter(); out_last=model(q['x'][None],q['contrast'],q['scanner'])[0].permute(1,2,0).contiguous()
        pred=torch.view_as_complex(out_last); out=out_last.permute(2,0,1).contiguous(); sens=q['sens']; k=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(sens*pred[None],dim=(-2,-1)),norm='ortho'),dim=(-2,-1));
        resid=((k-q['ks'])*q['mask']).norm()/(q['ks']*q['mask']).norm().clamp_min(1e-12)
        err=(out-q['y']).square().sum()/(q['y'].square().sum()+1e-12); mse=(out-q['y']).square().mean()
        nm.append(float(err)); ps.append(float(20*torch.log10(q['y'].abs().max()/mse.sqrt().clamp_min(1e-12)))); coil.append(float(resid)); wall.append(time.perf_counter()-t)
    return {'nmse':float(np.mean(nm)),'psnr_db':float(np.mean(ps)),'measured_coil_residual':float(np.mean(coil)),'inference_seconds_per_case':float(np.mean(wall)),'n':len(data)}

def payload(model,method):
    obj={'method':method,'state':{k:v.detach().cpu() for k,v in model.state_dict().items() if k not in ('factor','direct')},'arch':'32x32 complex residual CNN 2-12-12-2','operator':'4-coil normalized analytic sensitivities; 4x Cartesian mask; centered orthonormal FFT'}
    if method=='factorized_scalar_film': obj['factorized_scalar_coordinates']=model.factor.detach().cpu()
    if method=='factorized_givens_mirror': obj['contrast_coil_givens_coordinates']=model.factor.detach().cpu()
    if method in ('direct_pair_givens','direct_pair_scalar'): obj['pair_coordinate']=model.direct.detach().cpu()
    b=io.BytesIO();torch.save(obj,b);return len(b.getvalue()),hashlib.sha256(b.getvalue()).hexdigest()

def serialize(obj):
    b=io.BytesIO();torch.save(obj,b);return {'bytes':len(b.getvalue()),'sha256':hashlib.sha256(b.getvalue()).hexdigest()}

def main():
    torch.manual_seed(1008001); np.random.seed(1008001)
    pairs=[(c,k) for c in range(3) for k in range(3) if (c,k)!=(2,2)]
    src=[case(1008001+100*c+10*k+i,c,k) for c,k in pairs for i in range(4)]
    base=Prior(); pre=train(base,src,240); frozen=copy.deepcopy(base)
    supports={(c,k):[case(1008101+100*c+10*k+i,c,k) for i in range(4)] for c,k in pairs}
    # Fit per-pair scalar/Givens codes on observed pairs, then project the 8-code bank
    # onto an additive contrast+coil chart. (2,2) is never used in fitting.
    fitted={}; codefits={}
    for family, mode in [('givens','direct'),('scalar','pair_scalar')]:
        vals={}
        for c,k in pairs:
            m=copy.deepcopy(frozen); m.mode=mode
            fit=train(m,supports[(c,k)],80,adapt='direct')
            vals[(c,k)]=float(m.direct.detach()); codefits[f'{family}_{c}_{k}']=fit
        # Gauge-fixed chart: global offset + two non-reference contrast and coil codes.
        X=[]; y=[]
        for c,k in pairs:
            X.append([1.0, float(c==1), float(c==2), float(k==1), float(k==2)]); y.append(vals[(c,k)])
        coef=np.linalg.lstsq(np.asarray(X),np.asarray(y),rcond=None)[0]
        m=copy.deepcopy(frozen);m.factor.data.copy_(torch.tensor(coef,dtype=m.factor.dtype));m.mode='mirror' if family=='givens' else 'scalar'
        fitted[family]={'model':m,'pair_values':{f'{c},{k}':v for (c,k),v in vals.items()},'factor_coordinates':coef.tolist(),'source_pair_fit_wall_seconds':sum(v['wall_seconds'] for key,v in codefits.items() if key.startswith(family+'_')),'source_pair_fit_updates':sum(v['updates'] for key,v in codefits.items() if key.startswith(family+'_'))}
    dev=[case(1008201+i,2,2) for i in range(8)]
    target_support=[case(1008251+i,2,2) for i in range(4)]
    models={'hard_shared_frozen':copy.deepcopy(frozen),'factorized_scalar_film':fitted['scalar']['model'],'factorized_givens_mirror':fitted['givens']['model']}
    models['factorized_givens_mirror'].mode='mirror';models['factorized_scalar_film'].mode='scalar'
    models['direct_pair_givens_target_adapt']=copy.deepcopy(frozen);models['direct_pair_givens_target_adapt'].mode='direct'
    models['direct_pair_givens_target_adapt'].direct.data.zero_()
    adapt_target={'direct_pair_givens_target_adapt':train(models['direct_pair_givens_target_adapt'],target_support,80,adapt='direct')}
    models['independent_target_finetune']=copy.deepcopy(frozen);adapt_target['independent_target_finetune']=train(models['independent_target_finetune'],target_support,80,adapt='full')
    metrics={k:evaluate(m,dev) for k,m in models.items()}
    known_dev={(c,k):[case(1008401+100*c+10*k+i,c,k) for i in range(4)] for c,k in pairs}
    known_retention={}
    for fam, fit in fitted.items():
        approx=fit['model']; approx.mode='mirror' if fam=='givens' else 'scalar'; approx_rows=[]; direct_rows=[]
        for c,k in pairs:
            approx_rows.append(evaluate(approx,known_dev[(c,k)]))
            dm=copy.deepcopy(frozen); dm.mode='direct' if fam=='givens' else 'pair_scalar'; dm.direct.data.fill_(fit['pair_values'][f'{c},{k}']); direct_rows.append(evaluate(dm,known_dev[(c,k)]))
        known_retention[fam]={'factorized_mean_nmse':float(np.mean([x['nmse'] for x in approx_rows])),'direct_code_mean_nmse':float(np.mean([x['nmse'] for x in direct_rows])),'factorized_per_cell_nmse':[x['nmse'] for x in approx_rows],'direct_per_cell_nmse':[x['nmse'] for x in direct_rows]}
    payloads={k:{'bytes':payload(m,'factorized_givens_mirror' if k=='factorized_givens_mirror' else 'factorized_scalar_film' if k=='factorized_scalar_film' else 'direct_pair_givens' if k.startswith('direct_pair') else k)[0]} for k,m in models.items()}
    shared={k:v.detach().cpu() for k,v in frozen.state_dict().items() if k not in ('factor','direct')}
    common={'prior':shared,'operator':'4-coil analytic sensitivity maps; 4x Cartesian mask; centered orthonormal FFT','architecture':'complex residual CNN 2-12-12-2','grid':'3x3; omit cell (2,2)'}
    bank=serialize({**common,'coordinates':torch.tensor([fitted['givens']['pair_values'][f'{c},{k}'] for c,k in pairs],dtype=torch.float32)})
    factor=serialize({**common,'coordinates':fitted['givens']['model'].factor.detach().cpu()})
    run_dir=pathlib.Path(__file__).parents[1]/'runs'; checkpoint=run_dir/'development_models.pt'
    torch.save({'shared':shared,'mirror_coordinates':fitted['givens']['model'].factor.detach().cpu(),'scalar_coordinates':fitted['scalar']['model'].factor.detach().cpu(),'target_direct_angle':models['direct_pair_givens_target_adapt'].direct.detach().cpu(),'target_full':{k:v.detach().cpu() for k,v in models['independent_target_finetune'].state_dict().items()}},checkpoint)
    checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    out={'protocol':'MA-1008 amendment 01; development only; one heldout cell (contrast 2, coil 2)','source_pretrain':pre,'observed_pair_code_fits':codefits,'factorization':{k:{q:v for q,v in obj.items() if q!='model'} for k,obj in fitted.items()},'known_cell_retention':known_retention,'target_adaptation':adapt_target,'metrics':metrics,'payloads':payloads,'bank_storage_comparison':{'direct_eight_pair_givens_bank':bank,'shared_prior_plus_five_factor_givens':factor,'compression_fraction':1-factor['bytes']/bank['bytes']},'device':DEVICE,'source_pairs':pairs,'source_examples':len(src),'support_examples_per_observed_pair':4,'target_support_examples':len(target_support),'development_examples':len(dev),'source_updates':240,'updates_per_observed_pair_code':80,'target_adaptation_updates':80,'fresh_seeds':[1008301,1008307,1008313],'fresh_opened':False,'checkpoint':'development_models.pt','checkpoint_sha256':checkpoint_sha,'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    p=run_dir/'development.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
