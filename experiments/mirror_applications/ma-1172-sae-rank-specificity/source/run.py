"""MA-1172: independently preregistered Mirror-vs-generic-nonlinearity SAE screen.

Train objective sees unlabeled activation x ONLY; oracle feature IDs and dictionary
exist for evaluator-only support F1 after checkpoints frozen by dev NMSE.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, os, platform, statistics, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1]
D,FNUM,K=32,64,3
SEEDS=[101,103,107]
WORLDS=['iid_independent_dictionary','nonlinear_orbit_dictionary_correlated_pairs']
METHODS=['affine_loop','tanh_plain_loop','mirror_loop','poly_plain_loop','dense_onepass']
UPDATES=500
NUM_THREADS=1


def fixed_maps():
    g=torch.Generator().manual_seed(1171)
    perms=torch.stack([torch.randperm(D,generator=g) for _ in range(8)])
    signs=torch.stack([2*torch.randint(0,2,(D,),generator=g).float()-1 for _ in range(8)])
    keys=F.normalize(torch.randn(8,6,generator=g),dim=-1)
    v=F.normalize(torch.randn(D,generator=g),dim=0)
    H=torch.eye(D)-2*torch.outer(v,v)
    return perms,signs,keys,H


def sparse_relu_topk(x:torch.Tensor):
    x=F.relu(x)
    val,idx=torch.topk(x,K,dim=-1)
    return torch.zeros_like(x).scatter(-1,idx,val)


class SAE(nn.Module):
    def __init__(self,method):
        super().__init__()
        assert method in METHODS
        self.method=method
        self.decoder_raw=nn.Parameter(torch.randn(FNUM,D)*D**-0.5)
        self.outbias=nn.Parameter(torch.zeros(D))
        self.enc_bias=nn.Parameter(torch.zeros(FNUM))
        if method=='dense_onepass':
            self.enc_raw=nn.Parameter(torch.randn(FNUM,D)*D**-0.5)
        else:
            self.core=nn.Parameter(torch.randn(D)*0.7)
            self.address=nn.Parameter(torch.randn(FNUM,6)*0.5)
            self.gain=nn.Parameter(torch.zeros(FNUM))
            pm,sg,keys,h=fixed_maps()
            self.register_buffer('perms',pm,persistent=False)
            self.register_buffer('signs',sg,persistent=False)
            self.register_buffer('keys',keys,persistent=False)
            self.register_buffer('householder',h,persistent=False)
        n=sum(p.numel() for p in self.parameters())
        assert n==(4192 if method=='dense_onepass' else 2624),(method,n)

    def encoder_matrix(self,nonlinear_override=None):
        if self.method=='dense_onepass':
            return F.normalize(self.enc_raw,dim=-1)
        pages=self.core[self.perms]*self.signs
        mix=torch.softmax((self.address @ self.keys.T)*2.5,dim=-1)
        v=mix@pages
        mode=nonlinear_override or self.method
        if mode=='mirror_loop':
            H=self.householder
            v=(torch.tanh(2.5*(v@H))@H.T)/2.5
        elif mode=='tanh_plain_loop':
            v=torch.tanh(2.5*v)/2.5
        elif mode=='poly_plain_loop':
            v=v+0.5*v.pow(3)
        elif mode=='linear_rot_invert':
            v=(v@self.householder)@self.householder.T
        elif mode=='affine_loop':
            pass
        else:raise ValueError(mode)
        return F.normalize(v,dim=-1)*(F.softplus(self.gain)+0.1)[:,None]

    def matrices(self):
        return self.encoder_matrix(),F.normalize(self.decoder_raw,dim=-1)

    def encode_reconstruct(self,x):
        e,d=self.matrices()
        z=sparse_relu_topk(x@e.T+self.enc_bias)
        if self.method!='dense_onepass':
            for _ in range(2):
                residual=x-(z@d+self.outbias)
                z=sparse_relu_topk(z+0.4*(residual@e.T))
        return z,z@d+self.outbias

    def forward(self,x):return self.encode_reconstruct(x)[1]


def teacher(seed,world):
    g=torch.Generator().manual_seed(seed*34117+(100001 if world==WORLDS[0] else 200003))
    if world==WORLDS[0]:
        W=torch.randn(FNUM,D,generator=g)
    else:
        bases=3.0*torch.randn(8,D,generator=g)
        mixture=torch.softmax(2.0*torch.randn(FNUM,8,generator=g),dim=-1)
        q,r=torch.linalg.qr(torch.randn(D,D,generator=g))
        q=q*torch.sign(torch.diagonal(r).clamp(min=0)*2-1).unsqueeze(0)
        W=torch.tanh(2.5*(mixture@bases@q))@q.T/2.5
    return F.normalize(W,dim=-1)


def dataset(seed,world,split,n):
    code={'train':0,'dev':1,'fresh_id':2,'fresh_ood':3}[split]
    g=torch.Generator().manual_seed(seed*1013+code*71+(0 if world==WORLDS[0] else 98701))
    if world==WORLDS[1] and split!='fresh_ood':
        anchor=torch.randint(0,FNUM,(n,),generator=g)
        partner=anchor^1
        replace=torch.rand(n,generator=g)>.75
        partner[replace]=torch.randint(0,FNUM,(int(replace.sum()),),generator=g)
        third=torch.randint(0,FNUM,(n,),generator=g)
        ids=torch.stack([anchor,partner,third],dim=-1)
        for j in (1,2):
            conflict=(ids[:,j:j+1]==ids[:,:j]).any(-1)
            while bool(conflict.any()):
                ids[conflict,j]=torch.randint(0,FNUM,(int(conflict.sum()),),generator=g)
                conflict=(ids[:,j:j+1]==ids[:,:j]).any(-1)
    else:
        ids=torch.topk(torch.rand(n,FNUM,generator=g),K,dim=-1).indices
    amp=0.65+0.7*torch.rand(n,K,generator=g)
    gt=teacher(seed,world)
    x=(gt[ids]*amp[...,None]).sum(1)+0.01*torch.randn(n,D,generator=g)
    return x,ids


def nmse(y,x):return float(((y-x)**2).mean()/(x.square().mean()+1e-12))


def support_f1(z,ids,d,gt):
    score=(gt@F.normalize(d,dim=-1).T).cpu().numpy()
    rows,cols=linear_sum_assignment(-score)
    match=np.zeros(FNUM,dtype=np.int64)
    match[rows]=cols
    pred=(z[:,match]>1e-8)
    truth=torch.zeros_like(pred)
    truth.scatter_(1,ids,True)
    tp=int((truth&pred).sum());fp=int((~truth&pred).sum());fn=int((truth&~pred).sum())
    return 2*tp/max(1,2*tp+fp+fn),float(score[rows,cols].mean()),float(pred.float().sum(-1).mean())


def spectrum(E):
    s=torch.linalg.svdvals(E.double())
    r=s/(s[0]+1e-300)
    p=s.square()/s.square().sum()
    er=float(torch.exp(-torch.sum(p*torch.log(p+1e-300))))
    return {'rank_rel_1e3':int((r>1e-3).sum()),'rank_rel_1e2':int((r>1e-2).sum()),
            'rank_rel_1e1':int((r>1e-1).sum()),'effective_rank':er,
            'energy_top8':float(p[:8].sum()),'sv8_ratio':float(r[7]),'sv9_ratio':float(r[8]),
            'sv32_ratio':float(r[-1]),'matrix_fro':float(torch.linalg.norm(E.double()))}


def packet(model):
    b=io.BytesIO();torch.save({'method':model.method,'fixed_program_seed':1171,'state':model.state_dict()},b)
    return b.getvalue()


def save_csv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def train():
    torch.set_num_threads(1)
    torch.manual_seed(1172)
    (ROOT/'checkpoints').mkdir(exist_ok=True)
    (ROOT/'results').mkdir(exist_ok=True)
    rows=[]
    for world in WORLDS:
      for seed in SEEDS:
        xtrain,_=dataset(seed,world,'train',8192)
        xdev,_=dataset(seed,world,'dev',1024)
        scale=xtrain.square().mean()
        for method in METHODS:
            torch.manual_seed(seed*777 + (0 if world==WORLDS[0] else 50000))
            model=SAE(method)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-5)
            sampler=torch.Generator().manual_seed(seed*3000+(0 if world==WORLDS[0] else 400))
            best=float('inf');bstate=None;bstep=0;curve=[]
            for step in range(1,UPDATES+1):
                ids=torch.randint(0,len(xtrain),(160,),generator=sampler)
                optimizer.zero_grad(set_to_none=True)
                y=model(xtrain[ids]);loss=(y-xtrain[ids]).square().mean()/scale
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                loss.backward();optimizer.step()
                if step%100==0:
                    model.eval()
                    with torch.no_grad():v=nmse(model(xdev),xdev)
                    model.train();curve.append([step,v])
                    if v<best:
                        best=v;bstep=step
                        bstate={k:v.detach().clone() for k,v in model.state_dict().items()}
            assert bstate is not None
            model.load_state_dict(bstate)
            data=packet(model)
            name=f'{world}_{seed}_{method}.pt'
            (ROOT/'checkpoints'/name).write_bytes(data)
            rows.append({'world':world,'seed':seed,'method':method,'dev_NMSE':best,'best_step':bstep,
                         'trainable_params':sum(p.numel() for p in model.parameters()),
                         'checkpoint_sha256':hashlib.sha256(data).hexdigest(),'packet_bytes':len(data),
                         'updates':UPDATES,'train_examples':8192,'batch':160,'curve':json.dumps(curve)})
            print('TRAIN',world,seed,method,'dev',round(best,5),'step',bstep,flush=True)
    save_csv(ROOT/'results'/'DEV_RESULTS.csv',rows)
    (ROOT/'results'/'TRAIN_COMPLETE.json').write_text(json.dumps({'trained':len(rows),'fresh_never_evaluated':True,'training_updates':UPDATES,'seeds':SEEDS,'worlds':WORLDS,'methods':METHODS},indent=2)+'\n')
    print('TRAIN_ALL_COMPLETE',len(rows),flush=True)


def time_infer(model,batch):
    x=torch.randn(batch,D)
    with torch.no_grad():
        for _ in range(15):model(x)
        samples=[]
        for _ in range(100):
            t=time.perf_counter_ns();model(x);samples.append((time.perf_counter_ns()-t)/1000)
    return float(statistics.median(samples))


def audit():
    torch.set_num_threads(1)
    flag=json.loads((ROOT/'results'/'TRAIN_COMPLETE.json').read_text())
    assert flag['trained']==len(WORLDS)*len(SEEDS)*len(METHODS)
    rows=[];rank=[];ver=[]
    for world in WORLDS:
      for seed in SEEDS:
        xid,ids_id=dataset(seed,world,'fresh_id',2048)
        xood,ids_ood=dataset(seed,world,'fresh_ood',2048)
        gt=teacher(seed,world)
        for method in METHODS:
            raw=(ROOT/'checkpoints'/f'{world}_{seed}_{method}.pt').read_bytes()
            obj=torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')
            torch.manual_seed(seed*777+(0 if world==WORLDS[0] else 50000))
            model=SAE(method);model.load_state_dict(obj['state']);model.eval()
            with torch.no_grad():
                E,dec=model.matrices()
                zid,yid=model.encode_reconstruct(xid)
                zood,yood=model.encode_reconstruct(xood)
                f_id,cs_id,l0id=support_f1(zid,ids_id,dec,gt)
                f_ood,cs_ood,l0ood=support_f1(zood,ids_ood,dec,gt)
                sp=spectrum(E)
                raw2=packet(model)
                assert raw2==raw,'packet not byte-exact'
                med1=time_infer(model,1);med128=time_infer(model,128)
                rec={'world':world,'seed':seed,'method':method,'fresh_id_F1':f_id,'fresh_ood_F1':f_ood,
                     'fresh_id_NMSE':nmse(yid,xid),'fresh_ood_NMSE':nmse(yood,xood),
                     'id_cos':cs_id,'ood_cos':cs_ood,'id_L0':l0id,'ood_L0':l0ood,
                     'learned_params':sum(p.numel() for p in model.parameters()),
                     'packet_bytes':len(raw),'runtime_b1_us':med1,'runtime_b128_us':med128,
                     **sp,'checkpoint_sha256':hashlib.sha256(raw).hexdigest()}
                rows.append(rec)
                rank.append({'world':world,'seed':seed,'method':method,**sp})
                ver.append({'world':world,'seed':seed,'method':method,'replayed':True})
                print('AUDIT',world,seed,method,'F1',round(f_ood,4),'r',sp['rank_rel_1e3'],'er',round(sp['effective_rank'],2),flush=True)
    save_csv(ROOT/'RESULTS_CORE.csv',rows)
    save_csv(ROOT/'results'/'RANK.csv',rank)
    by={(r['world'],r['seed'],r['method']):r for r in rows};pairs=[]
    for world in WORLDS:
        for seed in SEEDS:
            mi=by[world,seed,'mirror_loop'];af=by[world,seed,'affine_loop'];ph=by[world,seed,'tanh_plain_loop'];de=by[world,seed,'dense_onepass']
            gain_a=mi['fresh_ood_F1']-af['fresh_ood_F1'];gain_p=mi['fresh_ood_F1']-ph['fresh_ood_F1'];
            id_ratio=max(mi['fresh_id_NMSE']/af['fresh_id_NMSE'],mi['fresh_id_NMSE']/ph['fresh_id_NMSE'])
            passed=gain_a>=.03 and gain_p>=.03 and id_ratio<=1.05
            pairs.append({'world':world,'seed':seed,'F1_gain_vs_affine':gain_a,'F1_gain_vs_plain_tanh':gain_p,
                          'fresh_ID_NMSE_worst_ratio':id_ratio,'pair_pass':passed,
                          'mirror_vs_dense_F1':mi['fresh_ood_F1']-de['fresh_ood_F1']})
    save_csv(ROOT/'results'/'PAIRED_GATES.csv',pairs)
    mechanism=all(by[w,s,'affine_loop']['rank_rel_1e3']<=8 and by[w,s,'mirror_loop']['rank_rel_1e3']>=32 for w in WORLDS for s in SEEDS)
    quality=sum(x['pair_pass'] for x in pairs)>=5
    checks={'protocol':'frozen on branch before training','fresh_opened_after_training':True,
            'checkpoint_replays':len(ver),'trainings':len(rows),'H1_mechanism_pass':mechanism,'H2_quality_pass':quality,'H2_pair_passes':sum(x['pair_pass'] for x in pairs),
            'H3_plain_tanh_rank_growth':all(by[w,s,'tanh_plain_loop']['rank_rel_1e3']>8 for w in WORLDS for s in SEEDS),
            'seeds':SEEDS,'worlds':WORLDS,'environment':{'python':platform.python_version(),'torch':torch.__version__,
            'cpu':open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].split(':')[-1].strip(),'device':'CPU','threads':1,'clock':'unlocked'}}
    (ROOT/'VERIFICATION.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('FINAL_VERDICT',json.dumps({k:checks[k] for k in ['H1_mechanism_pass','H2_quality_pass','H2_pair_passes','H3_plain_tanh_rank_growth']}),flush=True)


def mechanism():
    torch.set_num_threads(1)
    # Same trained Mirror decoder/orbit; counterfactual transformations at frozen state.
    out=[]
    for world in WORLDS:
      for seed in SEEDS:
        raw=(ROOT/'checkpoints'/f'{world}_{seed}_mirror_loop.pt').read_bytes()
        obj=torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')
        model=SAE('mirror_loop');model.load_state_dict(obj['state']);model.eval()
        with torch.no_grad():
            for transform in ['affine_loop','linear_rot_invert','tanh_plain_loop','mirror_loop','poly_plain_loop']:
                E=model.encoder_matrix(nonlinear_override=transform)
                a=spectrum(E)
                out.append({'world':world,'seed':seed,'counterfactual':transform,**a})
    save_csv(ROOT/'results'/'COUNTERFACTUAL_RANK.csv',out)
    # Mathematical diagnostic: a rank-8 linear matrix pushed through direct tanh.
    g=torch.Generator().manual_seed(4455)
    A=torch.randn(64,8,generator=g,dtype=torch.float64)
    B=torch.randn(8,32,generator=g,dtype=torch.float64)
    M=A@B/3.0
    diag={'rank8_linear':spectrum(M),'rank8_direct_tanh':spectrum(torch.tanh(2.5*M)/2.5),
          'linear_rotation_cancel_error':max(abs(x['matrix_fro']-y['matrix_fro']) for x,y in [] ) if False else None}
    with torch.no_grad():
        E=model.encoder_matrix(nonlinear_override='affine_loop')
        Er=model.encoder_matrix(nonlinear_override='linear_rot_invert')
        diag['last_model_rotation_cancel_max_abs']=float((E-Er).abs().max())
    (ROOT/'results'/'COUNTEREXAMPLE.json').write_text(json.dumps(diag,indent=2)+'\n')
    print('COUNTERFACTUAL_DONE',json.dumps({k:diag[k] for k in ['rank8_linear','rank8_direct_tanh']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['train','audit','mechanism'],required=True)
    args=p.parse_args();{'train':train,'audit':audit,'mechanism':mechanism}[args.phase]()
if __name__=='__main__':main()
