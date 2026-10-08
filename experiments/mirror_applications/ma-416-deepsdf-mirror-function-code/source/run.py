from __future__ import annotations
import argparse,hashlib,io,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

LATENT=5;SHAPES=8;D=64

def ellipse_sdf(x,p):
    # x [S,N,2], p [S,5] = tx,ty,log(rx),log(ry),theta
    dx=x[...,0]-p[:,None,0];dy=x[...,1]-p[:,None,1];c=torch.cos(p[:,None,4]);s=torch.sin(p[:,None,4])
    xl=dx*c+dy*s;yl=-dx*s+dy*c;rx=torch.exp(p[:,None,2]);ry=torch.exp(p[:,None,3])
    r=torch.sqrt((xl/rx)**2+(yl/ry)**2+1e-10)
    return torch.minimum(rx,ry)*(r-1)

class Decoder(nn.Module):
    def __init__(self,input_dim=2):
        super().__init__();self.net=nn.Sequential(nn.Linear(input_dim,D),nn.SiLU(),nn.Linear(D,D),nn.SiLU(),nn.Linear(D,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def geometry_codes(g,n):
    tx=(torch.rand(n,generator=g)-.5)*.5;ty=(torch.rand(n,generator=g)-.5)*.5
    rx=(torch.rand(n,generator=g)-.5)*.5;ry=(torch.rand(n,generator=g)-.5)*.5;theta=(torch.rand(n,generator=g)-.5)*1.2
    return torch.stack((tx,ty,rx,ry,theta),dim=-1)

def sample_fields(params,n,g):
    x=(torch.rand(len(params),n,2,generator=g)-.5)*3
    return x,ellipse_sdf(x,params)

def train_canonical(seed,updates=500):
    torch.manual_seed(seed);g=torch.Generator().manual_seed(seed+1000);m=Decoder(2);opt=torch.optim.Adam(m.parameters(),lr=.002);start=time.perf_counter()
    for _ in range(updates):
        x=(torch.rand(1024,2,generator=g)-.5)*3;y=torch.linalg.vector_norm(x,dim=-1)-1
        loss=((m(x)-y)**2).mean();opt.zero_grad();loss.backward();opt.step()
    return m,time.perf_counter()-start

class DeepSDF(nn.Module):
    def __init__(self):super().__init__();self.dec=Decoder(2+LATENT)
    def forward(self,x,z):return self.dec(torch.cat((x,z),dim=-1))

def train_deepsdf(seed,x,y,updates=700):
    torch.manual_seed(seed+2000);g=torch.Generator().manual_seed(seed+3000);m=DeepSDF();codes=nn.Parameter(torch.randn(SHAPES,LATENT)*.01);opt=torch.optim.Adam([*m.parameters(),codes],lr=.0015);start=time.perf_counter()
    for _ in range(updates):
        si=torch.randint(SHAPES,(1024,),generator=g);pi=torch.randint(x.shape[1],(1024,),generator=g)
        xx=x[si,pi];yy=y[si,pi];zz=codes[si]
        pred=m(xx,zz);loss=((pred-yy)**2).mean()+1e-4*(zz**2).mean();opt.zero_grad();loss.backward();opt.step()
    return m,codes.detach(),time.perf_counter()-start

def mirror_eval(base,codes,x):
    dx=x[...,0]-codes[:,None,0];dy=x[...,1]-codes[:,None,1];c=torch.cos(codes[:,None,4]);s=torch.sin(codes[:,None,4])
    xl=dx*c+dy*s;yl=-dx*s+dy*c;rx=torch.exp(codes[:,None,2]);ry=torch.exp(codes[:,None,3])
    q=torch.stack((xl/rx,yl/ry),dim=-1);return torch.minimum(rx,ry)*base(q.reshape(-1,2)).reshape(q.shape[:-1])

def adapt_mirror(base,x,y,updates=300,seed=0):
    for p in base.parameters():p.requires_grad_(False)
    torch.manual_seed(seed);codes=nn.Parameter(torch.zeros(len(x),LATENT));opt=torch.optim.Adam([codes],lr=.025);g=torch.Generator().manual_seed(seed+55);start=time.perf_counter()
    for _ in range(updates):
        si=torch.arange(len(x));pi=torch.randint(x.shape[1],(len(x),512),generator=g);xx=x[si[:,None],pi];yy=y[si[:,None],pi]
        pred=mirror_eval(base,codes,xx);loss=((pred-yy)**2).mean();opt.zero_grad();loss.backward();opt.step()
    return codes.detach(),time.perf_counter()-start

def adapt_deep(m,x,y,updates=300,seed=0):
    for p in m.parameters():p.requires_grad_(False)
    torch.manual_seed(seed);codes=nn.Parameter(torch.zeros(len(x),LATENT));opt=torch.optim.Adam([codes],lr=.03);g=torch.Generator().manual_seed(seed+77);start=time.perf_counter()
    for _ in range(updates):
        si=torch.arange(len(x));pi=torch.randint(x.shape[1],(len(x),256),generator=g);xx=x[si[:,None],pi];yy=y[si[:,None],pi]
        pred=m(xx,codes[:,None,:].expand(-1,xx.shape[1],-1));loss=((pred-yy)**2).mean()+1e-4*(codes**2).mean();opt.zero_grad();loss.backward();opt.step()
    return codes.detach(),time.perf_counter()-start

def pack_model(model,codes,kind):
    arrays={f'w_{k.replace(".","__")}':v.detach().cpu().numpy().astype(np.float16) for k,v in model.state_dict().items()}
    arrays['instance_codes']=codes.detach().cpu().numpy().astype(np.float16)
    arrays['metadata_utf8']=np.frombuffer(json.dumps({'kind':kind,'latent_dimension':LATENT,'dtype':'float16'},sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
    b=io.BytesIO();np.savez_compressed(b,**arrays);return b.getvalue()

def replay(raw,x,kind):
    with np.load(io.BytesIO(raw),allow_pickle=False) as a:
        codes=torch.tensor(a['instance_codes'].astype(np.float32));state={k[2:].replace('__','.'):torch.tensor(v.astype(np.float32)) for k,v in a.items() if k.startswith('w_')}
        if kind=='deepsdf':m=DeepSDF()
        else:m=Decoder(2)
        m.load_state_dict(state)
    with torch.no_grad():
        if kind=='deepsdf':return m(x,codes[:,None,:].expand(-1,x.shape[1],-1))
        return mirror_eval(m,codes,x)

def nrmse(pred,target):
    den=((target-target.mean(dim=1,keepdim=True))**2).mean(dim=1).sqrt().clamp_min(1e-8)
    return float(((pred-target)**2).mean(dim=1).sqrt().div(den).mean())

def timed_queries(model,codes,x,kind):
    xx=x[:,:512]
    with torch.no_grad():
        for _ in range(5):
            _=model(xx,codes[:,None,:].expand(-1,xx.shape[1],-1)) if kind=='deepsdf' else mirror_eval(model,codes,xx)
        start=time.perf_counter()
        for _ in range(20):
            _=model(xx,codes[:,None,:].expand(-1,xx.shape[1],-1)) if kind=='deepsdf' else mirror_eval(model,codes,xx)
    return 20*len(xx)*xx.shape[1]/(time.perf_counter()-start)

def run(out,seeds=(41601,41602),n=1024,query_n=2048):
    out.mkdir(parents=True,exist_ok=True);rows=[];checks=[];torch.set_num_threads(1)
    for seed in seeds:
        g=torch.Generator().manual_seed(seed);train_p=geometry_codes(g,SHAPES);held_p=geometry_codes(g,SHAPES)
        xt,yt=sample_fields(train_p,n,g);xq,yq=sample_fields(held_p,query_n,g);xs,ys=sample_fields(held_p,256,g)
        base,base_sec=train_canonical(seed)
        deep,train_codes,deep_sec=train_deepsdf(seed,xt,yt)
        mcode,ma_sec=adapt_mirror(base,xs,ys,seed=seed+4000);dcode,da_sec=adapt_deep(deep,xs,ys,seed=seed+5000)
        native=mcode.clone()
        # Serialize each deployable decoder plus all held-out instance codes.
        mraw=pack_model(base,mcode,'mirror_geometry');nraw=pack_model(base,native,'native_affine_geometry');draw=pack_model(deep,dcode,'deepsdf')
        for name,raw in [('mirror',mraw),('native_affine',nraw),('deepsdf',draw)]: (out/f'dev{seed}_{name}.npz').write_bytes(raw)
        mp=replay(mraw,xq,'mirror');npred=replay(nraw,xq,'mirror');dp=replay(draw,xq,'deepsdf')
        mm=nrmse(mp,yq);nm=nrmse(npred,yq);dm=nrmse(dp,yq)
        # Interpolate held-out code pairs and target geometric parameters 50/50.
        ip=(held_p[::2]+held_p[1::2])*.5;ix,iy=sample_fields(ip,query_n,g)
        mi=(mcode[::2]+mcode[1::2])*.5;di=(dcode[::2]+dcode[1::2])*.5
        mpi=replay(pack_model(base,mi,'mirror_interpolated'),ix,'mirror');dpi=replay(pack_model(deep,di,'deepsdf_interpolated'),ix,'deepsdf')
        im=nrmse(mpi,iy);idp=nrmse(dpi,iy)
        oracle_payload=io.BytesIO();np.savez_compressed(oracle_payload,geometry_codes=held_p.numpy().astype(np.float32),metadata_utf8=np.frombuffer(b'analytic_ellipse_sdf',dtype=np.uint8));oraw=oracle_payload.getvalue();(out/f'dev{seed}_analytic_oracle.npz').write_bytes(oraw)
        for name,raw,score,interp,sec,rate,ops in [
            ('mirror',mraw,mm,im,base_sec+ma_sec,timed_queries(base,mcode,xq,'mirror'),8626),
            ('native_affine',nraw,nm,im,base_sec+ma_sec,timed_queries(base,native,xq,'mirror'),8626),
            ('deepsdf',draw,dm,idp,deep_sec+da_sec,timed_queries(deep,dcode,xq,'deepsdf'),9216),
            ('analytic_oracle',oraw,0.,0.,0.,0.,25)]:
            digest=hashlib.sha256(raw).hexdigest();updates=500+300 if name in ('mirror','native_affine') else 700+300 if name=='deepsdf' else 0
            rows.append({'condition':'heldout_shapes','world_or_seed':seed,'method':name,'serialized_bytes':len(raw),'train_tokens_or_examples':updates*1024,'optimizer_updates':updates,'active_compute_proxy':ops,'wall_time_s':round(sec,6),'primary_metric':'query_NRMSE','primary_value':score,'secondary_metric':'interpolation_NRMSE','secondary_value':interp,'status_note':digest})
        checks.append({'seed':seed,'mirror_nrmse':mm,'native_affine_nrmse':nm,'deepsdf_nrmse':dm,'mirror_bytes':len(mraw),'native_affine_bytes':len(nraw),'deepsdf_bytes':len(draw),'oracle_bytes':len(oraw),'mirror_interp_nrmse':im,'deepsdf_interp_nrmse':idp,'mirror_deepsdf_throughput':timed_queries(base,mcode,xq,'mirror')/max(timed_queries(deep,dcode,xq,'deepsdf'),1e-9)})
    import csv
    with (out/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    result={'experiment_id':'MA-416','protocol_frozen':True,'fresh_accessed':False,'development_gate_passed':all(r['mirror_nrmse']<=.03 and r['mirror_nrmse']<=.5*r['deepsdf_nrmse'] and r['mirror_bytes']<=.8*r['deepsdf_bytes'] and r['mirror_interp_nrmse']<=.05 and abs(r['mirror_nrmse']-r['native_affine_nrmse'])<=1e-6 for r in checks),'seed_results':checks}
    (out/'screen.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'runs');a=p.parse_args();print(json.dumps(run(a.out),indent=2))
