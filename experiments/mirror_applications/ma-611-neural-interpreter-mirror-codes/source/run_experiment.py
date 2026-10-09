#!/usr/bin/env python3
"""MA-611 phase-view codes in a shared Fourier interpreter."""
import argparse,hashlib,io,json,platform,time
from pathlib import Path
import torch
FREQS=torch.tensor([1.,2.,3.,4.]);TASKS=64;SUPPORT=32;QUERY=256

def basis(x,s):
    w=FREQS[s];return torch.stack((torch.sin(w*x),torch.cos(w*x)),-1)
def target(x,s,phi):return torch.sin(FREQS[s]*x+phi)
def mirror_predict(x,s,theta):
    z=basis(x,s);return z[...,0]*torch.cos(theta)+z[...,1]*torch.sin(theta)
def coeff_predict(x,s,c):return basis(x,s)@c

def adapt(seed):
    g=torch.Generator().manual_seed(seed);phases=(torch.rand(TASKS,generator=g)*2*torch.pi-torch.pi);sigs=torch.arange(TASKS)%4
    mt=[];ct=[];query=[];truth=[];support_total=0
    grid=torch.arange(720)* (2*torch.pi/720)-torch.pi
    for i in range(TASKS):
        s=int(sigs[i]);phi=phases[i]
        gs=torch.Generator().manual_seed(seed+1000+i);xs=(torch.rand(SUPPORT,generator=gs)*2-1)*torch.pi;ys=target(xs,s,phi)
        qg=torch.Generator().manual_seed(seed+2000+i);xq=(torch.rand(QUERY,generator=qg)*2-1)*torch.pi;yq=target(xq,s,phi)
        losses=((mirror_predict(xs[:,None],s,grid[None,:])-ys[:,None])**2).mean(0);theta=grid[losses.argmin()]
        c=torch.linalg.lstsq(basis(xs,s),ys).solution
        mt.append(theta);ct.append(c);query.append(xq);truth.append(yq);support_total+=SUPPORT
    mt=torch.stack(mt);ct=torch.stack(ct);mq=[];cq=[];target_comp=[];comp_sig=[]
    # Eight within-signature pairs per signature; composed phases are new logical functions.
    for s in range(4):
        ids=[i for i in range(TASKS) if int(sigs[i])==s]
        for j in range(8):
            i1,i2=ids[j],ids[(j+5)%len(ids)];mq.append(torch.remainder(mt[i1]+mt[i2]+torch.pi,2*torch.pi)-torch.pi)
            a,b=ct[i1],ct[i2];cq.append(torch.stack((a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])))
            comp_sig.append(s);qg=torch.Generator().manual_seed(seed+3000+s*8+j);xq=(torch.rand(QUERY,generator=qg)*2-1)*torch.pi
            phi=phases[i1]+phases[i2];target_comp.append((xq,target(xq,s,phi)))
    mq=torch.stack(mq);cq=torch.stack(cq);comp_sig=torch.tensor(comp_sig);ml=[];cl=[]
    for i in range(TASKS):
        s=int(sigs[i]);ml.append((((mirror_predict(query[i],s,mt[i])-truth[i])**2).mean()/truth[i].var()).item());cl.append((((coeff_predict(query[i],s,ct[i])-truth[i])**2).mean()/truth[i].var()).item())
    mlc=[];clc=[]
    for j,(x,y) in enumerate(target_comp):
        s=int(comp_sig[j]);mlc.append((((mirror_predict(x,s,mq[j])-y)**2).mean()/y.var()).item());clc.append((((coeff_predict(x,s,cq[j])-y)**2).mean()/y.var()).item())
    return {'sigs':sigs,'mirror_codes':mt,'coeff_codes':ct,'comp_sigs':comp_sig,'mirror_comp_codes':mq,'coeff_comp_codes':cq,'query_mirror_nrmse2_mean':sum(ml)/len(ml),'query_coeff_nrmse2_mean':sum(cl)/len(cl),'compose_mirror_nrmse2_mean':sum(mlc)/len(mlc),'compose_coeff_nrmse2_mean':sum(clc)/len(clc),'support_examples':support_total,'query_examples':TASKS*QUERY,'compose_query_examples':len(target_comp)*QUERY}

def pack(mode,sigs,codes,comp_sigs,comp_codes,seed):
    bio=io.BytesIO();torch.save({'mode':mode,'signatures':sigs.to(torch.uint8),'codes':codes.float().contiguous(),'composed_signatures':comp_sigs.to(torch.uint8),'composed_codes':comp_codes.float().contiguous(),'metadata':{'seed':seed,'frequencies':[1,2,3,4],'format':'MA611-v1'}},bio);return bio.getvalue()
def replay(blob,expected_sigs,expected_codes,expected_comp_sigs,expected_comp_codes):
    d=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);s=d['signatures'].long();cs=d['composed_signatures'].long();codes=d['codes'];cc=d['composed_codes'];maxerr=0.
    for i in range(len(s)):
        x=torch.linspace(-3,3,64);z=mirror_predict(x,int(s[i]),codes[i]) if d['mode']=='mirror' else coeff_predict(x,int(s[i]),codes[i]);ref=mirror_predict(x,int(expected_sigs[i]),expected_codes[i]) if d['mode']=='mirror' else coeff_predict(x,int(expected_sigs[i]),expected_codes[i]);maxerr=max(maxerr,float((z-ref).abs().max()))
    for i in range(len(cs)):
        x=torch.linspace(-3,3,64);z=mirror_predict(x,int(cs[i]),cc[i]) if d['mode']=='mirror' else coeff_predict(x,int(cs[i]),cc[i]);ref=mirror_predict(x,int(expected_comp_sigs[i]),expected_comp_codes[i]) if d['mode']=='mirror' else coeff_predict(x,int(expected_comp_sigs[i]),expected_comp_codes[i]);maxerr=max(maxerr,float((z-ref).abs().max()))
    return maxerr

def run(seed):
    r=adapt(seed);root=Path(__file__).resolve().parents[1]/'runs/dev_payloads';root.mkdir(parents=True,exist_ok=True);sizes={};hashes={};files={};errs={}
    for mode,codes,comps in [('mirror',r['mirror_codes'],r['mirror_comp_codes']),('coeff',r['coeff_codes'],r['coeff_comp_codes'])]:
        blob=pack(mode,r['sigs'],codes,r['comp_sigs'],comps,seed);p=root/f'{seed}_{mode}.pt';p.write_bytes(blob);sizes[mode]=len(blob);hashes[mode]=hashlib.sha256(blob).hexdigest();files[mode]=str(p.relative_to(Path(__file__).resolve().parents[1]));errs[mode]=replay(blob,r['sigs'],codes,r['comp_sigs'],comps)
    return {'logical_functions':TASKS,'composed_functions':32,'support_examples':r['support_examples'],'query_examples':r['query_examples'],'compose_query_examples':r['compose_query_examples'],'optimizer_updates':0,'query_mirror_nrmse2_mean':r['query_mirror_nrmse2_mean'],'query_coeff_nrmse2_mean':r['query_coeff_nrmse2_mean'],'compose_mirror_nrmse2_mean':r['compose_mirror_nrmse2_mean'],'compose_coeff_nrmse2_mean':r['compose_coeff_nrmse2_mean'],'payload_bytes':sizes,'payload_sha256':hashes,'payload_files':files,'payload_replay_max_abs':errs,'support_compute_proxy':{'mirror_grid_search_candidates':TASKS*720*SUPPORT,'ordinary_code_lstsq_examples':TASKS*SUPPORT},'query_compute_proxy_per_example':{'mirror':6,'coeff':3},'latency_ms_per_16384':latency(r),'world_seed':seed}

def latency(r):
    x=torch.linspace(-3,3,TASKS*QUERY)
    def m():return torch.stack([mirror_predict(x[i*QUERY:(i+1)*QUERY],int(r['sigs'][i]),r['mirror_codes'][i]) for i in range(TASKS)])
    def c():return torch.stack([coeff_predict(x[i*QUERY:(i+1)*QUERY],int(r['sigs'][i]),r['coeff_codes'][i]) for i in range(TASKS)])
    m();c();out={}
    for name,fn in [('mirror',m),('coeff',c)]:
        start=time.perf_counter()
        for _ in range(30):fn()
        out[name]=(time.perf_counter()-start)*1000/30
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[61101,61102]);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(1)
    d={'experiment_id':'MA-611','phase':'deterministic adaptation/algebra screen','python':platform.python_version(),'torch':torch.__version__,'device':'cpu','worlds':{str(s):run(s) for s in a.seeds}}
    o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':main()
