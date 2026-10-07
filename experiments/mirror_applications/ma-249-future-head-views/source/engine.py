#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import HeadStudent,METHODS,compute_proxy,givens
ROOT=Path(__file__).resolve().parents[1];OFF=4;D=16;V=12;UPDATES=1200;FIELDS=['condition','world_or_seed','mode','method','serialized_model_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note']

def world(seed,mode):
    g=torch.Generator().manual_seed(seed);q,_=torch.linalg.qr(torch.randn(D,D,generator=g));w=torch.randn(V,D,generator=g)*.6;b=torch.randn(V,generator=g)*.15;theta=torch.rand(OFF,D//2,generator=g)*1.5-.75
    if mode=='independent_offset_heads':
        wi=torch.randn(OFF,V,D,generator=g)*.6;bi=torch.randn(OFF,V,generator=g)*.15
    else:wi=bi=None
    return {'q':q,'w':w,'b':b,'theta':theta,'wi':wi,'bi':bi}
def targets(x,w,mode):
    h=x@w['q']
    if mode=='aligned_shared_base':logits=torch.einsum('bhd,vd->bhv',givens(h[:,None,:].expand(-1,OFF,-1),w['theta']),w['w'])+w['b']
    else:logits=torch.einsum('bd,hvd->bhv',h,w['wi'])+w['bi']
    return F.softmax(logits,-1)
def metrics(m,x,t):
    m.eval()
    with torch.no_grad():
        q=t.clamp_min(1e-12);lp=F.log_softmax(m(x),-1);kl=(q*(q.log()-lp)).sum(-1).mean().item();acc=(lp.argmax(-1)==q.argmax(-1)).float().mean().item()
    return kl,acc
def fit(method,w,mode,seed,data_seed,lr,updates=UPDATES):
    torch.manual_seed(seed);m=HeadStudent(method,w['q'],OFF,V,D);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(data_seed);start=time.perf_counter();m.train()
    for _ in range(updates):
        ix=torch.randn(64,D,generator=g);t=targets(ix,w,mode);logp=F.log_softmax(m(ix),-1);loss=-(t*logp).sum(-1).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return m,time.perf_counter()-start
def throughput(m,x):
    xx=x[:64];m.eval()
    with torch.no_grad():
        for _ in range(5):m(xx)
        t=time.perf_counter()
        for _ in range(30):m(xx)
    return 64*30/(time.perf_counter()-t)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);ap.add_argument('--lr',type=float);a=ap.parse_args();torch.set_num_threads(1)
    modes=['aligned_shared_base','independent_offset_heads']
    if a.phase=='dev':worlds=[(24900,249000)];lrs=[.003,.01]
    else:
        if a.lr not in (.003,.01):raise SystemExit('fresh requires frozen lr')
        worlds=[(24901,249001),(24902,249002),(24903,249003)];lrs=[a.lr]
    rows=[];scores={lr:[] for lr in lrs}
    for wid,init in worlds:
        for mode in modes:
            w=world(wid,mode);vg=torch.Generator().manual_seed(init+101);xv=torch.randn(4096,D,generator=vg);tv=targets(xv,w,mode);data_seed=init+211+(0 if mode==modes[0] else 10000)
            for lr in lrs:
                for i,method in enumerate(METHODS):
                    m,elapsed=fit(method,w,mode,init+i*271+int(lr*10000),data_seed,lr);kl,acc=metrics(m,xv,tv)
                    if a.phase=='dev':scores[lr].append(kl)
                    rows.append({'condition':a.phase,'world_or_seed':wid,'mode':mode,'method':method,'serialized_model_bytes':m.serialized_payload_bytes(),'train_examples':UPDATES*64,'optimizer_updates':UPDATES,'active_compute_proxy':compute_proxy(method,UPDATES*64),'wall_time_s':round(elapsed,6),'inference_examples_per_s':round(throughput(m,xv),3),'primary_metric':'teacher_to_student_KL','primary_value':f'{kl:.10g}','secondary_metric':'offset_top1_agreement','secondary_value':f'{acc:.8g}','status_note':f'lr={lr}; matched data stream; fixed feature trunk charged'})
                    print(a.phase,wid,mode,method,lr,'KL',kl,'top1',acc,'bytes',rows[-1]['serialized_model_bytes'],flush=True)
    if a.phase=='dev':
        best=min(scores,key=lambda lr:sum(scores[lr])/len(scores[lr]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-249','selected_common_lr':best,'mean_kl_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[24901,24902,24903],'updates':UPDATES},indent=2)+'\n')
    with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
    print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':len(rows)}))
if __name__=='__main__':main()
