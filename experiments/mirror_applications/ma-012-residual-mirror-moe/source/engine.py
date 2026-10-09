"""MA-012 frozen development/fresh runner."""
import argparse,csv,io,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import D_IN,HIDDEN,D_OUT,ROLES,METHODS,ExpertBank,rotation

UPDATES,BATCH=1200,64

def make_teacher(mode,seed):
    g=torch.Generator().manual_seed(seed)
    if mode=='aligned':
        wi=torch.randn(HIDDEN,D_IN,generator=g)*.12
        wo=torch.randn(D_OUT,HIDDEN,generator=g)*.12
        ai=torch.rand(ROLES,generator=g)*1.2-.6;ao=torch.rand(ROLES,generator=g)*1.2-.6
        left=torch.randn(ROLES,D_OUT,1,generator=g)*.035
        right=torch.randn(ROLES,1,HIDDEN,generator=g)*.035
        return {'mode':mode,'wi':wi,'wo':wo,'ai':ai,'ao':ao,'left':left,'right':right}
    if mode=='independent':
        return {'mode':mode,'wi':torch.randn(ROLES,HIDDEN,D_IN,generator=g)*.12,'wo':torch.randn(ROLES,D_OUT,HIDDEN,generator=g)*.12}
    raise ValueError(mode)

def target(t,x,r):
    if t['mode']=='aligned':
        rin=rotation(t['ai'],D_IN,transpose=True)[r]
        h=F.gelu(torch.bmm(t['wi'].expand(len(x),-1,-1),torch.bmm(rin,x.unsqueeze(-1))).squeeze(-1))
        y=torch.bmm(t['wo'].expand(len(x),-1,-1),h.unsqueeze(-1)).squeeze(-1)
        y=y+torch.bmm(t['left'][r],torch.bmm(t['right'][r],h.unsqueeze(-1))).squeeze(-1)
        return torch.bmm(rotation(t['ao'],D_OUT)[r],y.unsqueeze(-1)).squeeze(-1)
    h=F.gelu(torch.bmm(t['wi'][r],x.unsqueeze(-1)).squeeze(-1))
    return torch.bmm(t['wo'][r],h.unsqueeze(-1)).squeeze(-1)

def data(t,seed,n):
    g=torch.Generator().manual_seed(seed);x=torch.randn(n,D_IN,generator=g);r=torch.arange(n)%ROLES
    perm=torch.randperm(n,generator=g);x=x[perm];r=r[perm]
    return x,r,target(t,x,r)

def payload(m,method):
    b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},b)
    cfg=json.dumps({'method':method,'input':D_IN,'hidden':HIDDEN,'output':D_OUT,'roles':ROLES},sort_keys=True,separators=(',',':')).encode()
    return len(b.getvalue())+len(cfg)

def mac(method):
    base=2*D_IN*HIDDEN
    rank=int(method[-1]) if 'rank' in method else 0
    return base+rank*(D_OUT+HIDDEN)

def run_one(mode,world,tseed,iseed,method,lr):
    t=make_teacher(mode,tseed);x,r,y=data(t,tseed+1,UPDATES*BATCH);xe,re,ye=data(t,tseed+2,4096)
    m=ExpertBank(method,iseed);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    g=torch.Generator().manual_seed(world+120000);ixs=torch.randint(len(x),(UPDATES,BATCH),generator=g)
    st=time.perf_counter()
    for i in range(UPDATES):
        ix=ixs[i];opt.zero_grad(set_to_none=True);loss=F.mse_loss(m(x[ix],r[ix]),y[ix]);loss.backward();opt.step()
    wall=time.perf_counter()-st
    with torch.no_grad():
        err=(m(xe,re)-ye).square().mean(1);by={str(k):float(err[re==k].mean()) for k in range(ROLES)}
        reps=20;st=time.perf_counter()
        for _ in range(reps):m(xe[:1024],re[:1024])
        speed=1024/((time.perf_counter()-st)/reps)
    return {'world':world,'mode':mode,'teacher_seed':tseed,'initialization_seed':iseed,'method':method,'lr':lr,'mse':float(err.mean()),'max_role_mse':max(by.values()),'role_mse':by,'bytes':payload(m,method),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy':mac(method),'train_wall_seconds':wall,'inference_examples_per_second':speed}

def save(path,rows):
    keys=['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'];p=Path(path);new=not p.exists()
    with p.open('a' if not new else 'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n')
        if new:w.writeheader()
        for r in rows:
            note={k:r[k] for k in ('teacher_seed','initialization_seed','lr','role_mse','inference_examples_per_second')}
            w.writerow({'condition':r['mode'],'world_or_seed':r['world'],'method':r['method'],'serialized_bytes':r['bytes'],'train_tokens_or_examples':r['examples'],'optimizer_updates':r['updates'],'active_compute_proxy':r['active_mac_proxy'],'wall_time_s':r['train_wall_seconds'],'primary_metric':'heldout_routed_mse','primary_value':r['mse'],'secondary_metric':'max_role_mse','secondary_value':r['max_role_mse'],'status_note':json.dumps(note,sort_keys=True)})

def development(out):
    lrs=(.001,.003,.01);rows=[]
    for lr in lrs:
      for mode in ('aligned','independent'):
       for i,m in enumerate(METHODS):rows.append(run_one(mode,120000,1200000+(mode=='independent')*100,12000000+i*1000,m,lr))
    by={lr:[r['mse'] for r in rows if r['lr']==lr] for lr in lrs};sel=min(lrs,key=lambda lr:sum(by[lr])/len(by[lr]))
    aligned=[r for r in rows if r['mode']=='aligned' and r['lr']==sel];rsel=min(('mirror_rank1','mirror_rank2'),key=lambda m:next(r['mse'] for r in aligned if r['method']==m))
    Path(out).mkdir(parents=True,exist_ok=True);save(Path(out)/'RESULTS_CORE.csv',rows);Path(out,'dev_raw.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
    Path(out,'DEV_SELECTION.json').write_text(json.dumps({'selected_learning_rate':sel,'selected_mirror_rank':rsel,'mean_heldout_mse_by_lr':{str(k):sum(v)/len(v) for k,v in by.items()},'development_world':120000,'fresh_accessed':False},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'selected_lr':sel,'selected_mirror_rank':rsel,'mean_mse':{str(k):sum(v)/len(v) for k,v in by.items()}},sort_keys=True))

def fresh(out,lr,selected):
    rows=[]
    for w,t,i in zip((120001,120002,120003),(1200001,1200002,1200003),(12000001,12000002,12000003)):
      for mode in ('aligned','independent'):
       for j,m in enumerate(METHODS):rows.append(run_one(mode,w,t+(mode=='independent')*100,i+j*1000,m,lr))
    save(Path(out)/'RESULTS_CORE.csv',rows);Path(out,'fresh_raw.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'fresh_rows':len(rows),'selected_mirror_rank':selected,'aligned_quality_ratio':{str(w):next(r['mse'] for r in rows if r['world']==w and r['mode']=='aligned' and r['method']==selected)/next(r['mse'] for r in rows if r['world']==w and r['mode']=='aligned' and r['method']=='independent') for w in (120001,120002,120003)}}))

def main():
    torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('phase',choices=['development','fresh']);p.add_argument('--out',required=True);p.add_argument('--lr',type=float);p.add_argument('--rank')
    a=p.parse_args()
    if a.phase=='development':development(a.out)
    else:
        if a.lr is None or a.rank is None:raise SystemExit('fresh requires frozen --lr and --rank')
        fresh(a.out,a.lr,a.rank)
if __name__=='__main__':main()
