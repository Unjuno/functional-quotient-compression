"""Frozen MA-1166 Phase E: parallel virtual-MLP no-prune CPU experiment."""
import argparse, csv, hashlib, io, json, math, platform, statistics, struct, sys, time
from pathlib import Path
import numpy as np
import torch
from phase_c_model import Teacher, NativeMLP, VirtualSupernet
from parallel_model import ParallelGraph

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'; CHECK=ROOT/'checkpoints'
SEEDS=(461,487)
WORLDS=('aligned','independent','parallel_teacher')
MODES=('mirror_serial','mirror_serial_batched','mirror_sync2','mirror_sync6',
       'affine_serial','affine_serial_batched','affine_sync2','affine_sync6','native253')
UPDATES=400; BATCH=128; LR=.015
SPLITS={'train':(2048,.95,110000),'dev':(1024,1.0,220000),'fresh':(1024,1.0,330000),'ood':(1024,1.5,440000)}

def sha_bytes(b): return hashlib.sha256(b).hexdigest()

def make_model(mode,seed):
    torch.manual_seed(seed*101+19213)
    if mode=='native253':
        return NativeMLP()
    decoder='mirror' if mode.startswith('mirror') else 'affine'
    if mode.endswith('_serial'):
        return VirtualSupernet(decoder,1977)
    if mode.endswith('_serial_batched'):
        return ParallelGraph(decoder,'serial_batched',1977,steps=1)
    if mode.endswith('_sync2') or mode.endswith('_sync6'):
        steps=int(mode[-1]);return ParallelGraph(decoder,'sync',1977,steps=steps)
    raise ValueError(mode)

def make_teacher(world,seed):
    if world!='parallel_teacher':
        return Teacher(world,seed)
    teacher=ParallelGraph('mirror','sync',1977,steps=2)
    g=torch.Generator().manual_seed(seed+33011)
    with torch.no_grad():
        teacher.theta[:64].copy_(torch.randn(64,generator=g)*.9)
        teacher.theta[64:127].copy_(torch.randn(63,generator=g)*.35-.15)
        teacher.theta[127:].copy_(torch.randn(126,generator=g)*.45)
    return teacher

def datasets(world,seed):
    teacher=make_teacher(world,seed)
    ret={}
    for name,(n,scale,offset) in SPLITS.items():
        x=torch.randn(n,6,generator=torch.Generator().manual_seed(seed*1000+offset))*scale
        with torch.no_grad():
            y=torch.cat([teacher(q) for q in x.split(256)],dim=0)
        ret[name]=(x,y)
    return ret

def nmse(yhat,x,y):
    return float(((yhat-y)**2).mean()/((y-x)**2).mean().clamp_min(1e-12))

@torch.no_grad()
def evaluate(mod,data,batch=256):
    mod.eval()
    x,y=data
    pred=torch.cat([mod(q) for q in x.split(batch)],dim=0)
    return nmse(pred,x,y),pred

def payload(mod,mode):
    # One deterministic physical learned tensor for every Mirror/affine graph.
    # The decoder's program is separately accounted; seeded tensors are generated
    # by fixed code and seed, not extra learned coefficients (but not free in runtime).
    if isinstance(mod,VirtualSupernet):
        arr=mod.theta.detach().cpu().numpy().astype('<f4',copy=False)
        meta=dict(schema=1,mode=mode,decoder=mod.decoder,seed=mod.seed,
                  virtual_shape=[6,8,6],edges=63,pruning=False,
                  parameters=[['theta',[253]]],program='phase_c_model.py + parallel_model.py')
    else:
        pars=[(k,v.detach().cpu().numpy().astype('<f4')) for k,v in mod.named_parameters()]
        arr=np.concatenate([v.ravel() for _,v in pars]).astype('<f4',copy=False)
        meta=dict(schema=1,mode=mode,pruning=False,parameters=[[k,list(v.shape)] for k,v in pars],program='phase_c_model.py#NativeMLP')
    head=json.dumps(meta,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    return struct.pack('<I',len(head))+head+arr.tobytes(),meta

def group_grad(mod,batch):
    if not isinstance(mod,VirtualSupernet):return None
    x,y=batch;mod.zero_grad(set_to_none=True)
    l=((mod(x)-y)**2).mean();l.backward()
    grad=mod.theta.grad
    return dict(core_l2=float(grad[:64].norm()),gate_l2=float(grad[64:127].norm()),code_l2=float(grad[127:].norm()),
                core_nonzero=int((grad[:64].abs()>1e-12).sum()),
                gate_nonzero=int((grad[64:127].abs()>1e-12).sum()),
                code_nonzero=int((grad[127:].abs()>1e-12).sum()),
                finite=bool(torch.isfinite(grad).all()))

def benchmark(mod,bs,reps=40):
    x=torch.randn(bs,6)
    mod.eval()
    with torch.no_grad():
        for _ in range(8):mod(x)
        out=[]
        for _ in range(reps):
            t0=time.perf_counter_ns(); mod(x); out.append((time.perf_counter_ns()-t0)/1000)
    return {'median_us':statistics.median(out),'p10_us':sorted(out)[int(.10*(reps-1))],
            'p90_us':sorted(out)[int(.90*(reps-1))],'repetitions':reps}

def train_one(world,seed,mode,data,steps=UPDATES,save=True,bench=True):
    mod=make_model(mode,seed)
    count=sum(p.numel() for p in mod.parameters())
    assert count==253, (mode,count)
    opt=torch.optim.Adam(mod.parameters(),lr=LR)
    gen=torch.Generator().manual_seed(seed*21001+129)
    x,y=data['train']
    norm=((y-x)**2).mean().item()
    curve=[]
    t0=time.perf_counter()
    for step in range(1,steps+1):
        mod.train()
        ids=torch.randint(len(x),(BATCH,),generator=gen)
        opt.zero_grad(set_to_none=True)
        loss=((mod(x[ids])-y[ids])**2).mean()/max(norm,1e-12)
        if not torch.isfinite(loss): raise ArithmeticError((world,seed,mode,step,loss))
        loss.backward()
        opt.step()
        if step==1 or step%25==0 or step==steps:
            curve.append(dict(update=step,minibatch_nmse=float(loss.detach())))
    secs=time.perf_counter()-t0
    d,_=evaluate(mod,data['dev']);f,fp=evaluate(mod,data['fresh']);o,op=evaluate(mod,data['ood']);tr,_=evaluate(mod,data['train'])
    pp,meta=payload(mod,mode)
    grad=group_grad(mod,(data['train'][0][:128],data['train'][1][:128]))
    if isinstance(mod,ParallelGraph):
        edge_stats=mod(data['train'][0][:8],return_stats=True)[1]
    elif isinstance(mod,VirtualSupernet):
        out,st=mod(data['train'][0][:8],return_stats=True)
        edge_stats=dict(candidate_edges=63,active_edge_evals=st['edge_evals'],wave_count=6,physical_learned_scalars=253)
    else:edge_stats=dict(candidate_edges=0,active_edge_evals=0,wave_count=2,physical_learned_scalars=253)
    name=f'{world}_{seed}_{mode}'
    res=dict(world=world,seed=seed,mode=mode,trained_updates=steps,
             train_nmse=tr,dev_nmse=d,fresh_nmse=f,ood_nmse=o,train_sec=secs,
             learned_scalars=count,physical_trainable_parameter_objects=len(list(mod.parameters())),
             virtual_edge_mlp_weights=6930 if mode!='native253' else None,
             active_edge_evals=edge_stats['active_edge_evals'],sequential_waves=edge_stats['wave_count'],
             payload_bytes=len(pp),payload_sha256=sha_bytes(pp),
             fresh_prediction_sha256=sha_bytes(fp.numpy().astype('<f4').tobytes()),
             ood_prediction_sha256=sha_bytes(op.numpy().astype('<f4').tobytes()),
             grad_group=grad,curve=curve,checkpoint='checkpoints/'+name+'.pt',
             decoder_source_sha256=sha_bytes((ROOT/'source'/'phase_c_model.py').read_bytes()),
             parallel_source_sha256=sha_bytes((ROOT/'source'/'parallel_model.py').read_bytes()))
    if bench:
        res['b1_cpu']=benchmark(mod,1,40)
        res['b128_cpu']=benchmark(mod,128,22)
    if save:
        RESULTS.mkdir(parents=True,exist_ok=True);CHECK.mkdir(parents=True,exist_ok=True)
        (RESULTS/(name+'.bin')).write_bytes(pp)
        chk=dict(mode=mode,world=world,seed=seed,steps=steps,state_dict=mod.state_dict(),
                 meta=meta,payload_sha256=res['payload_sha256'],fresh_prediction_sha256=res['fresh_prediction_sha256'])
        file=CHECK/(name+'.pt');torch.save(chk,file)
        res['checkpoint_sha256']=sha_bytes(file.read_bytes())
        (RESULTS/(name+'.json')).write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n')
    return res

def verify_one(res):
    name=f'{res["world"]}_{res["seed"]}_{res["mode"]}'
    chk=torch.load(CHECK/(name+'.pt'),map_location='cpu',weights_only=True)
    assert sha_bytes((CHECK/(name+'.pt')).read_bytes())==res['checkpoint_sha256']
    mod=make_model(res['mode'],res['seed']);mod.load_state_dict(chk['state_dict']); mod.eval()
    data=datasets(res['world'],res['seed'])
    train,_=evaluate(mod,data['train']); dev,_=evaluate(mod,data['dev'])
    fresh,fp=evaluate(mod,data['fresh']);ood,op=evaluate(mod,data['ood'])
    pp,_=payload(mod,res['mode'])
    assert sha_bytes(pp)==res['payload_sha256']
    assert sha_bytes(fp.numpy().astype('<f4').tobytes())==res['fresh_prediction_sha256']
    assert sha_bytes(op.numpy().astype('<f4').tobytes())==res['ood_prediction_sha256']
    for typ,val in [('train_nmse',train),('dev_nmse',dev),('fresh_nmse',fresh),('ood_nmse',ood)]:
        assert abs(val-res[typ])<1e-7,(name,typ,val,res[typ])
    return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--world',choices=WORLDS)
    p.add_argument('--seed',type=int)
    p.add_argument('--mode',choices=MODES)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--replay',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1)
    RESULTS.mkdir(parents=True,exist_ok=True)
    if args.replay:
        rs=json.loads((RESULTS/'RESULTS_CORE.json').read_text())
        n=0
        for r in rs:
            verify_one(r);n+=1
        print('REPLAY_PASS',n,flush=True)
        return
    (ROOT/'ENVIRONMENT.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cpu=platform.processor(),os=platform.platform(),threads=1,cuda=torch.cuda.is_available()),indent=2)+'\n')
    old_file=RESULTS/'RESULTS_CORE.json'
    results=json.loads(old_file.read_text()) if old_file.exists() and not args.smoke else []
    done={(q['world'],q['seed'],q['mode']) for q in results}
    for world in ([args.world] if args.world else WORLDS):
        for seed in ([args.seed] if args.seed else SEEDS):
            data=datasets(world,seed)
            print('DATA',world,seed,'identity_fresh_nmse',nmse(data['fresh'][0],*data['fresh']),flush=True)
            for mode in ([args.mode] if args.mode else MODES):
                if (world,seed,mode) in done:continue
                r=train_one(world,seed,mode,data,steps=25 if args.smoke else UPDATES,save=not args.smoke,bench=not args.smoke)
                if args.smoke:
                    print('SMOKE',world,seed,mode,'sec',round(r['train_sec'],3),'fresh',round(r['fresh_nmse'],5),flush=True)
                    continue
                results.append(r);done.add((world,seed,mode))
                print('RUN',len(results),'world',world,'seed',seed,'mode',mode,
                      'fresh',round(r['fresh_nmse'],5),'ood',round(r['ood_nmse'],5),
                      'train_sec',round(r['train_sec'],2),'b1_us',round(r['b1_cpu']['median_us'],2),flush=True)
                old_file.write_text(json.dumps(results,indent=2)+'\n')
                with (RESULTS/'RESULTS.csv').open('w',newline='') as f:
                    keys=['world','seed','mode','trained_updates','train_nmse','dev_nmse','fresh_nmse','ood_nmse','train_sec','learned_scalars','physical_trainable_parameter_objects','virtual_edge_mlp_weights','active_edge_evals','sequential_waves','payload_bytes','payload_sha256','checkpoint_sha256']
                    cw=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');cw.writeheader();cw.writerows(results)
    print('FINISHED',len(results),'expected',len(WORLDS)*len(SEEDS)*len(MODES),flush=True)

if __name__=='__main__':main()
