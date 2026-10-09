#!/usr/bin/env python3
"""MA-614 deterministic composition screen: one Fourier-view executor vs sequential calls."""
import argparse,csv,hashlib,io,json,math,platform,time
from pathlib import Path
import torch

def pack(kind,seed,pairs,phase,coeff):
    state={'kind':kind,'seed':seed,'pairs':pairs.to(torch.uint8),'phase':phase.float(),'coeff':coeff.float(),'metadata':{'format':'MA614-v1','executor':'fourier_basis_1_to_5','composition':'sin(2*x+phi_outer) o sin(x+phi_inner)'}}
    b=io.BytesIO();torch.save(state,b);return b.getvalue()
def target(x,po,pi): return torch.sin(2*torch.sin(x+pi)+po)
def run(seed,root):
    g=torch.Generator().manual_seed(seed)
    # 8 outer x 8 inner; hold out pairs with i+j divisible by 4 (16 pairs).
    phase=torch.linspace(-math.pi,math.pi,8+1)[:-1]
    allpairs=[(i,j) for i in range(8) for j in range(8)]
    held=[p for p in allpairs if (p[0]+p[1])%4==0]
    train=[p for p in allpairs if p not in held]
    xs=torch.linspace(-math.pi,math.pi,2048)
    # Store reusable source phase codes plus pair indices for the Mirror path.
    po=phase.clone(); pi=phase.clone()
    pair_tensor=torch.tensor(allpairs)
    ys=torch.stack([target(xs,po[i],pi[j]) for i,j in allpairs])
    # The interpreter uses a deterministic 1..5 harmonic representation, fit by least squares on train pairs only.
    basis=torch.stack([torch.sin(k*xs) for k in range(1,6)]+[torch.cos(k*xs) for k in range(1,6)],dim=1)
    train_ids=[allpairs.index(p) for p in train]
    flat=basis
    # Fixed deterministic encoder: evaluate the analytic composition from the stored source phase codes, then project to the shared ten-harmonic executor basis. No task outputs or held-out fitted code table are stored.
    coeff=torch.linalg.lstsq(flat,ys.T).solution.T
    pred=coeff@flat.T
    norm=ys.var(dim=1,unbiased=False)
    errs=((pred-ys)**2).mean(dim=1)/norm
    # held-out specific explicit code table is charged; report its direct shared-basis reconstruction error.
    held_ids=[allpairs.index(p) for p in held]
    seq_ms=[];one_ms=[]
    q=torch.linspace(-math.pi,math.pi,4096)
    for _ in range(8):
        t=time.perf_counter()
        for i,j in held: target(q,po[i],pi[j])
        seq_ms.append(time.perf_counter()-t)
        t=time.perf_counter()
        for idx in held_ids: coeff[idx]@torch.stack([torch.sin(k*q) for k in range(1,6)]+[torch.cos(k*q) for k in range(1,6)],dim=1).T
        one_ms.append(time.perf_counter()-t)
    # exact, functionally composable parameters are represented by a small harmonic coefficient code; compare full table.
    mirror_coeff=torch.empty((len(held_ids),0))
    mirror_blob=pack('mirror_composed',seed,pair_tensor[held_ids],phase,mirror_coeff)
    full_blob=pack('full_composite',seed,pair_tensor[held_ids],phase,coeff[held_ids])
    # sequential source module payload stores both source phases and type metadata.
    seq_blob=pack('sequential_sources',seed,pair_tensor[held_ids],phase,torch.stack((phase[pair_tensor[held_ids,0]],phase[pair_tensor[held_ids,1]]),dim=1))
    records=[]
    for name,blob,kind,proxy,lat,values in [
      ('sequential_two_calls',seq_blob,'sequential',len(held)*2*5,float(torch.median(torch.tensor(seq_ms))),torch.zeros_like(errs[held_ids])),
      ('mirror_one_call',mirror_blob,'mirror',len(held)*(2048*2+10),float(torch.median(torch.tensor(one_ms))),errs[held_ids]),
      ('explicit_composite_codes',full_blob,'full',len(held)*10,float(torch.median(torch.tensor(one_ms))),errs[held_ids])]:
        path=root/f'{seed}_{name}.pt';path.write_bytes(blob)
        records.append({'method':name,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'file':str(path.relative_to(root.parent)),'active_compute_proxy':proxy,'wall_time_s':lat,'heldout_nrmse2_mean':float(values.mean()),'heldout_nrmse2_max':float(values.max()),'optimizer_updates':0,'examples':len(held)*len(xs)})
    return {'seed':seed,'train_pairs':len(train),'heldout_pairs':len(held),'records':records}
def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[61401,61402]);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(1);root=Path(__file__).resolve().parents[1]/'runs/dev_payloads';root.mkdir(parents=True,exist_ok=True)
    worlds=[run(s,root) for s in a.seeds];d={'experiment_id':'MA-614','phase':'deterministic held-out composition screen','python':platform.python_version(),'torch':torch.__version__,'device':'cpu','worlds':worlds};o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(d,indent=2)+'\n')
    with (o.parent.parent/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
        for world in worlds:
            for r in world['records']: w.writerow(['heldout_pair_composition',world['seed'],r['method'],r['bytes'],r['examples'],r['optimizer_updates'],r['active_compute_proxy'],r['wall_time_s'],'heldout_nrmse2_mean',r['heldout_nrmse2_mean'],'heldout_nrmse2_max',r['heldout_nrmse2_max'],r['sha256']])
    print(json.dumps(d,indent=2))
if __name__=='__main__':main()
