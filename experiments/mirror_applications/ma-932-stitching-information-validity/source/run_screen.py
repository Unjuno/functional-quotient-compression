"""Run the frozen MA-932 development and fresh information-validity screen."""
import csv,json,statistics,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import MODES,REPS,Stitcher,make_split,ridge_probe,save_model

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'source'/'artifacts';ART.mkdir(exist_ok=True)
DEV=(93201,93202);FRESH=(93203,93204,93205);UPDATES=500;BATCH=128;LR=.003


def fit(seed,rep,mode,phase):
    torch.manual_seed(seed+MODES.index(mode)*31+REPS.index(rep)*401)
    model=Stitcher(mode,seed+MODES.index(mode)*31+REPS.index(rep)*401)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.0001)
    train_x,train_y=make_split(seed,'train',16384);x=train_x[rep]
    g=torch.Generator().manual_seed(seed+80_000+REPS.index(rep)*100+MODES.index(mode))
    model.train();t=time.perf_counter()
    for _ in range(UPDATES):
        idx=torch.randint(len(x),(BATCH,),generator=g);task=torch.randint(2,(BATCH,),generator=g)
        y=train_y[idx,task];pred=model(x[idx],task);loss=F.mse_loss(pred,y)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    wall=time.perf_counter()-t
    hold,targets=make_split(seed,'dev' if phase=='development' else 'fresh',4096);hx=hold[rep];model.eval()
    mse=[];r2=[]
    with torch.no_grad():
        for task in (0,1):
            ids=torch.full((len(hx),),task,dtype=torch.long);pred=model(hx,ids);y=targets[:,task]
            mse.append(float(F.mse_loss(pred,y)));r2.append(float(1-(pred-y).square().sum()/(y-y.mean()).square().sum()))
        # A third unrelated target is independent by construction; predict zero gives its Bayes baseline.
        unrelated=torch.randn(len(hx),generator=torch.Generator().manual_seed(seed+90_000))
        p0=model(hx,torch.zeros(len(hx),dtype=torch.long));noise_mse=float(F.mse_loss(p0,unrelated))
        for _ in range(2):model(hx[:128],torch.zeros(128,dtype=torch.long))
        t=time.perf_counter()
        for _ in range(8):model(hx[:128],torch.zeros(128,dtype=torch.long))
        throughput=128*8/(time.perf_counter()-t)
    path=ART/f'{phase}_{seed}_{rep}_{mode}.pt';payload=save_model(path,model,rep,seed);payload['path']=str(path.relative_to(ROOT))
    return {'seed':seed,'split':phase,'representation':rep,'mode':mode,'mse_task0':mse[0],'mse_task1':mse[1],
      'r2_task0':r2[0],'r2_task1':r2[1],'unrelated_random_target_mse':noise_mse,
      'train_examples':UPDATES*BATCH,'updates':UPDATES,'active_macs_per_example':model.active_macs(),
      'wall_seconds':wall,'throughput_examples_per_second':throughput,'payload':payload}


def main():
    torch.set_num_threads(1);rows=[];probe_rows=[]
    for phase,seeds in [('development',DEV),('fresh',FRESH)]:
      for seed in seeds:
       train,train_y=make_split(seed,'train',16384);hold,hold_y=make_split(seed,'dev' if phase=='development' else 'fresh',4096)
       for rep in REPS:
        probes=ridge_probe(train[rep],train_y)
        # Evaluate closed-form probe on same declared holdout.
        coef=torch.linalg.solve(train[rep].T@train[rep]+1e-4*torch.eye(8),train[rep].T@train_y)
        probe=((hold[rep]@coef-hold_y)**2).mean(0)
        probe_rows.append({'seed':seed,'split':phase,'representation':rep,'probe_u_mse':float(probe[0]),'probe_v_mse':float(probe[1])})
        for mode in MODES:
         row=fit(seed,rep,mode,phase);rows.append(row)
         print(phase,seed,rep,mode,row['mse_task0'],row['mse_task1'],row['payload']['bytes'],flush=True)
    fresh_gate=[]
    for seed in FRESH:
      by={(r['representation'],r['mode']):r for r in rows if r['seed']==seed and r['split']=='fresh'}
      probes={r['representation']:r for r in probe_rows if r['seed']==seed and r['split']=='fresh'}
      full=by[('full','mirror')];ind=by[('full','independent')];film=by[('full','film')]
      byte_match=abs(full['payload']['bytes']-film['payload']['bytes'])<=.05*full['payload']['bytes']
      fresh_gate.append({'seed':seed,
        'task0_only_matches_task0':abs(by[('full','mirror')]['mse_task0']-by[('task0_only','mirror')]['mse_task0'])<=.05,
        'task1_loss_reveals_missing_information':by[('task0_only','mirror')]['mse_task1']>=by[('full','mirror')]['mse_task1']+.5,
        'linear_probes_confirm_missing_information':probes['full']['probe_v_mse']<.05 and probes['task0_only']['probe_v_mse']>=.5 and probes['task0_only']['probe_u_mse']<.05 and probes['unrelated_noise']['probe_u_mse']>=.5 and probes['unrelated_noise']['probe_v_mse']>=.5,
        'mirror_quality_vs_independent':full['mse_task0']<=1.1*ind['mse_task0'] and full['mse_task1']<=1.1*ind['mse_task1'],
        'mirror_bytes_vs_independent':full['payload']['bytes']<=.8*ind['payload']['bytes'],
        'mirror_film_bytes_matched':byte_match,
        'mirror_beats_film_2pct':byte_match and statistics.mean([full['mse_task0'],full['mse_task1']])<=.98*statistics.mean([film['mse_task0'],film['mse_task1']])})
    audit_opened=all(all(v for k,v in c.items() if k!='seed') for c in fresh_gate)
    summary={'results':rows,'linear_probes':probe_rows,'fresh_gate':fresh_gate,'audit_opened':audit_opened,
      'audit':[],'protocol':{'updates':UPDATES,'batch':BATCH,'learning_rate':LR,'dev_seeds':DEV,'fresh_seeds':FRESH}}
    (ROOT/'source'/'screen_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=['condition','world_or_seed','method','serialized_bytes','adapter_only_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note']
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
      for r in rows:
       w.writerow({'condition':r['split'],'world_or_seed':r['seed'],'method':f"MA-932 {r['representation']} {r['mode']}",'serialized_bytes':r['payload']['bytes'],'adapter_only_bytes':r['payload']['bytes'],'train_tokens_or_examples':r['train_examples'],'optimizer_updates':r['updates'],'active_compute_proxy':r['active_macs_per_example'],'wall_time_s':r['wall_seconds'],'primary_metric':'task0/task1 MSE','primary_value':f"{r['mse_task0']:.10g};{r['mse_task1']:.10g}",'secondary_metric':'R2;throughput examples/s','secondary_value':f"{r['r2_task0']:.8g};{r['r2_task1']:.8g};{r['throughput_examples_per_second']:.6f}",'status_note':r['representation']})
    print(json.dumps({'fresh_gate':fresh_gate,'audit_opened':audit_opened},indent=2))


if __name__=='__main__':main()
