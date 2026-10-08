"""Train and evaluate the frozen MA-981 ULA beam-codebook screen."""
import csv,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import METHODS,NSECTOR,NCODE,SNR,BeamCodebook,make_channels,rates,save_payload

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'source'/'artifacts';ART.mkdir(exist_ok=True)
DEV=(98101,98102);FRESH=(98103,98104,98105);UPDATES=500;BATCH=128;LR=.02;TEMP=.2


def fit(seed,method,split):
    torch.manual_seed(seed+METHODS.index(method)*131)
    model=BeamCodebook(method,seed+METHODS.index(method)*19)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(params,lr=LR) if params else None
    train=make_channels(seed,'train',8192)
    gen=torch.Generator().manual_seed(seed+80_000+METHODS.index(method))
    start=time.perf_counter()
    if optimizer:
      model.train()
      for _ in range(UPDATES):
        sector=int(torch.randint(NSECTOR,(1,),generator=gen));idx=torch.randint(train.shape[1],(BATCH,),generator=gen)
        beam=model.beams(sector);r=rates(train[sector,idx],beam)
        soft=TEMP*(torch.logsumexp(r/TEMP,dim=1)-math.log(NCODE));loss=-soft.mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    train_wall=time.perf_counter()-start
    channels=make_channels(seed,split,4096);by_sector=[];acc=[];throughput=[]
    model.eval()
    with torch.no_grad():
      for sector in range(NSECTOR):
        beam=model.beams(sector);r=rates(channels[sector],beam);best=r.max(1).values;by_sector.append(float(best.mean()))
        matched=channels[sector]/channels[sector].abs().clamp_min(1e-12)
        matched=matched/math.sqrt(model.beams(sector).shape[-1])
        matched_gain=(channels[sector].conj()*matched).sum(-1).abs().square()
        upper=torch.log2(1+SNR*matched_gain)
        acc.append(float((best>=upper-0.1).float().mean()))
        for _ in range(2):rates(channels[sector,:256],beam)
        t=time.perf_counter()
        for _ in range(10):rates(channels[sector,:256],beam)
        throughput.append(256*10/(time.perf_counter()-t))
    path=ART/f'{split}_{seed}_{method}.pt';payload=save_payload(path,model,seed);payload['path']=str(path.relative_to(ROOT))
    rows={'seed':seed,'split':split,'method':method,'rate_by_sector':by_sector,'mean_rate':sum(by_sector)/NSECTOR,
      'worst_sector_rate':min(by_sector),'continuous_match_accuracy':sum(acc)/NSECTOR,
      'train_examples':UPDATES*BATCH if optimizer else 0,'updates':UPDATES if optimizer else 0,
      'active_compute_proxy':model.active_proxy(),'train_wall_seconds':train_wall,
      'throughput_examples_per_second':sum(throughput)/NSECTOR,'payload':payload}
    return rows


def main():
    torch.set_num_threads(1);rows=[]
    for split,seeds in [('development',DEV),('fresh',FRESH)]:
      for seed in seeds:
       for method in METHODS:
        r=fit(seed,method,split);rows.append(r)
        print(split,seed,method,r['mean_rate'],r['payload']['bytes'],flush=True)
    gate=[]
    for seed in FRESH:
      by={r['method']:r for r in rows if r['seed']==seed and r['split']=='fresh'}
      m,i,f=by['mirror_view'],by['independent_sector'],by['rank2_view']
      matched=abs(m['payload']['bytes']-f['payload']['bytes'])<=.05*m['payload']['bytes']
      gate.append({'seed':seed,'quality_vs_independent':m['mean_rate']>=i['mean_rate']-.20,
        'bytes_vs_independent':m['payload']['bytes']<=.75*i['payload']['bytes'],
        'byte_matched_rank2':matched,'mirror_beats_rank2':matched and m['mean_rate']>=f['mean_rate']+.10})
    summary={'results':rows,'fresh_gate':gate,'audit_opened':False,'audit':[],
      'protocol':{'dev_seeds':DEV,'fresh_seeds':FRESH,'updates':UPDATES,'batch':BATCH,'lr':LR,'snr_db':10}}
    (ROOT/'source'/'screen_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=['condition','world_or_seed','method','serialized_bytes','adapter_only_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note']
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
      for r in rows:
       w.writerow({'condition':r['split'],'world_or_seed':r['seed'],'method':r['method'],'serialized_bytes':r['payload']['bytes'],'adapter_only_bytes':r['payload']['bytes'],'train_tokens_or_examples':r['train_examples'],'optimizer_updates':r['updates'],'active_compute_proxy':r['active_compute_proxy'],'wall_time_s':r['train_wall_seconds'],'primary_metric':'mean sector spectral efficiency','primary_value':r['mean_rate'],'secondary_metric':'worst sector rate;continuous-match accuracy','secondary_value':f"{r['worst_sector_rate']:.9g};{r['continuous_match_accuracy']:.9g}",'status_note':'3-bit limited feedback; 8 beams; phase-only 8-element ULA'})
    print(json.dumps({'fresh_gate':gate,'audit_opened':False},indent=2))


if __name__=='__main__':main()
