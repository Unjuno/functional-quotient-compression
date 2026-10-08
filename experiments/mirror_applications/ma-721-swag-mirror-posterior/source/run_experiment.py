import csv,hashlib,json,time
from pathlib import Path
import torch
from model import DigitsMLP,data_splits,fit,flatten,load_flat,metrics,posterior_logits,rank1_logits,disagreement
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'source';ART=SRC/'payloads';ART.mkdir(exist_ok=True)
SEEDS=[7211,7212];RANKS=[4,8,16];SIGMAS=[0.0,.025,.05,.1];K=8;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def save(obj,path):
 torch.save(obj,path);b=Path(path).read_bytes();return len(b),hashlib.sha256(b).hexdigest()
def shifted(x,seed):
 g=torch.Generator().manual_seed(seed);return (x+0.25*torch.randn(x.shape,generator=g)).clamp(0,1)
def evaluate_probs(logits,y,members):
 met=metrics(logits,y);met['pairwise_disagreement']=disagreement(members);return met

def main():
 xtr,ytr,xdev,ydev,xaudit,yaudit=data_splits();rows=[];payloads=[];train_info=[];dev_cache=[];rank1_selected=[]
 for sd in SEEDS:
  start=time.perf_counter();model,snaps,updates=fit(sd,xtr,ytr,capture=True);swag_train_wall=time.perf_counter()-start
  snaps=torch.stack(snaps);mean=snaps.mean(0);devs=snaps-mean
  _,s,vh=torch.linalg.svd(devs,full_matrices=False);basis=vh;scales=s/(len(snaps)-1)**.5;diag=(devs.square().sum(0)/(len(snaps)-1)).sqrt()
  # Independent full-checkpoint ensemble upper reference.
  ind_states=[];ind_updates=0;ind_wall=0.0
  for j in range(K):
   st=time.perf_counter();im,_,iu=fit(sd*100+j+72210,xtr,ytr,capture=False);ind_wall+=time.perf_counter()-st;ind_updates+=iu;ind_states.append(flatten(im))
  ipath=ART/f'seed{sd}_independent_K8.pt';ib,ih=save({'format':'MA721-independent-ensemble-v1','members':ind_states,'member_count':K},ipath);payloads.append({'key':f'seed{sd}_independent_K8','path':str(ipath.relative_to(ROOT)),'bytes':ib,'sha256':ih})
  spath=ART/f'seed{sd}_swa_mean.pt';sb,sh=save({'format':'MA721-swa-mean-v1','mean':mean.clone()},spath);payloads.append({'key':f'seed{sd}_swa_mean','path':str(spath.relative_to(ROOT)),'bytes':sb,'sha256':sh})
  # Save/score SWA mean and matched Gaussian / Mirror posterior representations.
  conds=[('clean',xdev),('corrupt_sigma0.25',shifted(xdev,sd+880))]
  for ci,(condition,xx) in enumerate(conds):
   # SWA point predictor.
   st=time.perf_counter();point=load_flat(DigitsMLP(),mean)(xx).clamp(-40,40);wall=time.perf_counter()-st
   rows.append({'seed':sd,'condition':condition,'method':'swa_mean','rank':0,'sigma':0,'k_members':1,'train_updates':updates,'train_wall_s':swag_train_wall,'inference_wall_s':wall,'examples_per_s':len(xx)/wall,'mac_proxy_per_example':17024,'weight_reconstruction_ops_per_member':0,'metrics':metrics(point,ydev),'payload_key':f'seed{sd}_swa_mean','serialized_payload_bytes':sb})
   # Ensemble checkpoint upper control.
   ind_probs=[];st=time.perf_counter()
   for state in ind_states:ind_probs.append(load_flat(DigitsMLP(),state)(xx).softmax(-1))
   ip=torch.stack(ind_probs);ipred=ip.mean(0).clamp_min(1e-12).log();wall=time.perf_counter()-st
   rows.append({'seed':sd,'condition':condition,'method':'independent_ensemble','rank':0,'sigma':0,'k_members':K,'train_updates':ind_updates,'train_wall_s':ind_wall,'inference_wall_s':wall,'examples_per_s':len(xx)/wall,'mac_proxy_per_example':17024*K,'weight_reconstruction_ops_per_member':0,'metrics':evaluate_probs(ipred,ydev,ip),'payload_key':f'seed{sd}_independent_K8','serialized_payload_bytes':ib})
   for rank in RANKS:
    b=basis[:rank];sc=scales[:rank]
    for method in ('swag_gaussian','mirror_rademacher','gaussian_lowrank_only'):
     dd=diag if method!='gaussian_lowrank_only' else None
     rademacher=method=='mirror_rademacher';sseed=sd*100000+rank*100+ci
     st=time.perf_counter();logits,mem=posterior_logits(mean,b,sc,dd,xx,sseed,K,rademacher);wall=time.perf_counter()-st
     met=evaluate_probs(logits,ydev,mem)
     payload={'format':'MA721-posterior-v1','method':method,'model':'64-128-64-10 MLP','mean':mean.clone(),'basis':b.clone(),'eigen_scales':sc.clone(),'diagonal_std':dd.clone() if dd is not None else None,'posterior_members':K,'sample_seed':sseed,'sample_rule':'Rademacher low-rank coordinates' if rademacher else 'Gaussian low-rank coordinates; independent diagonal Gaussian' if dd is not None else 'Gaussian low-rank only'}
     key=f'seed{sd}_{method}_r{rank}_{condition}';path=ART/(key+'.pt');n,dig=save(payload,path);payloads.append({'key':key,'path':str(path.relative_to(ROOT)),'bytes':n,'sha256':dig})
     rows.append({'seed':sd,'condition':condition,'method':method,'rank':rank,'sigma':0,'k_members':K,'train_updates':updates,'train_wall_s':swag_train_wall,'inference_wall_s':wall,'examples_per_s':len(xx)/wall,'mac_proxy_per_example':17024*K,'weight_reconstruction_ops_per_member':rank*17226+(17226 if dd is not None else 0),'metrics':met,'payload_key':key,'serialized_payload_bytes':n})
   for sigma in SIGMAS:
    sseed=sd*100000+int(sigma*10000)+ci
    st=time.perf_counter();logits,mem=rank1_logits(mean,xx,sseed,sigma,K);wall=time.perf_counter()-st
    met=evaluate_probs(logits,ydev,mem);key=f'seed{sd}_rank1_sigma{sigma}_{condition}'
    payload={'format':'MA721-rank1-control-v1','method':'posthoc_rank1_factor','mean':mean.clone(),'global_factor_sigma':sigma,'posterior_members':K,'sample_seed':sseed,'factor_rule':'independent Normal(1,sigma) row/column factors per linear layer'}
    path=ART/(key+'.pt');n,dig=save(payload,path);payloads.append({'key':key,'path':str(path.relative_to(ROOT)),'bytes':n,'sha256':dig})
    rows.append({'seed':sd,'condition':condition,'method':'rank1_factor','rank':1,'sigma':sigma,'k_members':K,'train_updates':updates,'train_wall_s':swag_train_wall,'inference_wall_s':wall,'examples_per_s':len(xx)/wall,'mac_proxy_per_example':17024*K,'weight_reconstruction_ops_per_member':17226,'metrics':met,'payload_key':key,'serialized_payload_bytes':n})
  # Independent ensemble payload counted once per world.
  # Rank1 sigma is picked only from development corruption NLL, then frozen for reporting.
  corrupt=[r for r in rows if r['seed']==sd and r['condition']=='corrupt_sigma0.25' and r['method']=='rank1_factor']
  best=min(corrupt,key=lambda r:r['metrics']['nll']);rank1_selected.append({'seed':sd,'selected_sigma':best['sigma'],'selection_metric':'corrupt development NLL','nll':best['metrics']['nll']})
 (SRC/'development_raw.json').write_text(json.dumps({'experiment_id':'MA-721','rows':rows,'payloads':payloads,'rank1_sigma_selection':rank1_selected,'base_seeds':SEEDS,'fresh_accessed':False,'audit_labels_used':False},indent=2)+'\n')
 # Flat result table keeps every method/condition/seed and exact serialized size.
 fields=['seed','condition','method','rank','sigma','k_members','train_updates','train_wall_s','inference_wall_s','examples_per_s','mac_proxy_per_example','weight_reconstruction_ops_per_member','serialized_payload_bytes','nll','accuracy','ece10','brier','pairwise_disagreement','payload_key']
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
  for r in rows:w.writerow({**{k:r[k] for k in fields if k in r},**r['metrics']})
 print(json.dumps({'rows':len(rows),'payloads':len(payloads),'rank1_selection':rank1_selected,'fresh_accessed':False},indent=2))
if __name__=='__main__':main()
