"""Posthoc DEV-only causal token-order inversion probe; no test/fresh/ood access and no retraining."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch
from core import Model,make_data
from torch.nn import functional as F
p=Path(__file__).resolve().parents[1]
out=p/'results/parallel_run'
rows=[]
torch.set_num_threads(1)
for seed in [101,211,307]:
 for pos,arm in [('nope','shared'),('nope','mirror'),('hybrid','shared'),('hybrid','mirror'),('rope','shared'),('rope','mirror')]:
  f=out/'checkpoints'/f'last_matching_key_{pos}_{arm}_seed{seed}.pt'
  ck=torch.load(f,weights_only=True)
  m=Model(arm,pos,seed);m.load_state_dict(ck['state']);m.eval()
  x,y=make_data('last_matching_key',seed,640,2)
  flipped=x.clone()
  for i in range(len(x)):
   query=int(x[i,-1].item())-8
   j=torch.where(x[i,:-1]//2==query)[0]
   assert j.numel()==2 and int(x[i,j[0]]%2)!=int(x[i,j[1]]%2)
   a,b=int(j[0]),int(j[1])
   flipped[i,a]=x[i,b];flipped[i,b]=x[i,a]
  assert not torch.equal(flipped,x)
  with torch.no_grad():
   out0=m(x);out1=m(flipped)
   acc_orig=(out0.argmax(-1)==y).float().mean().item()
   acc_flip=(out1.argmax(-1)==1-y).float().mean().item()
   pred_flip_rate=(out0.argmax(-1)!=out1.argmax(-1)).float().mean().item()
  rows.append({'seed':seed,'pos':pos,'arm':arm,'dev_original_acc':acc_orig,'dev_flipped_label_acc':acc_flip,'prediction_flip_rate':pred_flip_rate,'n_dev':len(x)})
report={'status':'POSTHOC_DEV_DIAGNOSTIC_ONLY','description':'Exchange exactly two query-key-matching content tokens while holding other content and query fixed. Correct label inverts. No Fresh or OOD data used.','results':rows}
(p/'diagnostics/order_flip_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
for a in rows:print(a)
