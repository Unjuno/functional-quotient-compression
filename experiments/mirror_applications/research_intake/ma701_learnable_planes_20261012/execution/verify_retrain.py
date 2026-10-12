import json,hashlib
from pathlib import Path
import torch
from source.experiment import Model,make_world,fit,save_model,load_model,metrics
from source.planes import orthonormal_planes

torch.set_num_threads(1);torch.set_num_interop_threads(1)
rows=[]
cases=[Path('runs/dev/digits8/seed17101/learned_0/metric.json'),Path('runs_adjust/dev/aligned/seed27101/learned_0/step2400/metric.json')]
for i,p in enumerate(cases):
 r=json.loads(p.read_text());w=make_world(r['seed'],r['family']);m=Model(r['method'],w['di'],w['o'],w['k'],r['seed'],r['cfg']['sigma'],w['digits'])
 fit(m,w,dict(r['cfg'],steps=r['steps']),r['seed'])
 got=save_model(m,Path('replays')/f'retrain{i}.npz')
 assert got['sha256']==r['payload']['sha256'],(r['family'],got,r['payload'])
 rows.append(dict(seed=r['seed'],family=r['family'],steps=r['steps'],sha256=got['sha256'],exact=True))
Path('RETRAIN_VERIFICATION.json').write_text(json.dumps(rows,indent=2));print(rows)
# Separate diagnostic of overfit, no checkpoint selection using test scores.
score=[]
for root in ['runs','runs_adjust']:
 for p in sorted(Path(root+'/dev/digits8').rglob('metric.json')):
  r=json.loads(p.read_text())
  if r['label'] not in ['learned','fixed_matched','multihead','film']:continue
  m=load_model(p.parent/'model.npz');w=make_world(r['seed'],'digits8')
  row=dict(study=root,seed=r['seed'],method=r['label'],steps=r['steps'])
  for sp in ['train','val','test']:
   v=metrics(m,w,sp);row[sp+'_loss']=v['loss'];row[sp+'_accuracy']=v['accuracy']
  if r['label']=='learned':
   u,v=orthonormal_planes(m.a,m.b);g=v.unsqueeze(-1)*u.unsqueeze(-2)-u.unsqueeze(-1)*v.unsqueeze(-2)
   row['commutator_mean']=float(torch.stack([(g[i]@g[j]-g[j]@g[i]).norm() for i in range(4) for j in range(i)]).mean().detach())
  score.append(row)
Path('OVERFIT_DIAGNOSTIC.json').write_text(json.dumps(score,indent=2))
