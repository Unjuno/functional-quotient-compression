import json,hashlib
from pathlib import Path
import numpy as np,torch
from run_experiment import ITransformer,load,origins,score,SEED
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
y,tr,dv=load();oa=origins(len(y),tr,dv,'audit',12)
res=json.loads((ROOT/'runs/results.json').read_text())
replayed={}
for typ in ['shared','mirror','film']:
 m=ITransformer(16,typ); state=torch.load(ROOT/'runs'/typ/'weights.pt',map_location='cpu',weights_only=True);m.load_state_dict(state);replayed[typ]=score(m,y,oa)
ind=torch.load(ROOT/'runs/independent/weights.pt',map_location='cpu',weights_only=True)
models=[]
for st in ind:
 m=ITransformer(1,'shared');m.load_state_dict(st);models.append(m)
vals=[]
with torch.no_grad():
 for o in oa:
  cs=[]
  for c,m in enumerate(models):
   x=np.stack([y[o-96:o,c:c+1]])
   tar=y[o:o+24,c]
   p=m(torch.from_numpy(x))[0,:,0].numpy();cs.append(np.mean((p-tar)**2))
  vals.append(float(np.mean(cs)))
replayed['independent']=float(np.mean(vals))
for k,v in replayed.items():
 expected=res[k]['audit_mse']; got=v if isinstance(v,float) else v
 assert abs(expected-got)<1e-7,(k,expected,got)
manifest={}
for k,files in {'shared':['weights.pt','inference.json'],'mirror':['weights.pt','inference.json'],'film':['weights.pt','inference.json'],'independent':['weights.pt','inference.json']}.items():
 for fn in files:
  p=ROOT/'runs'/k/fn;b=p.read_bytes();manifest[f'{k}/{fn}']={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
(ROOT/'runs/payload_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'replayed_audit_mse':replayed,'max_abs_delta':max(abs(res[k]['audit_mse']-(v if isinstance(v,float) else v)) for k,v in replayed.items()),'payload_manifest':manifest},indent=2,sort_keys=True))
