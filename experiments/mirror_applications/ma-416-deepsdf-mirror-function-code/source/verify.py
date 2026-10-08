from __future__ import annotations
import csv,hashlib,json,io
from pathlib import Path
import numpy as np
import torch
from run import geometry_codes,sample_fields,replay,nrmse,pack_model,Decoder,DeepSDF,ellipse_sdf
ROOT=Path(__file__).resolve().parents[1]
def make_world(seed):
 g=torch.Generator().manual_seed(seed);train=geometry_codes(g,8);held=geometry_codes(g,8)
 x,y=sample_fields(train,1024,g);q,yq=sample_fields(held,2048,g);sx,sy=sample_fields(held,256,g);ip=(held[::2]+held[1::2])*.5;iq,iy=sample_fields(ip,2048,g)
 return held,q,yq,ip,iq,iy

def verify():
 rows=list(csv.DictReader((ROOT/'runs/RESULTS_CORE.csv').open()));assert len(rows)==8
 checks=[]
 for seed in (41601,41602):
  held,q,yq,ip,iq,iy=make_world(seed)
  raw={name:(ROOT/'runs'/f'dev{seed}_{name}.npz').read_bytes() for name in ('mirror','native_affine','deepsdf','analytic_oracle')}
  scores={name:(nrmse(replay(raw[name],q,'deepsdf' if name=='deepsdf' else 'mirror'),yq) if name!='analytic_oracle' else 0.) for name in raw}
  for name in raw:
   selected=next(r for r in rows if int(r['world_or_seed'])==seed and r['method']==name)
   assert abs(float(selected['primary_value'])-scores[name])<1e-6
   assert int(selected['serialized_bytes'])==len(raw[name])
   assert selected['status_note']==hashlib.sha256(raw[name]).hexdigest()
  with np.load(io.BytesIO(raw['analytic_oracle']),allow_pickle=False) as oracle:
   params=torch.tensor(oracle['geometry_codes'].astype(np.float32));assert params.shape==(8,5)
  assert nrmse(ellipse_sdf(q,params),yq)<1e-7
  assert nrmse(ellipse_sdf(iq,(params[::2]+params[1::2])*.5),iy)<1e-7
  # Recreate the held-out midpoint function from the serialized codes and shared decoder.
  for method,kind in (('mirror','mirror'),('deepsdf','deepsdf')):
   with np.load(io.BytesIO(raw[method]),allow_pickle=False) as a:
    codes=torch.tensor(a['instance_codes'].astype(np.float32));state={k[2:].replace('__','.'):torch.tensor(v.astype(np.float32)) for k,v in a.items() if k.startswith('w_')}
   model=DeepSDF() if kind=='deepsdf' else Decoder(2);model.load_state_dict(state)
   interp=(codes[::2]+codes[1::2])*.5;query,yt=iq,iy
   replay_payload=pack_model(model,interp,kind+'_interp');score=nrmse(replay(replay_payload,query,kind),yt)
   row=next(r for r in rows if int(r['world_or_seed'])==seed and r['method']==method)
   assert abs(float(row['secondary_value'])-score)<1e-5
  checks.append({'seed':seed,'metric_replay':'PASS','payload_replay':'PASS','methods':list(raw)})
 screen=json.loads((ROOT/'runs/screen.json').read_text());assert screen['fresh_accessed'] is False
 return {'experiment_id':'MA-416','status':'VERIFIED_DEVELOPMENT_SCREEN','protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),'metric_replay':'PASS','payload_replay':'PASS','fresh_accessed':False,'development_gate_passed':screen['development_gate_passed'],'checks':checks}
if __name__=='__main__':
 result=verify();(ROOT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
