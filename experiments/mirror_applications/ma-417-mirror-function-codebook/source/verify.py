from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import torch
from run import make_decoder,make_tasks,sample,decode_payload,nrmse
ROOT=Path(__file__).resolve().parents[1]
def verify():
 rows=list(csv.DictReader((ROOT/'runs/RESULTS_CORE.csv').open()));assert len(rows)==2*5*4
 checks=[]
 for seed in (41701,41702):
  _,ztrain,zheld,levels,g=make_tasks(seed);dec=make_decoder(seed);xq=sample(seed+3,64,2048);yq=dec(xq,zheld[:,None,:])
  ix=sample(seed+5,32,2048);zi=(zheld[::2]+zheld[1::2])*.5;yi=dec(ix,zi[:,None,:])
  for method,kind in [('deep','deep'),('vq','vq'),('native_vq','vq'),('pca','pca'),('oracle','oracle')]:
   raw=(ROOT/'runs'/f'dev{seed}_{method}.npz').read_bytes();pred,z=decode_payload(raw,xq,kind);interp_z=(z[::2]+z[1::2])*.5;interp=(dec.features(ix)*interp_z[:,None,:]).sum(-1)
   score=nrmse(pred,yq);isc=nrmse(interp,yi);digest=hashlib.sha256(raw).hexdigest()
   selected=[r for r in rows if int(r['world_or_seed'])==seed and r['method']==method]
   assert len(selected)==4
   for r in selected:
    rho=float(r['condition'].split('=')[1]);ids=[i for i,v in enumerate(levels) if v==rho]
    den=(yq[ids]-yq[ids].mean(1,keepdim=True)).pow(2).mean(1).sqrt().clamp_min(1e-8)
    q=((pred[ids]-yq[ids]).pow(2).mean(1).sqrt()/den).mean().item()
    assert abs(float(r['primary_value'])-q)<1e-6
    assert abs(float(r['secondary_value'])-isc)<1e-6
    assert int(r['serialized_bytes'])==len(raw) and r['status_note'].startswith(digest+';')
   if method=='vq':
    native=(ROOT/'runs'/f'dev{seed}_native_vq.npz').read_bytes();npred,_=decode_payload(native,xq,'vq');assert torch.max(torch.abs(pred-npred))==0
   checks.append({'seed':seed,'method':method,'payload_bytes':len(raw),'sha256':digest,'metric_replay':'PASS'})
 screen=json.loads((ROOT/'runs/screen.json').read_text());assert screen['fresh_accessed'] is False
 return {'experiment_id':'MA-417','status':'VERIFIED_DEVELOPMENT_SCREEN','protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),'metric_replay':'PASS','payload_replay':'PASS','fresh_accessed':False,'development_gate_passed':screen['development_gate_passed'],'checks':checks}
if __name__=='__main__':
 result=verify();(ROOT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
