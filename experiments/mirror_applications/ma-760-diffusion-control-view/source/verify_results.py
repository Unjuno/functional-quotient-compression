import hashlib,json
from pathlib import Path
import torch
from run_experiment import make_x,gen_tasks,conv,apply_view
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'source';ART=SRC/'artifacts'
raw=json.loads((SRC/'development_raw.json').read_text());base=ART/'tiny_unet_state.pt'
def check(path,n,d):
 b=path.read_bytes();assert len(b)==n,(path,len(b),n);assert hashlib.sha256(b).hexdigest()==d,(path,'sha');return b
if not base.exists():
 from diffusers import StableDiffusionPipeline
 pipe=StableDiffusionPipeline.from_pretrained(raw['base_model'],revision=raw['revision'],safety_checker=None,requires_safety_checker=False)
 ART.mkdir(parents=True,exist_ok=True)
 torch.save({'model':raw['base_model'],'revision':raw['revision'],'unet_state':pipe.unet.state_dict(),'config':dict(pipe.unet.config)},base)
base_obj=torch.load(base,map_location='cpu',weights_only=False)
base_bytes=len(base.read_bytes());base_sha=hashlib.sha256(base.read_bytes()).hexdigest()
assert base_bytes==raw['base_unet_bytes'];assert base_sha==raw['base_unet_sha256']
libraries={}
for x in raw['libraries']:
 p=ROOT/x['artifact'];obj=torch.load(p,map_location='cpu',weights_only=True);check(p,x['adapter_payload_bytes'],x['adapter_sha256'])
 assert obj['base_unet_sha256']==base_sha and obj['base_unet_bytes']==base_bytes
 libraries[(x['seed'],x['rho'],x['method'],x['rank'])]=obj
max_delta=0.0;checked=0
for r in raw['results']:
 key=(r['world_or_seed'],r['rho'],r['method'],r['rank']);obj=libraries[key];c=r['condition_id'];
 cp=ART/f"tmp_condition_seed{r['world_or_seed']}_rho{r['rho']}_{r['method']}_r{r['rank']}_c{c}.pt"
 # Standalone per-condition serialized address/private state must also round-trip.
 condition_bytes=cp.read_bytes();assert len(condition_bytes)==r['marginal_condition_payload_bytes']
 assert hashlib.sha256(condition_bytes).hexdigest()==r['condition_payload_sha256']
 condition=torch.load(cp,map_location='cpu',weights_only=True);assert condition['condition_id']==c
 seed=r['world_or_seed'];rho=r['rho'];true_k,_=gen_tasks(seed,rho);x=make_x(seed*1000000+int(rho*100000)+c,128);target=conv(x,true_k[c])
 method=r['method'];rank=r['rank']
 if method=='no_control':got=torch.zeros_like(target)
 elif method=='hard_shared':got=conv(x,obj['kernel'])
 elif method=='independent':got=conv(x,obj['kernels'][c])
 else:
  code=obj['codes'][c];res=None
  if method=='mirror_private':
   res=torch.zeros(108);res[obj['private_indices'][c]]=obj['private_values'][c];res=res.reshape(4,3,3,3)
  got=apply_view(x,obj['mean'],obj['basis'],code,res)
 delta=abs(float((got-target).square().mean())-r['primary_value']);max_delta=max(max_delta,delta)
 if delta>1e-8:raise SystemExit(f"metric replay delta {delta}: {key}, condition {c}")
 assert r['integration_finite'];checked+=1
rep={'checked':True,'rows':checked,'library_payloads':len(libraries),'base_unet_bytes':base_bytes,'base_unet_sha256':base_sha,'metric':'heldout_residual_mse','max_abs_metric_delta':max_delta,'payload_sha256_and_actual_sizes_checked':True,'condition_payload_roundtrips_checked':True,'fresh_accessed':False}
(SRC/'metric_replay.json').write_text(json.dumps(rep,indent=2)+'\n')
print(json.dumps(rep,indent=2))
