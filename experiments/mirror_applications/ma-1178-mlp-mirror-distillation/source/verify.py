"""Replay all 180 frozen student checkpoints and oracle conversion without retraining."""
from __future__ import annotations
import csv,hashlib,json,platform,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from model import DenseMLP,SharedStudent,new_student,teacher_mirror_family,teacher_film_family,param_count,teacher_outputs,packet_bytes
from data import student_sets,fresh_sets

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<19),b''):h.update(chunk)
 return h.hexdigest()

man=json.load(open(ROOT/'results'/'SEALED_TRAIN_MANIFEST.json'))
raw=json.load(open(ROOT/'results'/'FULL_RESULTS.json'))
res=raw['rows']; idx={(v['seed'],v['world'],v['width'],v['arm']):v for v in res}
assert len(man['records'])==180 and len(res)==180
assert man['fresh_generated'] is False and man['status']=='SEALED_BEFORE_FRESH'
assert man['protocol_file_sha256']==digest(ROOT/'PROTOCOL.json')
checks=0;max_abs=0.;oracle={};teacher_payload={};teacher_fit={}
for seed in [101,211,307,401,503]:
 path=ROOT/'checkpoints'/f'teacher_{seed}.pt'
 assert digest(path)==man['teacher_hashes'][str(seed)]
 td=torch.load(path,map_location='cpu',weights_only=True)
 base=DenseMLP(16);base.load_state_dict(td['base']);base.eval()
 indep=[]
 for state in td['independent']:
  t=DenseMLP(16);t.load_state_dict(state);t.eval();indep.append(t)
 groups={'mirror_related':teacher_mirror_family(base,td['mirror_dirs']),'film_related':teacher_film_family(base,td['film_codes']),'independent':indep}
 import io
 b=io.BytesIO();torch.save({'architecture':'3IndependentDenseTeachers','state':[{k:v.detach() for k,v in t.state_dict().items()} for t in groups['mirror_related']]},b)
 teacher_payload[str(seed)]=len(b.getvalue())
 with torch.inference_mode():
  mirror_oracle=SharedStudent(16,'mirror');mirror_oracle.up.load_state_dict(base.up.state_dict());mirror_oracle.down.load_state_dict(base.down.state_dict());mirror_oracle.codes.copy_(td['mirror_dirs'])
  film_oracle=SharedStudent(16,'film');film_oracle.up.load_state_dict(base.up.state_dict());film_oracle.down.load_state_dict(base.down.state_dict());film_oracle.codes.copy_(td['film_codes'])
 for world in ['mirror_related','film_related','independent']:
  ts=groups[world]
  _,dev,norm=student_sets(world,seed,ts)
  fresh,ood=fresh_sets(world,seed,ts)
  if world != 'independent':
   m=mirror_oracle if world=='mirror_related' else film_oracle
   x,r,y=dev
   with torch.inference_mode():
    oracle[f'{seed}_{world}']=((m(x,r)-y).square().mean()/norm).item()
  for width in [8,16,24]:
   for arm in ['mirror','film','shared','independent']:
    v=idx[(seed,world,width,arm)]
    ck=ROOT/'checkpoints'/v['checkpoint_file'];assert digest(ck)==v['checkpoint_sha256']
    state=torch.load(ck,map_location='cpu',weights_only=True)
    model=new_student(arm,width,10);model.load_state_dict(state['model_state']);model.eval()
    assert state['teacher_hash']==man['teacher_hashes'][str(seed)]
    assert param_count(model)==v['student_params']
    assert packet_bytes(model)==v['packet_bytes']
    with torch.inference_mode():
     for which,dat in [('fresh',fresh),('ood',ood)]:
      x,r,y=dat
      value=(model(x,r)-y).square().mean(dim=-1).mean().item()/norm
      delta=abs(value-v[f'{which}_nmse'])
      max_abs=max(max_abs,delta)
      if delta>1e-7:raise AssertionError(f'{seed,world,width,arm,which}: {value} vs {v[f"{which}_nmse"]}')
    checks+=1
vinfo={
 'id':'MA-1178','checked_checkpoints':checks,'all_hashes_match':True,'all_function_replays_match':True,'max_abs_nmse_diff':max_abs,'oracle_mirror_dev_nmse_max':max(value for key,value in oracle.items() if key.endswith('mirror_related')),'oracle_film_dev_nmse_max':max(value for key,value in oracle.items() if key.endswith('film_related')),
 'total_teachers_pretrained':5*4,'teacher_bundle_bytes':teacher_payload,'source':{p:digest(ROOT/'source'/p) for p in ['run.py','model.py','data.py','analyze.py','verify.py','posthoc_transfer.py']},'tests':'7 tests run separately','sealed_training_manifest_sha256':digest(ROOT/'results'/'SEALED_TRAIN_MANIFEST.json'),'complete_fresh_results_sha256':digest(ROOT/'results'/'FULL_RESULTS.json'),'protocol_sha256':digest(ROOT/'PROTOCOL.json'),
 'torch':torch.__version__,'python':platform.python_version(),'cpu_threads':torch.get_num_threads(),
 'experimental_limits':'synthetic low dimension; aligned teacher constructed in exact Mirror family; independent teachers require no relatedness; exploratory teacher-weight transfer performed only after primary Fresh evaluation'
}
(ROOT/'results'/'VERIFICATION.json').write_text(json.dumps(vinfo,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in vinfo.items() if k not in ('source','experimental_limits')},indent=2,ensure_ascii=False))
