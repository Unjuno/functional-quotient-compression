"""Recompute the fixed development screen and compare deterministic metrics."""
import hashlib, importlib.util, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
EXP=HERE.parent
spec=importlib.util.spec_from_file_location('ma462_replay',HERE/'run_experiment.py')
ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)
reference=json.loads((HERE/'development_raw.json').read_text())
replayed=ma.run_stage('development',[4621,4622],ma.ALPHAS,[2,4,8],300)
keys=['world_seed','alpha','method','rank','serialized_bytes','train_examples','optimizer_updates','train_macs_proxy','adapter_gen_macs_per_task','adapter_apply_macs_per_example','inference_macs_proxy','train_loss','seen_nmse','heldout_nmse','private_state_bytes']
if len(reference['rows'])!=len(replayed):raise SystemExit('row count mismatch')
max_abs=0.0
for ix,(a,b) in enumerate(zip(reference['rows'],replayed)):
 for k in keys:
  va,vb=a[k],b[k]
  if isinstance(va,float) and va is not None:
   delta=abs(va-vb);max_abs=max(max_abs,delta)
   if delta>1e-10:raise SystemExit(f'row {ix} {k}: {va} != {vb}')
  elif va!=vb:raise SystemExit(f'row {ix} {k}: {va} != {vb}')
result={'checked':True,'rows':len(replayed),'fields':keys,'note':'torch.save archive SHA is recorded per execution and may vary despite identical tensors; actual byte length and tensor metrics are replayed','max_abs_float_delta':max_abs,'reference_sha256':hashlib.sha256((HERE/'development_raw.json').read_bytes()).hexdigest(),'fresh_accessed':False}
(EXP/'source'/'metric_replay.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
