"""Balanced development-only CPU runtime repeat; no fresh labels are read."""
import importlib.util, json, random, time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ma966',HERE/'run_ma966.py')
ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)
torch.set_num_threads(1)
x,labels,_=ma.load_data();tr=ma.select_indices(labels,ma.TRAIN_LABELS);dv=ma.select_indices(labels,ma.DEV_LABELS)
rows=ma.flatten_trials(x,labels,dv);xx,ww,_,_=ma.tensor_batch(rows)
methods=['raw','periodic','mirror','offset','phaseonly','independent'];models={}
for method in methods: models[method],_=ma.fit_one(method,x,labels,tr,dv,96601)
with torch.no_grad():
 for method in methods:
  for _ in range(8): models[method](xx,ww)
per={m:[] for m in methods}
with torch.no_grad():
 for r in range(5):
  order=methods.copy();random.Random(96650+r).shuffle(order)
  for method in order:
   t0=time.perf_counter()
   for _ in range(20): models[method](xx,ww)
   per[method].append((time.perf_counter()-t0)/20)
result=[]
for m in methods:
 v=sorted(per[m]);med=v[len(v)//2]
 result.append({'condition':m,'seed':96601,'development_examples':len(xx),'seconds_per_404_window_examples':med,
                'examples_per_s':len(xx)/med,'repeat_seconds':per[m],
                'note':'median of 5 interleaved runs after warmup; development only'})
(HERE/'RUNTIME_DEVELOPMENT02.json').write_text(json.dumps(result,indent=2)+'\n')
for r in result: print(json.dumps(r,sort_keys=True),flush=True)
