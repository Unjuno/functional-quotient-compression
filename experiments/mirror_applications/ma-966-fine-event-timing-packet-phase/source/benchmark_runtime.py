"""Development-only CPU inference timing for the frozen MA-966 architecture."""
import importlib.util, json, time
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ma966',HERE/'run_ma966.py')
ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)
torch.set_num_threads(1)
x,labels,meta=ma.load_data();tr=ma.select_indices(labels,ma.TRAIN_LABELS);dv=ma.select_indices(labels,ma.DEV_LABELS)
rows=ma.flatten_trials(x,labels,dv);xx,ww,yy,_=ma.tensor_batch(rows)
results=[]
for method in ['raw','periodic','mirror','offset','phaseonly','independent']:
 model,_=ma.fit_one(method,x,labels,tr,dv,96601)
 model.eval()
 with torch.no_grad():
  for _ in range(5): model(xx,ww)
  t0=time.perf_counter()
  repeats=50
  for _ in range(repeats): model(xx,ww)
  elapsed=time.perf_counter()-t0
 results.append({'condition':method,'seed':96601,'development_examples':len(xx),'repeats':repeats,
                 'examples_per_s':len(xx)*repeats/elapsed,'elapsed_s':elapsed,
                 'note':'fixed-shape full-view CPU inference; excludes data loading and serialization'})
 print(json.dumps(results[-1],sort_keys=True),flush=True)
(HERE/'RUNTIME_DEVELOPMENT.json').write_text(json.dumps(results,indent=2)+'\n')
