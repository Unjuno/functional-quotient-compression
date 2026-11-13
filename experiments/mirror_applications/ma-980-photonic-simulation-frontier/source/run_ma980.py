"""Ideal complex-linear simulation; no photonic hardware metrics."""
import csv, hashlib, io, json
from pathlib import Path
import numpy as np
import torch

R=Path(__file__).resolve().parents[1]
DEV=(98001,); FRESH=(98011,98013,98017); TASKS=8; N=4
H=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],np.complex128)/2

def phase(seed):
 rng=np.random.default_rng(seed)
 return rng.uniform(-np.pi,np.pi,size=(TASKS,2,N))

def operator(phi,t):
 return np.diag(np.exp(1j*phi[t,0])) @ H @ np.diag(np.exp(1j*phi[t,1]))

def blob(obj):
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()

def evaluate(seed):
 p=phase(seed);targets=[operator(p,t) for t in range(TASKS)]
 rng=np.random.default_rng(seed+7); probes=(rng.normal(size=(16,N))+1j*rng.normal(size=(16,N)))/np.sqrt(2)
 recon={
  'shared': [H for _ in range(TASKS)],
  'direct_phase': [operator(p,t) for t in range(TASKS)],
  'mirror_phase': [operator(p,t) for t in range(TASKS)],
  'independent_dense': targets,
 }
 states={
  'shared': {'shared_core':torch.tensor(H),'task_ids':list(range(TASKS)),'metadata':{'n':N,'kind':'shared-H'}},
  'direct_phase': {'shared_core':torch.tensor(H),'input_phase':torch.tensor(p[:,1]),'output_phase':torch.tensor(p[:,0]),'task_ids':list(range(TASKS)),'metadata':{'n':N,'kind':'native-phase-bank'}},
  'mirror_phase': {'shared_core':torch.tensor(H),'m_input':torch.tensor(p[:,1]),'m_output':torch.tensor(p[:,0]),'task_ids':list(range(TASKS)),'metadata':{'n':N,'kind':'mirror-view'}},
  'independent_dense': {'operators':torch.tensor(np.stack(targets)),'task_ids':list(range(TASKS)),'metadata':{'n':N,'kind':'independent-dense'}},
 }
 out=[]; aliases=[]
 for c in recon:
  pay=blob(states[c]); loaded=torch.load(io.BytesIO(pay),map_location='cpu',weights_only=False)
  if c=='shared': decoded=[loaded['shared_core'].numpy() for _ in range(TASKS)]
  elif c=='direct_phase': decoded=[np.diag(np.exp(1j*loaded['output_phase'][t].numpy()))@loaded['shared_core'].numpy()@np.diag(np.exp(1j*loaded['input_phase'][t].numpy())) for t in range(TASKS)]
  elif c=='mirror_phase': decoded=[np.diag(np.exp(1j*loaded['m_output'][t].numpy()))@loaded['shared_core'].numpy()@np.diag(np.exp(1j*loaded['m_input'][t].numpy())) for t in range(TASKS)]
  else: decoded=list(loaded['operators'].numpy())
  errs=[];probe=[]
  for t in range(TASKS):
   errs.append(float(np.linalg.norm(decoded[t]-targets[t])/np.linalg.norm(targets[t])))
   probe.append(float(np.max(np.abs(probes@decoded[t].T-probes@targets[t].T))))
   out.append({'seed':seed,'task':t,'condition':c,'relative_frobenius_error':errs[-1],'max_probe_error':probe[-1],'payload_bytes':len(pay),'complex_macs':16,'phase_ops':0 if c in ('shared','independent_dense') else 8})
  aliases.append({'condition':c,'payload_sha256':hashlib.sha256(pay).hexdigest(),'bytes':len(pay),'mean_relative_error':float(np.mean(errs)),'max_probe_error':max(probe)})
 return out,aliases

def main():
 allrows=[]; worlds=[]
 for seed in DEV+FRESH:
  rows,detail=evaluate(seed)
  if seed in FRESH: allrows.extend(rows);worlds.append({'seed':seed,'payloads':detail})
  else:(R/'source'/'DEVELOPMENT.json').write_text(json.dumps({'seed':seed,'payloads':detail},indent=2)+'\n')
 with (R/'RESULTS_CORE.csv').open('a',newline='') as f: csv.DictWriter(f,fieldnames=list(allrows[0]),lineterminator='\n').writerows(allrows)
 (R/'source'/'FRESH.json').write_text(json.dumps(worlds,indent=2)+'\n')
 print(json.dumps(worlds,sort_keys=True))

if __name__=='__main__':main()
