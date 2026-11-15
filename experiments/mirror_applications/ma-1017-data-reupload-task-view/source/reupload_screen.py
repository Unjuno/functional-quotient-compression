"""Synthetic data-reupload task-view screen with an ideal 2-qubit simulator."""
from __future__ import annotations
import hashlib,json,pathlib,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'; RUNS.mkdir(exist_ok=True)
SEED=101702; TASKS=4; BLOCKS=3; SHOTS=1024; PHASES=np.array([.1,.85,1.75,2.9])

def angles(): return np.random.default_rng(SEED).normal(0,1.2,(BLOCKS,2,2))
def ry(a):
 c=np.cos(a/2);s=np.sin(a/2);return np.array([[c,-s],[s,c]],complex)
def rz(a):return np.diag([np.exp(-.5j*a),np.exp(.5j*a)])
def apply(state,U,q):
 t=state.reshape(2,2); t=np.tensordot(U,t,axes=(1,q)); return np.moveaxis(t,0,q).reshape(-1)
def cnot(state):return state[[0,1,3,2]]
def expectation(x,phi,theta):
 s=np.array([1,0,0,0],complex)
 for b in range(BLOCKS):
  s=apply(s,ry(np.pi*x[0]+phi),0);s=apply(s,ry(np.pi*x[1]-phi),1)
  for q in range(2):
   s=apply(s,ry(theta[b,q,0]),q);s=apply(s,rz(theta[b,q,1]),q)
  s=cnot(s)
 return float(np.abs(s[0])**2+np.abs(s[1])**2-np.abs(s[2])**2-np.abs(s[3])**2)
def inputs():
 v=np.linspace(-1,1,4);return [np.array([x,y]) for x in v for y in v]
def pack(method,theta):
 common={'schema':'MA1017-reupload-inference-v1','qubits':2,'blocks':BLOCKS,'encoding':'RY(pi*x0+m), RY(pi*x1-m)','entangler':'CNOT 0->1 after each block','readout':'Z0 sign','shots_per_input':SHOTS}
 if method=='independent_full': arrays=[('task_angles',np.repeat(theta[None],TASKS,axis=0)),('task_phase',PHASES)]
 elif method=='shared_no_view': arrays=[('shared_angles',theta)]
 else: arrays=[('shared_angles',theta),('task_phase',PHASES)]
 header={**common,'tensors':[{'name':n,'shape':list(a.shape),'dtype':'float32-le'} for n,a in arrays]}
 return json.dumps(header,separators=(',',':'),sort_keys=True).encode()+b'\n'+b''.join(np.asarray(a,dtype='<f4').tobytes() for _,a in arrays)
def run():
 t0=time.perf_counter();theta=angles();xs=inputs();teacher=np.array([[expectation(x,p,theta) for x in xs] for p in PHASES]); labels=np.where(teacher>=0,1,-1)
 distinct=[tuple(row.tolist()) for row in labels]; pairwise={f'{i}-{j}':float(np.mean(labels[i]!=labels[j])) for i in range(TASKS) for j in range(i+1,TASKS)}
 methods=('shared_no_view','independent_full','mirror_reupload','native_scalar_schedule'); results=[]
 for method in methods:
  st=time.perf_counter(); agrees=[]; correct=[]
  for task in range(TASKS):
   phase=0. if method=='shared_no_view' else PHASES[task]
   task_theta=theta
   if method=='independent_full': task_theta=np.repeat(theta[None],TASKS,axis=0)[task]
   for j,x in enumerate(xs):
    e=expectation(x,phase,task_theta); agrees.append((1 if e>=0 else -1)==labels[task,j])
    rng=np.random.default_rng(SEED+task*100+j); zplus=rng.binomial(SHOTS,(1+e)/2); correct.append((1 if 2*zplus/SHOTS-1>=0 else -1)==labels[task,j])
  results.append({'method':method,'teacher_agreement':float(np.mean(agrees)),'shot_accuracy':float(np.mean(correct)),'serialized_payload_bytes':len(pack(method,theta)),'qubits':2,'logical_gates':21,'two_qubit_gates':3,'depth':12,'shots_per_input':SHOTS,'total_shots':TASKS*len(xs)*SHOTS,'wall_s':time.perf_counter()-st})
 d={'schema':'MA1017-dev-v1','seed':SEED,'task_phases':PHASES.tolist(),'distinct_task_functions':len(set(distinct)),'pairwise_label_disagreement':pairwise,'shots_per_input':SHOTS,'fresh_seeds':[1017301,1017307,1017313],'fresh_opened':False,'compiled_hardware':False,'simulator':'exact 4-amplitude complex128 statevector plus seeded binomial Z shots','mirror_native_scalar_payload_identical':pack('mirror_reupload',theta)==pack('native_scalar_schedule',theta),'results':results,'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'wall_total_s':time.perf_counter()-t0,'boundaries':['aligned synthetic teacher','no training/tuning','no QPU or noisy-device evaluation','not an official data-reuploading benchmark']}
 (RUNS/'development.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':run()
