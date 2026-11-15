"""Four-qubit TT-generated VQC screen; ideal statevector plus shot sampling."""
from __future__ import annotations
import hashlib, json, pathlib, struct, time
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'; RUNS.mkdir(exist_ok=True)
SEED=101608; TASKS=4; NQ=4; LAYERS=2; SHOTS=1024
PHASES=np.array([0.1,0.9,2.0,3.3],dtype=np.float64)

def cores():
    rng=np.random.default_rng(SEED)
    return (rng.normal(0,2.0,(2,LAYERS,2)),rng.normal(0,2.0,(2,NQ,2)),rng.normal(0,2.0,(2,2)))

def task_codes_native(phases): return np.stack([np.cos(phases),np.sin(phases)],axis=-1)
def tt_angles(code, shared):
    a,b,c=shared
    return np.einsum('r,rls,stq,qx->ltx',code,a,b,c,optimize=True)

def ry(theta):
    c=np.cos(theta/2); s=np.sin(theta/2)
    return np.array([[c,-s],[s,c]],dtype=np.complex128)
def rz(theta):
    return np.diag([np.exp(-.5j*theta),np.exp(.5j*theta)])

def oneq(state,U,q):
    shaped=state.reshape((2,)*NQ); moved=np.moveaxis(shaped,q,0); out=np.tensordot(U,moved,axes=(1,0)); return np.moveaxis(out,0,q).reshape(-1)
def cnot(state,control,target):
    out=state.copy()
    for i in range(1<<NQ):
        if ((i>>(NQ-1-control))&1):
            j=i^(1<<(NQ-1-target)); out[j]=state[i]
    return out

def expectation(bits,angles):
    state=np.zeros(1<<NQ,dtype=np.complex128); state[0]=1
    for q,bit in enumerate(bits): state=oneq(state,ry(np.pi*bit),q)
    for layer in range(LAYERS):
        for q in range(NQ):
            state=oneq(state,ry(angles[layer,q,0]),q)
            state=oneq(state,rz(angles[layer,q,1]),q)
        for q in range(NQ-1): state=cnot(state,q,q+1)
    prob=np.abs(state.reshape((2,)*NQ))**2
    return float(prob[0].sum()-prob[1].sum())

def patterns(): return [np.array([(i>>(NQ-1-q))&1 for q in range(NQ)],dtype=np.int64) for i in range(1<<NQ)]
def circuit_table(method,shared):
    if method=='independent_full':
        return np.stack([tt_angles(task_codes_native([p])[0],shared) for p in PHASES])
    if method=='native_tt': codes=task_codes_native(PHASES)
    elif method in ('mirror_phase','native_scalar_phase'): codes=task_codes_native(PHASES)
    else: raise ValueError(method)
    return np.stack([tt_angles(code,shared) for code in codes])

def pack(method,shared):
    a,b,c=shared
    common={'schema':'MA1016-TTVQC-inference-v1','qubits':NQ,'layers':LAYERS,'input_encoding':'RY(pi*bit)','readout':'Z on qubit 0','shots_per_input':SHOTS,'postprocessor':'sign of estimated Z expectation'}
    if method=='independent_full': entries=[('task_angles',circuit_table(method,shared))]
    elif method=='native_tt': entries=[('tt_task_core',task_codes_native(PHASES)),('tt_layer_core',a),('tt_qubit_core',b),('tt_axis_core',c)]
    else: entries=[('task_phase',PHASES),('tt_layer_core',a),('tt_qubit_core',b),('tt_axis_core',c)]
    header={**common,'tensors':[{'name':name,'shape':list(np.asarray(array).shape),'dtype':'float32-le'} for name,array in entries]}
    raw=json.dumps(header,separators=(',',':'),sort_keys=True).encode('utf8')+b'\n'
    return raw+b''.join(np.asarray(array,dtype='<f4').tobytes(order='C') for _,array in entries)

def unpack(payload):
    raw,body=payload.split(b'\n',1); header=json.loads(raw); arrays={}
    for tensor in header['tensors']:
        shape=tuple(tensor['shape']); size=int(np.prod(shape)); block=body[:size*4]; body=body[size*4:]
        arrays[tensor['name']]=np.frombuffer(block,dtype='<f4').reshape(shape).copy()
    assert not body
    return header,arrays

def run():
    t0=time.perf_counter(); shared=cores(); methods=('independent_full','native_tt','mirror_phase','native_scalar_phase'); angles={m:circuit_table(m,shared) for m in methods}; pats=patterns(); teacher=np.zeros((TASKS,len(pats)));
    for task in range(TASKS):
        for j,x in enumerate(pats): teacher[task,j]=expectation(x,angles['mirror_phase'][task])
    labels=np.where(teacher>=0,1,-1); pairwise={}
    for i in range(TASKS):
        for j in range(i+1,TASKS): pairwise[f'{i}-{j}']=float(np.mean(labels[i]!=labels[j]))
    results=[]
    for method in methods:
        wall=time.perf_counter(); preds=np.zeros_like(teacher); acc=[]; agree=[]
        for task in range(TASKS):
            for k,x in enumerate(pats):
                exp=expectation(x,angles[method][task]); preds[task,k]=exp
                rng=np.random.default_rng(SEED+task*100+k); plus=rng.binomial(SHOTS,(1+exp)/2); estimate=2*plus/SHOTS-1
                acc.append((1 if estimate>=0 else -1)==labels[task,k]); agree.append((1 if exp>=0 else -1)==labels[task,k])
        results.append({'method':method,'teacher_agreement':float(np.mean(agree)),'shot_accuracy':float(np.mean(acc)),'serialized_payload_bytes':len(pack(method,shared)),'qubits':NQ,'logical_gates':26,'two_qubit_gates':6,'depth':11,'shots_per_pattern':SHOTS,'patterns_per_task':len(pats),'total_shots':TASKS*len(pats)*SHOTS,'wall_s':time.perf_counter()-wall,'max_teacher_expectation_delta':float(np.max(np.abs(preds-teacher)))})
    same=pack('mirror_phase',shared)==pack('native_scalar_phase',shared)
    d={'schema':'MA1016-dev-v1','seed':SEED,'task_phases':PHASES.tolist(),'teacher_pairwise_label_disagreement':pairwise,'shots_per_pattern':SHOTS,'fresh_seeds':[1016301,1016307,1016313],'fresh_opened':False,'compiled_hardware':False,'simulator':'exact 16-amplitude complex128 statevector; binomial Z measurement','native_mirror_payload_identical_to_scalar_phase_control':same,'shared_tt_core_shapes':[list(x.shape) for x in shared],'results':results,'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'wall_total_s':time.perf_counter()-t0,'boundaries':['aligned synthetic TT-generated task family','no circuit training or tuning','no QPU/noise result','not official TensorHyper-VQC reproduction']}
    (RUNS/'development.json').write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps(d,indent=2))
if __name__=='__main__': run()
