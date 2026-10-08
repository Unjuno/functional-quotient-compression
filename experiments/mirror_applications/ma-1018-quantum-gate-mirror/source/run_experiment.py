import csv,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from model import make_tasks,fit_cp,cp_basis,fit_tt,tt_basis,tt_reconstruct,fit_linear_basis,fit_code,unitary,unitary_metrics
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'source';ART=SRC/'artifacts';ART.mkdir(exist_ok=True)
SEEDS=[10181,10182];RHOS=[.1,.4,1.0];RANKS=[1,2,4,8];HELD=[6,7];PRIVATE_K=8;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def save(obj,path):
 torch.save(obj,path);b=Path(path).read_bytes();return len(b),hashlib.sha256(b).hexdigest()
def cpu32(x):return x.detach().cpu().float().clone()
def cp_predict(obj,task):return obj['mean']+torch.einsum('r,lr,qr,ar->lqa',obj['codes'][task],obj['layer_factors'],obj['qubit_factors'],obj['axis_factors'])
def tt_predict(obj,task):return obj['mean']+tt_reconstruct(obj['codes'][task],obj['cores'])
def matrix_predict(obj,task):return obj['mean']+torch.einsum('r,rlqa->lqa',obj['codes'][task],obj['basis'])
def predict(obj,method,task):
 if method=='independent':return obj['angles'][task]
 if method=='hard_shared':return obj['mean']
 if method=='matrix_lowrank':z=matrix_predict(obj,task)
 elif method=='tt_hyper':z=tt_predict(obj,task)
 elif method in ('mirror_cp','mirror_cp_private'):z=cp_predict(obj,task)
 else:raise ValueError(method)
 if method=='mirror_cp_private':
  v=torch.zeros(72);v[obj['private_indices'][task]]=obj['private_values'][task];z=z+v.reshape(6,4,3)
 return z

def sparse_residual(target,pred,k=PRIVATE_K):
 flat=(target-pred).reshape(-1);idx=torch.topk(flat.abs(),k).indices;return idx.cpu(),flat[idx].cpu()
def code_payload(method,obj,task):
 if method=='independent':state={'task_id':task,'angles':obj['angles'][task]}
 elif method=='hard_shared':state={'task_id':task,'shared':obj['mean']}
 elif method=='matrix_lowrank':state={'task_id':task,'code':obj['codes'][task]}
 elif method=='tt_hyper':state={'task_id':task,'code':obj['codes'][task]}
 else:
  state={'task_id':task,'code':obj['codes'][task]}
  if method=='mirror_cp_private':state.update({'private_indices':obj['private_indices'][task],'private_values':obj['private_values'][task]})
 return state

def main():
 raw_rows=[];libraries=[];start_all=time.perf_counter()
 for seed in SEEDS:
  for rho in RHOS:
   target,_=make_tasks(seed,rho);train=target[:6];mean=train.mean(0)
   methods=[]
   # Fixed shared angle baseline and independent upper reference.
   methods.append(('hard_shared',0,{'format':'MA1018-library-v1','method':'hard_shared','rank':0,'mean':cpu32(mean),'task_ids':list(range(8)),'skeleton':{'qubits':4,'layers':6,'rotations':72,'cnots':18,'depth':36}}))
   methods.append(('independent',0,{'format':'MA1018-library-v1','method':'independent','rank':0,'angles':cpu32(target),'task_ids':list(range(8)),'skeleton':{'qubits':4,'layers':6,'rotations':72,'cnots':18,'depth':36}}))
   # Matrix low-rank basis control.
   for rank in RANKS:
    mu,basis,codes=fit_linear_basis(train,rank);code_table=torch.zeros(8,len(basis),dtype=torch.float64);code_table[:6]=codes
    for t in HELD:code_table[t]=fit_code(target[t],mu,basis)
    methods.append(('matrix_lowrank',rank,{'format':'MA1018-library-v1','method':'matrix_lowrank','rank':rank,'mean':cpu32(mu),'basis':cpu32(basis),'codes':cpu32(code_table),'task_ids':list(range(8)),'skeleton':{'qubits':4,'layers':6,'rotations':72,'cnots':18,'depth':36}}))
   # TT-SVD hypernetwork control.
   for rank in RANKS:
    centered=train-mean;cores=fit_tt(centered,rank);basis=tt_basis(cores);rr=len(basis);code_table=torch.zeros(8,rr,dtype=torch.float64);code_table[:6]=cores[0]
    for t in HELD:code_table[t]=fit_code(target[t],mean,basis)
    methods.append(('tt_hyper',rank,{'format':'MA1018-library-v1','method':'tt_hyper','rank':rank,'mean':cpu32(mean),'cores':[cpu32(c) for c in cores[1:]],'codes':cpu32(code_table),'task_ids':list(range(8)),'skeleton':{'qubits':4,'layers':6,'rotations':72,'cnots':18,'depth':36}}))
   # CP Mirror across task x layer x qubit x axis.
   cp_objects={}
   for rank in RANKS:
    factors,loss=fit_cp(train-mean,rank,seed+int(rho*1000)+rank,1200);cpb=cp_basis(factors);rr=cpb.shape[0];code_table=torch.zeros(8,rr,dtype=torch.float64);code_table[:6]=factors[0]
    for t in HELD:code_table[t]=fit_code(target[t],mean,cpb)
    base={'format':'MA1018-library-v1','method':'mirror_cp','rank':rank,'mean':cpu32(mean),'layer_factors':cpu32(factors[1]),'qubit_factors':cpu32(factors[2]),'axis_factors':cpu32(factors[3]),'codes':cpu32(code_table),'task_ids':list(range(8)),'factor_fit_mse':loss,'skeleton':{'qubits':4,'layers':6,'rotations':72,'cnots':18,'depth':36}}
    methods.append(('mirror_cp',rank,base));cp_objects[rank]=(base,target,cpb)
    if rank in (2,4,8):
     priv=dict(base);priv['method']='mirror_cp_private';priv['private_indices']=torch.zeros(8,PRIVATE_K,dtype=torch.long);priv['private_values']=torch.zeros(8,PRIVATE_K)
     for t in range(8):
      code=priv['codes'][t];pred=cp_predict(priv,t);ids,vals=sparse_residual(target[t],pred)
      priv['private_indices'][t]=ids;priv['private_values'][t]=vals.float()
     methods.append(('mirror_cp_private',rank,priv))
   for method,rank,obj in methods:
    libpath=ART/f'seed{seed}_rho{rho}_{method}_r{rank}_K8.pt';libbytes,libsha=save(obj,libpath)
    libraries.append({'seed':seed,'rho':rho,'method':method,'rank':rank,'path':str(libpath.relative_to(ROOT)),'bytes':libbytes,'sha256':libsha})
    for task in HELD:
     state=code_payload(method,obj,task);cpath=ART/f'cond_seed{seed}_rho{rho}_{method}_r{rank}_task{task}.pt';cb,ch=save(state,cpath)
     tar=target[task].cpu().double()
     t0=time.perf_counter();angle=predict(obj,method,task).detach().cpu().double();pred_u=unitary(angle.numpy());elapsed=time.perf_counter()-t0
     target_u=unitary(tar.numpy());pf,bf=unitary_metrics(target_u,pred_u)
     raw_rows.append({'seed':seed,'rho':rho,'task_id':task,'method':method,'rank':rank,'process_fidelity':pf,'mean_basis_state_fidelity':bf,'library_bytes':libbytes,'condition_state_bytes':cb,'library_sha256':libsha,'condition_state_sha256':ch,'train_task_count':6,'cp_optimizer_updates':1200 if method.startswith('mirror_cp') else 0,'angle_reconstruction_ops':rank*72 if rank else 0,'logical_gate_count':90,'logical_depth':36,'simulator_complex_amplitude_updates':90*16*16,'inference_wall_s':elapsed,'fresh_accessed':False,'oracle_target_angles_used':True})
 # Core table retains each development task/method/rank.
 fields=['seed','rho','task_id','method','rank','library_bytes','condition_state_bytes','train_task_count','cp_optimizer_updates','angle_reconstruction_ops','logical_gate_count','logical_depth','simulator_complex_amplitude_updates','inference_wall_s','process_fidelity','mean_basis_state_fidelity','library_sha256','condition_state_sha256']
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows({k:r[k] for k in fields} for r in raw_rows)
 (SRC/'development_raw.json').write_text(json.dumps({'experiment_id':'MA-1018','rows':raw_rows,'libraries':libraries,'seeds':SEEDS,'heterogeneity_rho':RHOS,'ranks':RANKS,'fresh_accessed':False,'wall_total_s':time.perf_counter()-start_all},indent=2)+'\n')
 print(json.dumps({'rows':len(raw_rows),'libraries':len(libraries),'mean_process_fidelity':sum(x['process_fidelity'] for x in raw_rows)/len(raw_rows),'fresh_accessed':False,'wall_total_s':time.perf_counter()-start_all},indent=2))
if __name__=='__main__':main()
