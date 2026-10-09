from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_function_bank_uses_three_atoms_from_shared_dictionary():
 w=run.world(48601);assert torch.count_nonzero(w['coeff'],dim=1).eq(run.S).all();assert torch.linalg.matrix_rank(w['atoms'].reshape(run.K,-1)).item()==run.K

def test_omp_recovers_target_sparse_codes():
 w=run.world(48601);idx,val,_,_=run.omp(w['matrices'],w['atoms']);re=torch.zeros(run.N,run.K);re.scatter_add_(1,idx,val)
 assert torch.max(torch.abs(re-w['coeff'])).item()<1e-5

def test_mirror_sparse_payload_aliases_native_omp():
 w=run.world(48601);s,_,_=run.fit(w,'mirror_sparse_float');assert run.serialize('mirror_sparse_float',s)==run.serialize('native_omp_float',s)
