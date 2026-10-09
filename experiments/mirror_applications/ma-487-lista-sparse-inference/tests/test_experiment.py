from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_orthogonal_dictionary_and_three_sparse_teacher():
 w=run.world(48701);assert torch.linalg.matrix_rank(w['atoms'].reshape(run.K,-1)).item()==run.K;assert torch.count_nonzero(w['coeff'],dim=1).eq(run.S).all()

def test_omp_and_direct_top3_recover_exact_codes():
 w=run.world(48701)
 for method in ('omp3','direct_top3'):
  z,_=run.encode(method,w);assert torch.max(torch.abs(z-w['coeff'])).item()<1e-5

def test_lista_payload_pays_dictionary_and_depth_state():
 w=run.world(48701);state,_=run.fit_lista(w,2,48701);b=run.payload('lista2',w,state);assert len(b)>w['atoms'].numel()*4
