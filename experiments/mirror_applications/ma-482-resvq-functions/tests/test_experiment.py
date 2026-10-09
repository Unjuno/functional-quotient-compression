from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_teacher_bank_is_rank_four():
 w=run.world(48201);assert torch.linalg.matrix_rank(w['matrices'].reshape(run.N,-1),atol=1e-5).item()<=run.R

def test_residual_code_sequence_matches_function_count():
 w=run.world(48201);s,_,_=run.fit(w,'mirror_rvq8x3',48201)
 assert s['centers'].shape==(3,8,run.R);assert s['indices'].shape==(run.N,3)

def test_mirror_and_native_rvq_are_exact_same_parameterization():
 w=run.world(48201);s,_,_=run.fit(w,'mirror_rvq16x2',48201)
 assert run.serialize('mirror_rvq16x2',s)==run.serialize('native_rvq16x2',s)
