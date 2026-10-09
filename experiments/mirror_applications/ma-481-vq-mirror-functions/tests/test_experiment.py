from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_teacher_function_family_has_rank_four_shared_atoms():
    w=run.world(48101);assert torch.linalg.matrix_rank(w['matrices'].reshape(run.N,-1),atol=1e-5).item()<=run.R

def test_vq_codebooks_fit_only_train_functions_and_emit_address_per_function():
    w=run.world(48101);s,_,_=run.fit(w,'mirror_vq16',48101)
    assert s['centers'].shape==(16,run.R)
    assert s['indices'].shape==(run.N,)
    assert int(s['indices'].max())<16

def test_mirror_native_vq_bank_is_identical():
    w=run.world(48101);s,_,_=run.fit(w,'mirror_vq16',48101)
    assert run.serialize('mirror_vq16',s)==run.serialize('native_vq16',s)
