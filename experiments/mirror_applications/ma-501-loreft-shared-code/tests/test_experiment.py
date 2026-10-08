from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_synthetic_full_matrix_fit_replays_targets():
    w=run.world(23,0.25)
    for t in range(run.T):
        assert torch.allclose(w['est'][t],w['delta'][t],atol=1e-5,rtol=1e-5)

def test_shared_code_is_causal_and_native_alias_is_exact():
    w=run.world(24,0.0);obj=run.make_objects('shared_mirror',w)
    raw=run.pack('shared_mirror',obj);native=run.pack('shared_mirror',run.make_objects('shared_mirror',w))
    assert raw==native
    before=run.apply('shared_mirror',obj,w['xte'][12],12)
    changed={k:v.clone() for k,v in obj.items()};changed['codes'][12].zero_()
    after=run.apply('shared_mirror',changed,w['xte'][12],12)
    assert float((before-after).abs().max())>1e-5

def test_shared_payload_is_smaller_than_independent_rank_four():
    w=run.world(25,0.0)
    shared=run.pack('shared_mirror',run.make_objects('shared_mirror',w))
    independent=run.pack('task_loreft_r4',run.make_objects('task_loreft_r4',w))
    assert len(shared)<0.5*len(independent)
