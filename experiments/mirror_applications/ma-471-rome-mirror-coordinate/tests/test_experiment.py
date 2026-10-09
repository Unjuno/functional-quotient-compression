from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_teacher_edits_are_rank_one_and_share_rotated_template():
    w=run.world(47101)
    for d in w['deltas']:
        assert torch.linalg.matrix_rank(d,atol=1e-5).item()==1
    assert w['keys'].shape==(run.EDITS,run.D)

def test_router_is_local_and_selects_own_query():
    w=run.world(47101);ok=[]
    for e,xq in enumerate(w['xq']):
        for x in xq:
            i,on=run.route(x,w['keys']);ok.append(i==e and on)
    assert sum(ok)/len(ok)>=.99
    assert not any(run.route(x,w['keys'])[1] for x in w['xl'])

def test_mirror_native_givens_initial_states_equal():
    a=run.init('mirror',47101);b=run.init('native_givens',47101)
    for x,y in zip(a['net'].parameters(),b['net'].parameters()):assert torch.equal(x,y)
    assert torch.equal(a['u'],b['u']) and torch.equal(a['v'],b['v'])
