from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_updates_share_two_atom_basis():
    w=run.world(47001)
    assert torch.linalg.matrix_rank(w['deltas'].reshape(run.EDITS,-1),atol=1e-5).item()<=2

def test_router_selects_own_key_queries_and_rejects_locality_points():
    w=run.world(47001)
    own=[]
    for e,xq in enumerate(w['xq']):
        own.extend(run.route(x,w['keys'])[0]==e and run.route(x,w['keys'])[1] for x in xq)
    assert sum(own)/len(own)>=.99
    locality=[run.route(x,w['keys'])[1] for x in w['xl']]
    assert sum(locality)/len(locality)<=.01

def test_native_basis_is_exact_mirror_parameterization():
    a=run.init('mirror',47001);b=run.init('native_lowrank',47001)
    for x,y in zip(a['net'].parameters(),b['net'].parameters()):assert torch.equal(x,y)
    assert torch.equal(a['basis'],b['basis'])


def test_editor_accepts_matrix_signal_and_returns_expected_updates():
    w=run.world(47001)
    mirror=run.init('mirror',47001)
    mend=run.init('mend',47001)
    assert run.gen('mirror',mirror,w['signals'][0]).shape==(run.D,run.D)
    assert run.gen('mend',mend,w['signals'][0]).shape==(run.D,run.D)
