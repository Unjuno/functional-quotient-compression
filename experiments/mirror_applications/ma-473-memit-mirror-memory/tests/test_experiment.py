from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_teacher_has_two_shared_layerwise_atoms():
    w=run.world(47301)
    for l in range(run.LAYERS):
        assert torch.linalg.matrix_rank(w['deltas'][:,l].reshape(run.EDITS,-1),atol=1e-5).item()<=2

def test_router_is_local_and_support_signal_is_flattened():
    w=run.world(47301)
    assert w['signals'].shape==(run.EDITS,run.LAYERS*run.D*run.D)
    for e,xq in enumerate(w['xq']):
        for x in xq:
            i,on=run.route(x,w['keys']);assert i==e and on
    assert sum(run.route(x,w['keys'])[1] for x in w['xl'])/len(w['xl'])<=.01

def test_native_cp_mirror_initialization_matches():
    a=run.init('mirror',47301);b=run.init('native_cp',47301)
    for x,y in zip(a['net'].parameters(),b['net'].parameters()):assert torch.equal(x,y)
    assert torch.equal(a['basis'],b['basis'])
    assert run.decode(a['net'](run.world(47301)['signals'][0]),a['basis']).shape==(run.LAYERS,run.D,run.D)
