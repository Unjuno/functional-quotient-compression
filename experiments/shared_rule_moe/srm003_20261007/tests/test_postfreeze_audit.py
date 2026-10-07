"""Additional audit tests; these do not change the frozen experiment sources."""
import pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'source'))
from core import *
from engine import BASE

def test_nonzero_compaction_matches_manually_restricted_router():
    c=dict(BASE,kind='hybrid');m=make_model(c,20)
    torch.nn.init.normal_(m.residual.private.up,std=.1)
    keep=[1,3,6,7];n=compact(m,keep)
    x=torch.randn(8,32);b=m.residual.private
    scores=torch.nn.functional.linear(x,b.router.weight[keep],b.router.bias[keep])
    val,idx=scores.topk(2,-1);weights=val.softmax(-1)
    out=[]
    for i in range(len(x)):
        row=sum(weights[i,j]*(x[i]@b.down[keep[int(idx[i,j])]]@b.up[keep[int(idx[i,j])]]) for j in range(2))
        out.append(row)
    assert torch.allclose(n.residual.private(x),torch.stack(out),atol=1e-6)
    nn=compact(n,[0,2]);assert nn.residual.private.slot_ids.tolist()==[1,6]

def test_decoding_does_not_change_global_random_state():
    m=make_model(BASE,20);raw=encode_model(m)
    state=torch.random.get_rng_state().clone();decode_model(raw)
    assert torch.equal(state,torch.random.get_rng_state())

def test_nonaffine_private_labels_are_not_single_basis_assignments():
    w=make_world(66001)
    assert w['private_truth'].sum()==6
    # Dataset only exports tokens+targets to the network; no integer routing argument.
    import inspect
    assert list(inspect.signature(Model.forward).parameters)==['self','tokens','return_all']
