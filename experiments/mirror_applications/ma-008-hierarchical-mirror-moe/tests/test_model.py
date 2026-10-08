import sys
from pathlib import Path

import torch
import torch.nn.functional as F

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'source'))
from model import Config, METHODS, HierarchicalMoE, Teacher  # noqa: E402
from engine import make_balanced  # noqa: E402


def test_balanced_hierarchical_labels():
    cfg=Config(); x,y=make_balanced(2,128,cfg)
    assert x.shape==(128,16)
    assert torch.bincount(y,minlength=4).tolist()==[32,32,32,32]
    assert torch.all((y//2)<2)


def test_all_methods_backprop_and_serialize():
    cfg=Config(); x,labels=make_balanced(3,64,cfg)
    teacher=Teacher(cfg,'aligned',4); target=teacher.forward(x,labels)
    for method in METHODS:
        model=HierarchicalMoE(cfg,method)
        pred,role,primary,child=model(x)
        assert pred.shape==(64,16) and role.shape==(64,)
        assert primary.shape[0]==64
        loss=F.mse_loss(pred,target)+model.router_loss(primary,child,labels)
        loss.backward()
        payload=model.serialize()
        state=torch.load(__import__('io').BytesIO(payload),weights_only=False)['state_dict']
        clone=HierarchicalMoE(cfg,method); clone.load_state_dict(state)
        assert torch.equal(clone(x)[0],pred)


def test_hierarchical_group_and_child_route_shapes():
    cfg=Config(); x,_=make_balanced(5,64,cfg)
    model=HierarchicalMoE(cfg,'mirror_hier')
    _,role,group_logits,child_logits=model(x)
    assert group_logits.shape==(64,2)
    assert child_logits.shape==(64,2)
    assert int(role.min())>=0 and int(role.max())<4
