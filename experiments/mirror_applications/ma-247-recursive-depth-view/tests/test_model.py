import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source"))
from model import Config,RecurrentModel,Teacher,rotate


def test_givens_rotation_roundtrip():
    x=torch.randn(13,8);a=torch.tensor(0.37)
    assert torch.allclose(x,rotate(rotate(x,a),a,inverse=True),atol=1e-6)


def test_all_depth_parameterizations_are_finite():
    cfg=Config();x=torch.randn(19,cfg.input_dim)
    for method in RecurrentModel.METHODS:
        model=RecurrentModel(cfg,method);y=model(x)
        assert y.shape==x.shape and torch.isfinite(y).all()
        assert model.inference_payload_bytes()>0


def test_mirror_steps_exactly_reconstruct_aligned_teacher():
    cfg=Config();teacher=Teacher(cfg,31);model=RecurrentModel(cfg,"mirror")
    with torch.no_grad():
        model.w1.copy_(teacher.w1);model.w2.copy_(teacher.w2);model.angles.copy_(teacher.angles)
        model.residual_gate_logits.copy_(torch.logit(teacher.gates))
    x=torch.randn(17,cfg.input_dim)
    assert torch.equal(model(x),teacher(x))


def test_step_scalar_gate_has_same_count_as_mirror_codes():
    cfg=Config();mirror=RecurrentModel(cfg,"mirror");gate=RecurrentModel(cfg,"scalar_gate")
    assert mirror.parameter_count()==gate.parameter_count()
