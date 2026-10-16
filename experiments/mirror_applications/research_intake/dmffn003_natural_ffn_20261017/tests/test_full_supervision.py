import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from model import Student
from full_supervision import loss_all, compute_metrics

def test_all_nine_labels_influence_loss():
    x=torch.randn(3,9,requires_grad=True)
    target=torch.randint(0,2,(3,9)).float()
    gradient=torch.autograd.grad(loss_all(x,target),x)[0]
    assert (gradient.abs().sum(0)>0).all()

def test_metrics_are_finite():
    m=Student('mirror');x=torch.randn(5,64);y=torch.randint(0,2,(5,9)).float()
    metrics=compute_metrics(m,(x,y))
    assert set(metrics)>={'all_BCE','all_accuracy','worst_pair_BCE'}
    assert all(torch.isfinite(torch.tensor(k)) for k in metrics.values())
