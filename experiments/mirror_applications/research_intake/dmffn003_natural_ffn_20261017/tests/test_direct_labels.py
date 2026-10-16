import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from model import SEEN,HOLD,Student
from direct_labels import loss_seen, compute_metrics

def test_direct_loss_does_not_observe_heldout_target_labels():
    logits=torch.randn(6,9,requires_grad=True)
    y=torch.randint(0,2,(6,9)).float()
    b=y.clone();b[:,HOLD]=1-b[:,HOLD]
    assert loss_seen(logits,y)==loss_seen(logits,b)
    gradient=torch.autograd.grad(loss_seen(logits,y),logits)[0]
    assert torch.count_nonzero(gradient[:,HOLD])==0

def test_output_metrics_include_real_labels_not_teacher_distillation():
    m=Student('mirror');x=torch.randn(4,64);y=torch.randint(0,2,(4,9)).float()
    metrics=compute_metrics(m,(x,y))
    assert 0<=metrics['seen_accuracy']<=1
    assert metrics['unseen_BCE']>=0
    assert metrics['worst_pair_BCE']>=0
