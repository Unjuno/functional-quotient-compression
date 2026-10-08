import importlib.util
from pathlib import Path
import torch

SRC=Path(__file__).resolve().parents[1]/'source'/'run_experiment.py'
spec=importlib.util.spec_from_file_location('ma395',SRC);ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)


def test_domain_pair_split_has_no_training_leakage_and_full_coverage():
    data,held=ma.make_world(39501);t,d,_=data['train']
    assert not held[d,t].any()
    assert torch.unique(t).numel()==ma.V and torch.unique(d).numel()==ma.DOMAINS


def test_low_dimension_mirror_phase_shifts_class_basis():
    model=ma.FactorizedDomainModel('mirror')
    with torch.no_grad():
        model.emb.zero_();model.emb[5,5]=1
        model.proj.copy_(torch.eye(ma.ED,ma.MD));model.classifier.weight.zero_();model.classifier.weight[:,:ma.C].copy_(torch.eye(ma.C));model.classifier.bias.zero_()
        model.phase.copy_(torch.arange(ma.DOMAINS)*2*torch.pi/ma.DOMAINS)
    assert int(model(torch.tensor([5]),torch.tensor([2])).argmax(-1))==(5+2)%ma.C


def test_all_factorized_controls_return_eight_class_logits():
    for method in ('independent','shared','post_rank4','post_full','mirror'):
        m=ma.FactorizedDomainModel(method)
        assert m(torch.tensor([1,2]),torch.tensor([3,4])).shape==(2,ma.C)


def test_factorized_dimensions_match_registered_albert_bottleneck():
    m=ma.FactorizedDomainModel('mirror')
    assert m.emb.shape==(ma.V,ma.ED) and m.proj.shape==(ma.ED,ma.MD)

