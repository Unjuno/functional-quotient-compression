import importlib.util
from pathlib import Path
import zipfile
import numpy as np
import torch

SRC=Path(__file__).resolve().parents[1]/'source'/'run_experiment.py'
spec=importlib.util.spec_from_file_location('ma392',SRC);ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)


def test_heldout_split_covers_each_token_and_domain_without_leakage():
    splits,mask=ma.make_world(39201)
    train_t,train_d,_=splits['train']
    assert not mask[train_d,train_t].any()
    assert torch.unique(train_t).numel()==ma.V and torch.unique(train_d).numel()==ma.DOMAINS
    assert mask.float().mean().item()>.15 and mask.float().mean().item()<.35


def test_fourier_mirror_phase_performs_expected_domain_shift():
    m=ma.DomainModel('mirror')
    with torch.no_grad():
        m.token.zero_();m.token[7,7]=1
        m.phase.copy_(torch.arange(8)*2*torch.pi/8)
    t=torch.tensor([7]);d=torch.tensor([3])
    assert int(m(t,d).argmax(-1))==(7+3)%8


def test_all_controls_produce_domain_conditioned_class_logits():
    for method in ('independent','bias','rank4','domain_map','mirror'):
        m=ma.DomainModel(method);assert m(torch.tensor([1,2]),torch.tensor([3,4])).shape==(2,ma.C)


def test_mirror_payload_charges_domain_phase(tmp_path):
    m=ma.DomainModel('mirror');n,d=ma.archive(tmp_path/'m.zip',m)
    assert n==(tmp_path/'m.zip').stat().st_size and len(d)==64
    with zipfile.ZipFile(tmp_path/'m.zip') as z:
        assert 'arrays/phase.npy' in z.namelist()
        assert np.load(z.open('arrays/phase.npy')).shape==(ma.DOMAINS,)

