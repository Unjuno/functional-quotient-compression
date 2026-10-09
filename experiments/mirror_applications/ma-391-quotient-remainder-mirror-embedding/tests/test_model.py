import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))

from model import METHODS, QRBank, from_payload, qr_indices  # noqa:E402
from run import make_world, save_payload, load_payload, score  # noqa:E402


def test_quotient_remainder_addresses_are_unique():
    ids=torch.arange(1024); q,r=qr_indices(ids)
    pairs=torch.stack((q,r),dim=1)
    assert torch.unique(pairs,dim=0).shape[0]==1024
    assert int(q.max())==31 and int(r.max())==31


def test_mirror_reconstructs_unit_norm_coefficients():
    w=make_world(39101)
    bank=QRBank('mirror',39102,w['q_table'],w['r_table'],w['decoder'],w['projection'],w['projection_seed'])
    alpha=torch.stack((torch.cos(bank.angle),torch.sin(bank.angle)),dim=-1)
    torch.testing.assert_close((alpha*alpha).sum(1),torch.ones(1024),atol=2e-7,rtol=0)


def test_every_method_roundtrips_payload_and_scores(tmp_path):
    w=make_world(39102)
    for method in METHODS:
        bank=QRBank(method,39103,w['q_table'],w['r_table'],w['decoder'],w['projection'],w['projection_seed'])
        path=tmp_path/f'{method}.npz'; save_payload(path,bank)
        loaded=load_payload(path)
        assert loaded.method==method
        assert score(loaded,w)==score(bank,w)


def test_candidate_and_direct_control_payload_states():
    w=make_world(39101)
    mirror=QRBank('mirror',11,w['q_table'],w['r_table'],w['decoder'],w['projection'],w['projection_seed'])
    coeff=QRBank('coeff',11,w['q_table'],w['r_table'],w['decoder'],w['projection'],w['projection_seed'])
    assert mirror.angle.numel()==1024
    assert coeff.coeff.numel()==2048
