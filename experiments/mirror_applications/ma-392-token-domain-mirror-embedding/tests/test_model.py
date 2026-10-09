import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))

from model import DIM,DOMAINS,METHODS,VOCAB,DomainBank,from_payload  # noqa:E402
from run import apply_rotation,make_world,save_payload,load_payload,score  # noqa:E402


def test_split_covers_seen_and_unseen_combinations():
    world=make_world(39201);mask=world['heldout_mask']
    assert int(mask.sum())==world['heldout_pair_count']
    assert bool((~mask).any(dim=0).all())
    assert bool((~mask).any(dim=1).all())
    assert 0 < int(mask.sum()) < DOMAINS*VOCAB


def test_givens_transform_is_norm_preserving():
    torch.manual_seed(1);x=torch.randn(7,VOCAB,DIM);a=0.4*torch.randn(7,DIM//2)
    y=apply_rotation(x[0],a[:1]).squeeze(0)
    torch.testing.assert_close(y.norm(dim=-1),x[0].norm(dim=-1),atol=2e-6,rtol=2e-6)


def test_mirror_has_small_domain_code():
    world=make_world(39202)
    bank=DomainBank('mirror',world['base'],world['decoder'])
    assert bank.angle.shape==(DOMAINS,DIM//2)
    assert bank.angle.numel()==DOMAINS*DIM//2


def test_all_saved_methods_roundtrip_exact_metrics(tmp_path):
    world=make_world(39201)
    for method in METHODS:
        oracle=world['teacher'] if method=='oracle' else None
        bank=DomainBank(method,world['base'],world['decoder'],oracle)
        path=tmp_path/f'{method}.npz';save_payload(path,bank)
        loaded=load_payload(path)
        assert loaded.method==method
        assert score(loaded,world)==score(bank,world)
