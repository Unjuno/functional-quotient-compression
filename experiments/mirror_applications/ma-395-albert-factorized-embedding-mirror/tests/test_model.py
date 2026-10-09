import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
from model import DIM,DOMAINS,LATENT,METHODS,FactorizedDomainBank,from_payload  # noqa:E402
from run import make_world,rotate,save_payload,load_payload,score  # noqa:E402

def test_latent_rotation_preserves_norm():
    z=torch.randn(8,128,LATENT);a=torch.randn(8,LATENT//2)
    torch.testing.assert_close(rotate(z[0],a[:1]).squeeze(0).norm(dim=-1),z[0].norm(dim=-1),atol=2e-6,rtol=2e-6)

def test_domain_split_has_train_and_heldout_for_all_tokens():
    w=make_world(39501);m=w['heldout']
    assert w['train_count']+w['heldout_count']==DOMAINS*128
    assert bool((~m).any(0).all()) and bool(m.any())

def test_mirror_code_matches_latent_givens_dimension():
    w=make_world(39502);b=FactorizedDomainBank('mirror',w['base'],w['projection'],w['decoder'])
    assert b.angle.shape==(DOMAINS,LATENT//2)

def test_all_methods_roundtrip_exact_metrics(tmp_path):
    w=make_world(39501)
    for method in METHODS:
        oracle=w['teacher'] if method=='oracle' else None
        bank=FactorizedDomainBank(method,w['base'],w['projection'],w['decoder'],oracle)
        p=tmp_path/f'{method}.npz';save_payload(p,bank);loaded=load_payload(p)
        assert loaded.method==method and score(loaded,w)==score(bank,w)
