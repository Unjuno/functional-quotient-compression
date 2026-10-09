import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
from model import BUCKETS,DIM,VOCAB,METHODS,ProductBank,product_indices  # noqa:E402
from run import make_world,save_payload,load_payload,score  # noqa:E402

def test_product_address_has_exactly_two_occupants():
    ids=torch.arange(VOCAB);a,b=product_indices(ids);keys=a*BUCKETS+b;counts=torch.bincount(keys,minlength=BUCKETS**2)
    assert int((counts>0).sum())==1024 and int(counts.min())==int(counts.max())==2

def test_mirror_has_one_angle_per_logical_token():
    w=make_world(39701);m=ProductBank('mirror',9,w['table0'],w['table1'],w['decoder'])
    assert m.angle.shape==(VOCAB,)

def test_component_shapes_and_pair_address_repeat():
    ids=torch.arange(1024);a,b=product_indices(ids);a2,b2=product_indices(ids+1024)
    assert torch.equal(a,a2) and torch.equal(b,b2)
    w=make_world(39703);bank=ProductBank('mirror',5,w['table0'],w['table1'],w['decoder'])
    assert bank.table0.shape==bank.table1.shape==(BUCKETS,DIM)

def test_every_method_roundtrips_bytes_and_metrics(tmp_path):
    w=make_world(39702)
    for method in METHODS:
        bank=ProductBank(method,15,w['table0'],w['table1'],w['decoder'])
        p=tmp_path/f'{method}.npz';save_payload(p,bank);loaded=load_payload(p)
        assert loaded.method==method and score(loaded,w)==score(bank,w)
