import importlib.util
from pathlib import Path
import zipfile
import numpy as np
import torch

SRC=Path(__file__).resolve().parents[1]/'source'/'run_experiment.py'
spec=importlib.util.spec_from_file_location('ma397',SRC);ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)


def test_each_product_pair_collides_across_four_distinct_labels():
    ids=torch.arange(ma.V);q,r=ma.address(ids);keys=q*ma.N2+r
    counts=torch.bincount(keys,minlength=ma.N1*ma.N2)
    assert torch.all(counts==4)
    for k in range(ma.N1*ma.N2):
        assert torch.unique(ma.labels(ids[keys==k])).numel()==4


def test_product_subaddress_rule_is_deterministic_and_bounded():
    ids=torch.arange(ma.V);q,r=ma.address(ids)
    assert q.min()==0 and q.max()<ma.N1 and r.max()<ma.N2
    assert torch.equal(ma.address(ids)[0],ma.address(ids)[0])


def test_all_collision_controls_emit_embedding_and_logits():
    for method in ('full','product','scalar','hash','mirror'):
        m=ma.ProductTokenModel(method)
        assert m.token_embeddings(torch.arange(4)).shape==(4,ma.D)
        assert m(torch.arange(4)).shape==(4,ma.C)


def test_mirror_payload_charges_per_token_angle(tmp_path):
    m=ma.ProductTokenModel('mirror');n,d=ma.archive(tmp_path/'m.zip',m)
    assert n==(tmp_path/'m.zip').stat().st_size and len(d)==64
    with zipfile.ZipFile(tmp_path/'m.zip') as z:
        assert 'arrays/angle.npy' in z.namelist()
        assert np.load(z.open('arrays/angle.npy')).shape==(ma.V,)
        assert b'tokens_per_address' in z.read('metadata.json')

