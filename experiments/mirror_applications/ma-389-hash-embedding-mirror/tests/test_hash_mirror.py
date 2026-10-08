import importlib.util
from pathlib import Path
import zipfile
import numpy as np
import torch

SRC=Path(__file__).resolve().parents[1]/'source'/'run_experiment.py'
spec=importlib.util.spec_from_file_location('ma389',SRC);ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)


def test_world_hashes_are_fixed_and_targets_are_balanced():
    _,h,y=ma.make_world(38901)
    assert h.shape==(ma.VOCAB,2) and h.max()<ma.POOL
    assert torch.bincount(y).tolist()==[ma.VOCAB//ma.CLASSES]*ma.CLASSES


def test_mirror_importance_is_a_single_angle_over_same_hash_pair():
    _,h,_=ma.make_world(38901);m=ma.TokenModel('mirror',h)
    ids=torch.tensor([1,2,3]);out=m.token_embeddings(ids)
    assert out.shape==(3,ma.DIM)
    assert m.angle.numel()==ma.VOCAB
    assert m.hash_indices.equal(h.to(torch.uint8))


def test_all_registered_methods_return_embedding_vectors():
    _,h,_=ma.make_world(38902)
    for name in ('full','hash','unweighted','scalar','mirror'):
        model=ma.TokenModel(name,h)
        assert model(torch.tensor([0,1,2])).shape==(3,ma.CLASSES)


def test_payload_charges_hash_indices_and_reloads_exact_archive_size(tmp_path):
    _,h,_=ma.make_world(38901);m=ma.TokenModel('mirror',h)
    size,digest=ma.archive(tmp_path/'m.zip',m)
    assert size==(tmp_path/'m.zip').stat().st_size and len(digest)==64
    with zipfile.ZipFile(tmp_path/'m.zip') as z:
        assert 'arrays/hash_indices.npy' in z.namelist()
        assert np.load(z.open('arrays/hash_indices.npy')).dtype==np.uint8

