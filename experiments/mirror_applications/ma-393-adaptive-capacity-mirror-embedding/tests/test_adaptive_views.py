import importlib.util
from pathlib import Path
import zipfile
import numpy as np
import torch

SRC=Path(__file__).resolve().parents[1]/'source'/'run_experiment.py'
spec=importlib.util.spec_from_file_location('ma393',SRC);ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)


def test_zipf_samples_are_head_heavy_and_splits_are_reproducible():
    a,_=ma.make_world(39301);b,_=ma.make_world(39301)
    assert torch.equal(a['train'][0],b['train'][0])
    assert (a['train'][0]<ma.HEAD).float().mean()>.45


def test_mirror_rotation_preserves_tail_latent_reconstruction_norm():
    z=torch.randn(20,ma.D);theta=torch.randn(20)
    assert torch.allclose(ma.rotate(z,theta).norm(dim=-1),z.norm(dim=-1),atol=1e-5)


def test_all_embedding_controls_emit_32d_vectors():
    ids=torch.tensor([0,63,64,255,256,1023])
    for method in ('full','adaptive','uniform12','tail_basis','tail_scalar','mirror'):
        model=ma.AdaptiveTokenModel(method);assert model.token_embeddings(ids).shape==(len(ids),ma.D)


def test_mirror_payload_contains_shared_basis_and_per_tail_angles(tmp_path):
    model=ma.AdaptiveTokenModel('mirror');n,d=ma.archive(tmp_path/'m.zip',model)
    assert n==(tmp_path/'m.zip').stat().st_size and len(d)==64
    with zipfile.ZipFile(tmp_path/'m.zip') as z:
        assert 'arrays/basis.npy' in z.namelist() and 'arrays/angle.npy' in z.namelist()
        assert np.load(z.open('arrays/angle.npy')).shape==(ma.TAIL,)

