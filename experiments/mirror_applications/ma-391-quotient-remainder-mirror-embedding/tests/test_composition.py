import importlib.util
from pathlib import Path
import zipfile
import numpy as np
import torch

SRC=Path(__file__).resolve().parents[1]/'source'/'run_experiment.py'
spec=importlib.util.spec_from_file_location('ma391',SRC);ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)


def test_quotient_remainder_address_is_unique_and_complete():
    ids=torch.arange(ma.V);pairs=torch.stack([ids//ma.RROWS,ids%ma.RROWS],-1)
    assert torch.unique(pairs,dim=0).shape[0]==ma.V


def test_target_families_have_expected_shapes_and_labels():
    ids=torch.arange(ma.V)
    assert ma.target(ids,'separable').shape==(ma.V,)
    assert ma.target(ids,'xor').max()<ma.CLASSES


def test_all_composition_controls_decode_full_vocabulary():
    for method in ('full','add','multiply','concat','mirror'):
        model=ma.Composition(method)
        assert model.embedding_rows().shape==(ma.V,ma.D)
        assert model(torch.arange(4)).shape==(4,ma.CLASSES)


def test_mirror_payload_serializes_angles_and_metadata(tmp_path):
    model=ma.Composition('mirror');n,d=ma.archive(tmp_path/'m.zip',model)
    assert n==(tmp_path/'m.zip').stat().st_size and len(d)==64
    with zipfile.ZipFile(tmp_path/'m.zip') as z:
        assert 'arrays/angle.npy' in z.namelist() and 'arrays/q.npy' in z.namelist() and 'arrays/r.npy' in z.namelist()
        assert b'q=id//32' in z.read('metadata.json')
        assert np.load(z.open('arrays/angle.npy')).shape==(ma.V,)

