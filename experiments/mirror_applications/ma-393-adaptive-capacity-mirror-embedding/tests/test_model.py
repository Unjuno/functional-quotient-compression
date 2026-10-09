import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
from model import BAND_SIZES,BAND_WIDTHS,BAND_MASSES,METHODS,AdaptiveBank,projection,rotate_first_pair  # noqa:E402
from run import make_world,save_payload,load_payload,score  # noqa:E402


def test_frequency_bands_match_registered_counts_and_mass():
    assert sum(BAND_SIZES)==384 and len(BAND_WIDTHS)==3
    assert abs(sum(BAND_MASSES)-1)<1e-8
    assert sum(n*k for n,k in zip(BAND_SIZES,BAND_WIDTHS))==2816


def test_givens_coordinate_preserves_embedding_norm():
    x=torch.randn(384,16);a=torch.rand(384)*6.28
    torch.testing.assert_close(rotate_first_pair(x,a).norm(dim=1),x.norm(dim=1),atol=2e-6,rtol=2e-6)


def test_projection_seed_reconstructs_exactly():
    torch.testing.assert_close(projection(12,1,8),projection(12,1,8),atol=0,rtol=0)
    assert projection(12,1,8).shape==(16,8)


def test_every_method_payload_replays_all_weighted_and_band_metrics(tmp_path):
    world=make_world(39301)
    for method in METHODS:
        bank=AdaptiveBank(method,39302,world['basis_seed'],world['decoder'])
        path=tmp_path/f'{method}.npz';save_payload(path,bank)
        loaded=load_payload(path)
        assert loaded.method==method
        assert score(loaded,world)==score(bank,world)
