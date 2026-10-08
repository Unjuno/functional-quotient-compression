import sys
from pathlib import Path
import torch

SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))
from model import METHODS, SignedExperts, signed_coefficients, givens


def test_signed_codes_include_positive_and_negative_weights():
    x = torch.tensor([[1., 1.] + [0.] * 14, [-1., 1.] + [0.] * 14,
                      [1., -1.] + [0.] * 14, [-1., -1.] + [0.] * 14])
    c = signed_coefficients(x)
    assert c.shape == (4, 4)
    assert set(c.flatten().tolist()) == {-0.5, 0.5}
    assert torch.unique(c, dim=0).shape[0] == 4


def test_givens_preserves_norm():
    x = torch.randn(8, 16)
    angles = torch.randn(8)
    y = givens(x, angles)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=2e-6)


def test_all_models_return_finite_expected_shape_and_serialized_size():
    x = torch.randn(5, 16)
    for method in METHODS:
        m = SignedExperts(method, 7)
        y = m(x)
        assert y.shape == (5, 12)
        assert torch.isfinite(y).all()
        assert m.serialized_payload_bytes() > 0


def test_mirror_view_can_represent_signed_givens_mixture():
    torch.manual_seed(4)
    x = torch.randn(32, 16)
    angles = torch.randn(4, 8) * .2
    w = torch.randn(16, 12)
    views = torch.stack([givens(x, a.expand(x.shape[0], -1)) for a in angles], dim=1)
    expected = torch.einsum("bn,bno->bo", signed_coefficients(x), torch.einsum("bnd,do->bno", views, w))
    model = SignedExperts("mirror")
    with torch.no_grad():
        model.weight.copy_(w)
        model.angles.copy_(angles)
    assert torch.allclose(model(x), expected, atol=1e-6)
