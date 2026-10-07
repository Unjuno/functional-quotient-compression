import io
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from model import Config, DomainUpdate, TinyCacheDecoder


def test_final_layer_view_preserves_every_prefix_kv_exactly():
    model = TinyCacheDecoder(seed=77)
    x = torch.randn(1, 8, 16)
    _, c0 = model(x, domain=0, placement="final_only")
    _, c1 = model(x, domain=1, placement="final_only")
    diffs = [float((a - b).abs().max()) for layer0, layer1 in zip(c0, c1) for a, b in zip(layer0, layer1)]
    assert max(diffs) == 0.0


def test_early_view_changes_downstream_kv_but_not_its_own_kv():
    model = TinyCacheDecoder(seed=78)
    x = torch.randn(1, 8, 16)
    _, c0 = model(x, domain=0, placement="early_and_final")
    _, c1 = model(x, domain=1, placement="early_and_final")
    early_delta = max(float((a - b).abs().max()) for a, b in zip(c0[0], c1[0]))
    downstream_delta = max(float((a - b).abs().max()) for a, b in zip(c0[1], c1[1]))
    assert early_delta == 0.0
    assert downstream_delta > 1e-7


def test_method_storage_and_shapes():
    cfg = Config()
    x = torch.randn(32, cfg.input_dim)
    domain = torch.arange(32) % cfg.domains
    base = torch.randn(cfg.input_dim, cfg.output_dim)
    for method in DomainUpdate.METHODS:
        model = DomainUpdate(cfg, method, base)
        assert model(x, domain).shape == (32, cfg.output_dim)
        assert model.inference_payload_bytes() > 0


def test_inference_state_dict_roundtrip_is_exact():
    cfg = Config()
    base = torch.randn(cfg.input_dim, cfg.output_dim)
    x = torch.randn(16, cfg.input_dim)
    d = torch.arange(16) % cfg.domains
    for method in DomainUpdate.METHODS:
        a, b = DomainUpdate(cfg, method, base), DomainUpdate(cfg, method, base)
        buf = io.BytesIO()
        torch.save(a.state_dict(), buf)
        b.load_state_dict(torch.load(io.BytesIO(buf.getvalue()), weights_only=True))
        assert torch.equal(a(x, d), b(x, d))
