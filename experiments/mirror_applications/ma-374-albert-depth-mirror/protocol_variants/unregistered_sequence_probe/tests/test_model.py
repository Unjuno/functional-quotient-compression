import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from model import DEPTH, METHODS, TinyAlbertLM  # noqa: E402


def test_all_methods_emit_prefix_logits():
    tokens = torch.randint(0, 8, (2, 7))
    for method in METHODS:
        model = TinyAlbertLM(method)
        prefixes = model.forward_prefixes(tokens)
        assert len(prefixes) == DEPTH
        assert all(logits.shape == (2, 7, 8) for logits in prefixes)
        assert all(torch.equal(model.forward_to_depth(tokens, d), prefixes[d - 1]) for d in range(1, DEPTH + 1))


def test_albert_reuses_block_object():
    model = TinyAlbertLM("albert_tied")
    assert model.shared is model.shared
    assert not hasattr(model, "layers")


def test_depth_mirror_codes_change_layer_function():
    model = TinyAlbertLM("tied_mirror")
    tokens = torch.randint(0, 8, (2, 8))
    before = model.forward_prefixes(tokens)[1].detach().clone()
    with torch.no_grad():
        model.depth_codes[1] = 0.2
    after = model.forward_prefixes(tokens)[1]
    assert not torch.allclose(before, after)


def test_attention_and_ffn_only_sharing_have_separate_components():
    attention_shared = TinyAlbertLM("attention_shared")
    ffn_shared = TinyAlbertLM("ffn_shared")
    assert len(attention_shared.layer_ffn) == DEPTH
    assert len(ffn_shared.layer_attention) == DEPTH
    assert attention_shared.shared_attention is attention_shared.shared_attention
    assert ffn_shared.shared_ffn is ffn_shared.shared_ffn
