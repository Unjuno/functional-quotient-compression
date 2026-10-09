import sys
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))

from model import PROMPTS, PromptBank, nearest_key, select_prompt_outputs  # noqa: E402


def test_prompt_bank_variants_have_expected_shape_and_gradients():
    x = torch.randn(9, 8)
    for method in PromptBank.METHODS:
        bank = PromptBank(method, 41)
        y = bank(x)
        assert y.shape == (9, PROMPTS, 8)
        y.square().mean().backward()
        assert all(p.grad is not None for p in bank.parameters())


def test_retrieved_forward_matches_gather_from_full_prompt_outputs():
    bank = PromptBank("mirror", 7)
    x = torch.randn(10, 8)
    ids = torch.arange(10) % PROMPTS
    full = bank(x)
    selected = bank.forward_selected(x, ids)
    assert torch.allclose(selected, select_prompt_outputs(full, ids), atol=1e-7, rtol=0)


def test_mirror_coordinate_changes_prompt_function_and_key_retrieval():
    bank = PromptBank("mirror", 11)
    x = torch.randn(8, 8)
    before = bank(x)
    with torch.no_grad():
        bank.angle[0, 0] = 1.2
    after = bank(x)
    assert not torch.equal(before[:, 0], after[:, 0])
    keys = torch.eye(PROMPTS)
    assert torch.equal(nearest_key(keys, keys), torch.arange(PROMPTS))
