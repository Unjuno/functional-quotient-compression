import sys
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))

from model import TASKS, ExpertPromptBank, prompt_reconstruction_mac_proxy  # noqa: E402


def test_expert_variants_share_paid_general_prompt_and_route_by_task_id():
    general = torch.randn(64)
    x = torch.randn(10, 8)
    ids = torch.arange(10) % TASKS
    for method in ExpertPromptBank.METHODS:
        bank = ExpertPromptBank(method, 47, general)
        assert torch.equal(bank.general, general)
        all_tasks = bank.forward_all(x)
        selected = bank.forward_task(x, ids)
        want = all_tasks[torch.arange(10), ids]
        assert torch.allclose(selected, want, atol=1e-6, rtol=0)


def test_givens_expert_coordinate_changes_task_function():
    bank = ExpertPromptBank("mirror", 13, torch.zeros(64))
    x = torch.randn(8, 8)
    before = bank.forward_task(x, torch.zeros(8, dtype=torch.long))
    with torch.no_grad():
        bank.angle[0, 0] = 1.1
    after = bank.forward_task(x, torch.zeros(8, dtype=torch.long))
    assert not torch.equal(before, after)


def test_prompt_reconstruction_cost_is_charged_for_shared_codes():
    assert prompt_reconstruction_mac_proxy("independent") == 0
    assert prompt_reconstruction_mac_proxy("mirror") == TASKS * 64 * 2
