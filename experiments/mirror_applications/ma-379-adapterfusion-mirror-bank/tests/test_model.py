import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from model import METHODS, SOURCE_TASKS, TARGET_TASKS, AdapterBank  # noqa: E402


def test_all_adapter_banks_emit_source_and_target_tasks():
    x = torch.randn(5, 16)
    for method in METHODS:
        model = AdapterBank(method, 1)
        assert model.source_outputs(x).shape == (5, SOURCE_TASKS, 16)
        assert model.target_outputs(x).shape == (5, TARGET_TASKS, 16)


def test_mirror_source_code_changes_task_function():
    model = AdapterBank("mirror_shared", 2)
    x = torch.randn(4, 16)
    before = model.source_outputs(x).detach().clone()
    with torch.no_grad():
        model.source_codes[1] = 0.4
    after = model.source_outputs(x)
    assert not torch.allclose(before[:, 1], after[:, 1])
    assert torch.allclose(before[:, 0], after[:, 0])


def test_conjugation_preserves_singular_values():
    model = AdapterBank("mirror_shared", 3)
    matrices = model.source_matrices()
    singular = torch.linalg.svdvals(matrices)
    assert torch.allclose(singular[0], singular[1], atol=1e-6)


def test_equal_code_dimensions_for_scalar_and_mirror():
    scalar = AdapterBank("scalar_shared", 4)
    mirror = AdapterBank("mirror_shared", 4)
    assert scalar.source_codes.numel() == mirror.source_codes.numel() == SOURCE_TASKS
    assert scalar.fusion.shape == mirror.fusion.shape == (TARGET_TASKS, SOURCE_TASKS)
