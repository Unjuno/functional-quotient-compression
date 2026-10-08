import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from model import METHODS, SOURCE_TASKS, TARGET_TASKS, CandidateBank  # noqa: E402
from run import fit_fusion, make_data  # noqa: E402


def test_all_candidate_banks_produce_expected_shapes():
    x = torch.randn(6, 16)
    for method in METHODS:
        bank = CandidateBank(method, 1)
        assert bank.source_outputs(x).shape == (6, SOURCE_TASKS, 16)
        assert bank.target_outputs(x).shape == (6, TARGET_TASKS, 16)


def test_mirror_code_changes_only_selected_candidate():
    bank = CandidateBank("mirror_shared", 2)
    x = torch.randn(4, 16)
    before = bank.source_outputs(x).detach().clone()
    with torch.no_grad():
        bank.codes[3] = 0.31
    after = bank.source_outputs(x)
    assert not torch.allclose(before[:, 3], after[:, 3])
    assert torch.allclose(before[:, 2], after[:, 2])


def test_givens_conjugacy_preserves_singular_values():
    bank = CandidateBank("mirror_shared", 3)
    with torch.no_grad():
        bank.codes.copy_(torch.linspace(0.1, 0.8, SOURCE_TASKS))
    singular = torch.linalg.svdvals(bank.source_matrices())
    assert all(torch.allclose(singular[0], singular[i], atol=1e-6) for i in range(1, SOURCE_TASKS))


def test_few_shot_fusion_sets_target_coefficients():
    data, _ = make_data(9, "aligned")
    bank = CandidateBank("independent", 4)
    compute = fit_fusion(bank, data["support"])
    assert bank.fusion.shape == (TARGET_TASKS, SOURCE_TASKS)
    assert compute > 0
