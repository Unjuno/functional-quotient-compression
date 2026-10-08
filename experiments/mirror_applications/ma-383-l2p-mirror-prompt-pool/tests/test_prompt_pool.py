import importlib.util
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "source" / "run_experiment.py"
spec = importlib.util.spec_from_file_location("ma383", SRC)
ma = importlib.util.module_from_spec(spec); spec.loader.exec_module(ma)


def test_givens_rotation_is_norm_preserving_and_identity_at_zero():
    x = torch.randn(32)
    assert torch.allclose(ma.rotate(x, 0), x)
    assert torch.allclose(ma.rotate(x, .73).norm(), x.norm(), atol=1e-6)


def test_retrieval_uses_query_and_keys_not_task_labels():
    x, _, task, keys, _, _ = ma.dataset(38301, "aligned", "test")
    predicted = ma.choose_keys(x, keys)
    assert (predicted == task).float().mean() >= .95


def test_aligned_prompts_follow_one_shared_givens_orbit():
    *_, prompts = ma.dataset(38301, "aligned", "train")
    norms = prompts.norm(dim=1)
    assert torch.allclose(norms, norms[0].expand_as(norms), atol=1e-5)


def test_serialized_payload_accounts_for_keys_classifier_and_method_state(tmp_path):
    model = ma.Pool("mirror")
    keys = torch.randn(8, 32); w = torch.randn(32, 4)
    n, digest = ma.archive(tmp_path / "payload.zip", "mirror", model, keys, w)
    assert n == (tmp_path / "payload.zip").stat().st_size
    assert len(digest) == 64
    import zipfile
    with zipfile.ZipFile(tmp_path / "payload.zip") as z:
        names = z.namelist()
    assert "arrays/keys.npy" in names and "arrays/classifier.npy" in names

