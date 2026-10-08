import importlib.util
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "source" / "run.py"
spec = importlib.util.spec_from_file_location("ma325_run", SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_tt_embedding_and_tied_softmax_shapes():
    teacher, trans = m.teacher_world(32501)
    assert teacher.shape == (3, 16, 16)
    assert trans.shape == (3, 16, 16)
    assert torch.allclose(trans.sum(-1), torch.ones(3, 16), atol=1e-6)


def test_domain_phase_changes_shared_tt_embedding():
    model = m.TTBank("mirror_phase", 5)
    before = model.effective_table(0).detach().clone()
    with torch.no_grad():
        model.phases[0] = 1.2
    after = model.effective_table(0).detach()
    assert not torch.equal(before, after)
    assert model.effective_table(0).shape == (16, 16)
    assert model.effective_table(2).shape == (16, 16)


def test_direct_and_private_controls_cover_unrelated_domain():
    direct = m.TTBank("shared_direct", 7)
    private = m.TTBank("shared_private", 7)
    assert direct.effective_table(2).shape == (16, 16)
    assert private.effective_table(2).shape == (16, 16)
    assert not torch.equal(private.effective_table(2), private.effective_table(1))


def test_fp16_payload_reloads_inference_model(tmp_path):
    model = m.TTBank("mirror_phase", 9)
    p = tmp_path / "model.npz"
    n, digest, arrays = m.inference_payload(model, p)
    loaded = m.load_payload_model("mirror_phase", arrays, 9)
    assert n == p.stat().st_size and len(digest) == 64
    for key, value in loaded.state_dict().items():
        expected = torch.from_numpy(np.array(arrays[key], copy=True)).to(value.dtype)
        assert torch.equal(value, expected)
