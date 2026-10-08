import importlib.util
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "source" / "run.py"
spec = importlib.util.spec_from_file_location("ma330_run", SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_teacher_and_attention_shapes():
    data = m.world(33001)
    assert data[0].shape == (4, 128, 16)
    assert data[3].shape == (4, 4, 128, 16)
    assert m.reference_outputs(data[3], data[4], data[5]).shape == (4, 4, 128, 16)


def test_mirror_phase_reconstructs_aligned_cache():
    data = m.world(33002)
    arrays = {"time_basis": data[8].numpy().astype("<f2"), "k_coeff": data[9].numpy().astype("<f2"), "v_coeff": data[10].numpy().astype("<f2"),
              "phase": data[2][:3].numpy().astype("<f2"),
              "private_k": data[3][3].numpy().astype("<f2"), "private_v": data[4][3].numpy().astype("<f2")}
    k, v = m.unpack_cache("mirror_phase", arrays)
    assert torch.allclose(k[:3], data[3][:3], atol=1e-3, rtol=1e-3)
    assert torch.allclose(v[:3], data[4][:3], atol=1e-3, rtol=1e-3)
    assert torch.equal(k[3], torch.from_numpy(arrays["private_k"].astype("<f4")))


def test_direct_coeff_and_phase_define_same_rotation():
    data = m.world(33003)
    phase = data[2][:3]
    coeff = torch.stack((torch.cos(phase), torch.sin(phase)), -1).numpy().astype("<f2")
    common = {"time_basis": data[8].numpy().astype("<f2"), "k_coeff": data[9].numpy().astype("<f2"), "v_coeff": data[10].numpy().astype("<f2"),
              "private_k": data[3][3].numpy().astype("<f2"), "private_v": data[4][3].numpy().astype("<f2")}
    pa = {**common,
          "phase": phase.numpy().astype("<f2"), "private_k": data[3][3].numpy().astype("<f2"),
          "private_v": data[4][3].numpy().astype("<f2")}
    da = {**common, "rotation": coeff,
          "private_k": pa["private_k"], "private_v": pa["private_v"]}
    pk, pv = m.unpack_cache("mirror_phase", pa)
    dk, dv = m.unpack_cache("shared_direct", da)
    assert torch.allclose(pk, dk, atol=2e-3, rtol=2e-3)
    assert torch.allclose(pv, dv, atol=2e-3, rtol=2e-3)


def test_payload_serialization_reloads(tmp_path):
    data = m.world(33004)
    n, digest, _ = m.payload("mirror_phase", data, tmp_path / "mirror.zip")
    arrays = m.load_arrays(tmp_path / "mirror.zip")
    assert n == (tmp_path / "mirror.zip").stat().st_size
    assert len(digest) == 64
    assert "phase" in arrays and "private_k" in arrays
