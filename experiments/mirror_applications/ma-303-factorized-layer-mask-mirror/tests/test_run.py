import importlib.util
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "source" / "run.py"
spec = importlib.util.spec_from_file_location("ma303_run", SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_world_shapes_and_planted_masks():
    w = m.make_world(30301)
    masks = m.teacher_masks(w)
    assert masks.shape == (8, 8, 2, 64, 64)
    assert np.all(masks.sum(axis=(-1, -2)) == 1024)


def test_factorized_controls_reconstruct_heldout_pair():
    w = m.make_world(30301)
    masks = m.teacher_masks(w)
    for method in ("direct_factorized", "mirror_phase"):
        state = m.stored_state(method, w, masks)
        state, private = m.add_validation_fallbacks(w, method, state, masks, 30301)
        assert private == 0
        assert np.array_equal(m.pair_masks(w, 7, 7, method, state), masks[7, 7])


def test_validation_selects_private_fallback_for_bad_codes():
    w = m.make_world(30301)
    masks = m.teacher_masks(w)
    state = m.stored_state("mirror_phase", w, masks)
    state["task_angles"][7] = np.float16(1.2)
    state, _ = m.add_validation_fallbacks(w, "mirror_phase", state, masks, 30301)
    assert state["private_ids"].shape[1] == 2
    assert np.any(state["private_ids"][:, 0].astype(int) == 7)


def test_payload_roundtrip_is_deterministic(tmp_path):
    w = m.make_world(30301)
    masks = m.teacher_masks(w)
    state = m.stored_state("independent_masks", w, masks)
    arrays = {"masks": np.packbits(state["masks"].reshape(-1)), "shape": np.asarray(state["masks"].shape, dtype=np.uint8)}
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    m.pack(arrays, p1)
    m.pack(arrays, p2)
    assert p1.read_bytes() == p2.read_bytes()
    got = m.unpack(p1)
    shape = tuple(int(x) for x in got["shape"])
    restored = np.unpackbits(got["masks"])[:int(np.prod(shape))].reshape(shape).astype(bool)
    assert np.array_equal(restored, masks)
