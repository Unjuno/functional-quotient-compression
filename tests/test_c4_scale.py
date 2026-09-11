"""C4 scale-replication tests: 8M/1M structural validation + lane integrity.

Weight-dependent tests skip when local checkpoints are absent (not in git).
Weight-free tests (deterministic serialization) always run.
Existing 265 tests are untouched.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/t282/code"))

M8 = ROOT / "models/HF-8M"
M1 = ROOT / "models/HF-1M"
EXPECTED = {
    "8M": {"dir": M8, "hidden": 256, "paid": 19702528,
           "ckpt": "22c355bfabebc1f6c861b3f5d7a801e96c7f6da4af4bb0f7780096ab82ea6716"},
    "1M": {"dir": M1, "hidden": 64, "paid": 3745984,
           "ckpt": "07f9609ea882b8163ff3b23d40e2b82cb715d409631beb15c84b164f3877dae7"},
}
need = lambda tag: pytest.mark.skipif(
    not (EXPECTED[tag]["dir"] / "pytorch_model.bin").exists(),
    reason=f"{tag} weights absent")


def sha(p):
    return hashlib.file_digest(Path(p).open("rb"), "sha256").hexdigest()


@pytest.mark.parametrize("tag", ["8M", "1M"])
def test_scale_config_dimensions(tag):
    exp = EXPECTED[tag]
    if not (exp["dir"] / "config.json").exists():
        pytest.skip(f"{tag} config absent")
    cfg = json.loads((exp["dir"] / "config.json").read_text())
    assert cfg["hidden_size"] == exp["hidden"]
    assert cfg["num_layers"] == 8 and cfg["num_heads"] == 16
    assert cfg["vocab_size"] == 50257
    assert cfg["activation_function"] == "gelu_new"


@pytest.mark.parametrize("tag", ["8M", "1M"])
@need("8M")
@need("1M")
def test_scale_checkpoint_hash_and_tied(tag):
    import torch
    exp = EXPECTED[tag]
    assert sha(exp["dir"] / "pytorch_model.bin") == exp["ckpt"]
    raw = torch.load(exp["dir"] / "pytorch_model.bin", map_location="cpu", weights_only=True)
    assert "lm_head.weight" not in raw, "lane assumes tied embeddings"
    learned = {k: v for k, v in raw.items()
               if not k.endswith(".attn.attention.bias") and not k.endswith(".masked_bias")}
    assert len(learned) == 108
    assert sum(v.numel() for v in learned.values()) == exp["paid"]
    assert all(bool((v == v).all()) for v in learned.values())


def test_deterministic_serialization_synthetic():
    import numpy as np
    from binary_codec import encode_uniform, write_model
    rng = np.random.default_rng(7)
    a = (rng.standard_normal((64, 64)) * 3).astype(np.float32)
    d1, b1 = encode_uniform(a, 4, 64)
    d2, b2 = encode_uniform(a, 4, 64)
    assert b1 == b2 and d1 == d2


@pytest.mark.parametrize("tag", ["8M", "1M"])
@need("8M")
@need("1M")
def test_scale_tokenizer_roundtrip(tag):
    from engine import BPETokenizer
    tok = BPETokenizer(EXPECTED[tag]["dir"])
    s = "Once upon a time there was a little dragon."
    assert tok.decode(tok.encode(s)) == s


@pytest.mark.parametrize("tag", ["8M", "1M"])
@need("8M")
@need("1M")
def test_scale_tokenizer_mismatch_rejected(tag, tmp_path):
    import shutil
    from engine import BPETokenizer
    d = tmp_path / "tok"
    d.mkdir()
    for f in ("tokenizer.json", "vocab.json"):
        shutil.copy(EXPECTED[tag]["dir"] / f, d / f)
    vocab = json.loads((d / "vocab.json").read_text())
    vocab["tampered_token_xyz"] = 0
    (d / "vocab.json").write_text(json.dumps(vocab))
    with pytest.raises(AssertionError):
        BPETokenizer(d)


def _first_lane_artifact():
    cands = sorted(ROOT.glob("runs/c4-*-controls/*.fqc")) + \
        sorted(ROOT.glob("runs/c4-*-sharing/*.fqc"))
    return cands[0] if cands else None


def test_lane_artifact_corruption_rejected(tmp_path):
    from binary_codec import read_model
    art = _first_lane_artifact()
    if art is None:
        pytest.skip("no lane artifact built")
    bad = tmp_path / "bad.fqc"
    blob = bytearray(art.read_bytes())
    blob[100] ^= 1
    bad.write_bytes(blob)
    with pytest.raises(ValueError):
        read_model(bad)
    trunc = tmp_path / "trunc.fqc"
    trunc.write_bytes(bytes(blob[:len(blob) // 2]))
    with pytest.raises(ValueError):
        read_model(trunc)
