"""Gate-2 assembly/integration tests for the official-validation control lane.

Fast header-level checks run without weights; decoder/eval checks skip when
the local model or built artifacts are absent (they are not in git).
"""
import json
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "runs/gate2-controls"
MODEL = ROOT / "models/HF-28M"

needs_model = pytest.mark.skipif(not (MODEL / "pytorch_model.bin").exists(),
                                 reason="local 28M weights absent")
needs_controls = pytest.mark.skipif(not (CTRL / "GATE2_CONTROLS.json").exists(),
                                    reason="gate-2 controls not built")


def read_header(path):
    blob = Path(path).read_bytes()
    assert blob[:8] == b"FQCWR001"
    magic, nh, nb, pad = struct.unpack_from("<8sIQQ", blob)
    assert 28 + nh + nb + pad + 32 == len(blob)
    assert blob[-32:] == __import__("hashlib").sha256(blob[:-32]).digest()
    return json.loads(blob[28:28 + nh])


@needs_controls
def test_manifest_byte_counts_match_files():
    man = json.loads((CTRL / "GATE2_CONTROLS.json").read_text())
    assert len(man) >= 20
    for m in man:
        p = CTRL / (m["candidate"] + ".fqc")
        assert p.stat().st_size == m["bytes"], m["candidate"]
        assert m["bytes"] > 0 and m["ratio_vs_16bit"] > 1


@needs_controls
def test_vq_codebook_before_use_and_consistent():
    man = json.loads((CTRL / "GATE2_CONTROLS.json").read_text())
    vq = [m for m in man if m["candidate"].startswith("28M_vq")]
    assert vq, "no VQ controls in manifest"
    for m in vq:
        h = read_header(CTRL / (m["candidate"] + ".fqc"))
        books = {}
        seen = set()
        off = 0
        for d in h["sections"]:
            assert d["offset"] == off, "section coverage gap"
            off += d["length"]
            if d["kind"] == "codebook":
                books[d["name"]] = d
            elif d["kind"] == "vq":
                assert d["codebook"] in books, "codebook used before definition"
                cb = books[d["codebook"]]
                assert cb["K"] == d["K"] and cb["block"] == d["block"]
                assert (1 << d["index_bits"]) == d["K"]
            assert d["name"] not in seen or d["kind"] == "codebook", "duplicate"
            seen.add(d["name"])
        assert h["learned_tensors"] == 108


@needs_controls
def test_uniform_rate_equality_rtn_vs_act():
    man = {m["candidate"]: m for m in
           json.loads((CTRL / "GATE2_CONTROLS.json").read_text())}
    for b, g in [(8, 64), (6, 64), (4, 64), (4, 128), (3, 64), (2, 64), (2, 256)]:
        r = man[f"28M_rtn_b{b}_g{g}"]["bytes"]
        a = man[f"28M_act_b{b}_g{g}"]["bytes"]
        assert r == a, f"refit changed rate at b{b} g{g}"


@needs_model
@needs_controls
def test_independent_decoder_agrees_on_smallest_vq():
    import sys
    sys.path.insert(0, str(ROOT / "experiments/t282/code"))
    from binary_codec import read_model
    from independent_decoder import decode_model
    from engine import tensor_hash
    man = json.loads((CTRL / "GATE2_CONTROLS.json").read_text())
    smallest = min((m for m in man if m["candidate"].startswith("28M_vq")),
                   key=lambda m: m["bytes"])
    art = CTRL / (smallest["candidate"] + ".fqc")
    _, s1, _ = read_model(art)
    _, s2, _ = decode_model(art)
    assert set(s1) == set(s2) == set(s1) and len(s1) == 108
    assert tensor_hash(s1) == tensor_hash(s2)


@needs_model
def test_eval_self_consistency_two_stories():
    import sys
    sys.path.insert(0, str(ROOT / "experiments/gate1"))
    sys.path.insert(0, str(ROOT / "experiments/t282/code"))
    import torch
    from engine import BPETokenizer, forward, load_checkpoint
    torch.set_num_threads(2)
    cfg, state, _ = load_checkpoint(str(MODEL), "cpu")
    tok = BPETokenizer(str(MODEL))
    raw = (ROOT / "data/official/TinyStories-valid.txt").read_text()
    stories = [s.strip() for s in raw.split("<|endoftext|>") if s.strip()][:2]
    assert len(stories) == 2
    with torch.inference_mode():
        for text in stories:
            ids = torch.tensor([tok.encode(text)])
            z = forward(state, cfg, ids)
            assert torch.isfinite(z).all()
