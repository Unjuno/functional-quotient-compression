"""Acquire Tiny Shakespeare and materialize only train/development token spans."""
from __future__ import annotations
import hashlib, json, urllib.request
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source" / "data"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def acquire() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    raw_path = DATA / "input.txt"
    if not raw_path.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": "MA-783-research/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw_path.write_bytes(response.read())
    raw = raw_path.read_bytes()
    n = len(raw)
    train_end = int(0.8 * n)
    dev_end = int(0.9 * n)
    # The fixed boundaries are byte offsets. Decode errors are replaced only at a
    # boundary or for malformed source bytes; audit bytes are not decoded here.
    train_text = raw[:train_end].decode("utf-8", errors="replace")
    dev_text = raw[train_end:dev_end].decode("utf-8", errors="replace")
    vocab = sorted(set(train_text))
    if "\ufffd" not in vocab:
        vocab.append("\ufffd")
    stoi = {ch: i for i, ch in enumerate(vocab)}
    unk = stoi["\ufffd"]
    train = torch.tensor([stoi.get(ch, unk) for ch in train_text], dtype=torch.int64)
    dev = torch.tensor([stoi.get(ch, unk) for ch in dev_text], dtype=torch.int64)
    torch.save(train, DATA / "train_tokens.pt")
    torch.save(dev, DATA / "development_tokens.pt")
    manifest = {
        "source_url": URL,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": n,
        "train_byte_span": [0, train_end],
        "development_byte_span": [train_end, dev_end],
        "audit_byte_span_locked": [dev_end, n],
        "train_characters": len(train_text),
        "development_characters": len(dev_text),
        "training_vocabulary_size": len(vocab),
        "training_vocabulary": vocab,
        "unknown_token_id": unk,
        "audit_bytes_decoded_or_tokenized": False,
        "split_rule": "fixed contiguous byte spans at 80% and 90%; no windows cross boundaries",
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(acquire(), indent=2))
