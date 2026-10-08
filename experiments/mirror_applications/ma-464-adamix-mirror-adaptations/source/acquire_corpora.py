"""Acquire two public-domain character corpora after the preregistration freeze."""
from __future__ import annotations
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source" / "data"
SOURCES = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "austen": "https://www.gutenberg.org/files/1342/1342-0.txt",
}


def acquire_one(name: str, url: str) -> dict:
    path = DATA / f"{name}.txt"
    DATA.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "MA-464 research worker"})
        with urllib.request.urlopen(request, timeout=90) as response, path.open("wb") as target:
            target.write(response.read())
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {"name": name, "url": url, "bytes": path.stat().st_size, "sha256": digest,
            "path": str(path.relative_to(ROOT))}


def acquire_all() -> list[dict]:
    return [acquire_one(name, url) for name, url in SOURCES.items()]
