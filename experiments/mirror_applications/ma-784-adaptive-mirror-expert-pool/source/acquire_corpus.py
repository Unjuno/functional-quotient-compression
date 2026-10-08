"""Download the public-domain development corpus only after protocol freeze."""
from __future__ import annotations
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source" / "data" / "input.txt"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def acquire() -> dict:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": "MA-784 research worker"})
        with urllib.request.urlopen(request, timeout=60) as response, DATA.open("wb") as target:
            target.write(response.read())
    with DATA.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {"path": str(DATA.relative_to(ROOT)), "url": URL, "sha256": digest, "bytes": DATA.stat().st_size}
