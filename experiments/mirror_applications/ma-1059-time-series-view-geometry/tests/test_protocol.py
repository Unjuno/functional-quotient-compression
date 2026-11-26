import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_registry_prior_art_and_split_frozen():
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    freeze = json.loads((ROOT / "FRESH_FREEZE.json").read_text())
    assert p["experiment_id"] == "MA-1059"
    assert p["prior_art_refs"] == ["PA328", "PA330", "PA335"]
    assert p["fresh"]["locked_before_access"] is True
    assert freeze["protocol_and_split_amended_before_audit_access"] is True
    assert freeze["audit_numeric_values_opened"] is False
    assert json.loads((ROOT / "PROTOCOL.json").read_text())["amendments"][0]["id"] == "AM-01"
    assert freeze["protocol_sha256"] == hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest()


def test_all_splits_disjoint_and_previewed_rows_excluded():
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    split = p["data"]["series_ids"]
    fit, dev, audit = map(set, (split["fit"], split["development"], split["audit"]))
    assert len(fit) == 40 and len(dev) == 16 and len(audit) == 24
    assert not (fit & dev or fit & audit or dev & audit)
    assert not (fit | dev | audit) & set(p["data"]["exposed_excluded"])
