import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from protocol_check import validate


def test_registered_candidate_and_prior_art_are_locked():
    p = validate()
    assert p["experiment_id"] == "MA-1059"
    assert p["prior_art_refs"] == ["PA328", "PA330", "PA335"]
    assert "shared pretrained temporal" in p["hypothesis"]
    assert any("Mirror" in c for c in p["controls"])


def test_audit_split_is_not_claimed_frozen_prematurely():
    p = validate()
    assert "fresh" not in p or p["fresh"].get("locked_before_access") is False
