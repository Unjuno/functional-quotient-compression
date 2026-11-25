"""Sanity checks for the frozen MA-1059 protocol before source/data access."""
import json
from pathlib import Path

PROTOCOL = Path(__file__).resolve().parents[1] / "PROTOCOL.json"


def validate(path=PROTOCOL):
    p = json.loads(Path(path).read_text())
    required = {"experiment_id", "prior_art_refs", "hypothesis", "controls", "success_gate", "failure_gate"}
    missing = required - p.keys()
    if missing:
        raise ValueError(f"missing protocol fields: {sorted(missing)}")
    if p["experiment_id"] != "MA-1059" or p["prior_art_refs"] != ["PA328", "PA330", "PA335"]:
        raise ValueError("MA ID/prior-art mismatch")
    if not any("Mirror" in c for c in p["controls"]):
        raise ValueError("Mirror control is missing")
    return p


if __name__ == "__main__":
    validate()
    print("MA-1059 protocol structure valid; empirical protocol/data split remain unfrozen")
