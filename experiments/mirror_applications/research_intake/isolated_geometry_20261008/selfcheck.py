#!/usr/bin/env python3
"""Read-only verification of quarantined Mirror plans; stdlib only, no network."""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent.parent  # repo root
APPS = ROOT / "experiments" / "mirror_applications"
EXACT_NEW = range(1116, 1124)
EXACT_PRIOR = range(382, 391)
REQUIRED = ["id", "family", "proposal", "shared_object", "mirror_coordinate",
            "logical_multiplicity", "cheapest_control", "first_metric",
            "priority", "status", "prior_art_level", "prior_art_refs",
            "mirror_delta", "worker_note"]


def validate():
    errors = []
    with (APPS / "IDEA_REGISTRY.csv").open(encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != REQUIRED:
            errors.append("incorrect registry schema")
        rows = list(rd)
    if len(rows) != 1123:
        errors.append(f"expected staged 1123 rows, got {len(rows)}; reconcile rather than silently renumber")
    idlist = [row.get("id", "") for row in rows]
    expected = [f"MA-{i:03d}" for i in range(1, 1124)]
    if idlist != expected:
        errors.append("MA IDs are not a gap-free ordered sequence 001..1123")
    source = (ROOT / "docs" / "phase2" / "MIRROR_APPLICATION_PRIOR_ART.md").read_text(encoding="utf-8")
    headings = [int(x) for x in re.findall(r"^## PA([0-9]+)\b", source, re.M)]
    if headings != list(range(1, 391)):
        errors.append("PA headings not sequential 1..390")
    refs = {int(n) for row in rows for n in re.findall(r"PA([0-9]+)", row.get("prior_art_refs", ""))}
    if not refs.issubset(set(headings)):
        errors.append("orphan PA references")
    byid = {row["id"]: row for row in rows}
    for i in EXACT_NEW:
        id_ = f"MA-{i}"
        row = byid.get(id_, {})
        if row.get("status") != "UNTESTED":
            errors.append(f"{id_} claimed without experiment")
        if "ISOLATED" not in row.get("worker_note", ""):
            errors.append(f"{id_} not marked quarantined")
        folder = HERE / "plans" / id_
        for name in ["README.md", "PROTOCOL.json", "STATUS.md"]:
            if not (folder / name).is_file():
                errors.append(f"missing {id_}/{name}")
        if not (folder / "PROTOCOL.json").is_file():
            continue
        try:
            p = json.loads((folder / "PROTOCOL.json").read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            errors.append(f"{id_} invalid protocol: {e}")
            continue
        if p.get("experiment_id") != id_ or p.get("status") != "UNTESTED":
            errors.append(f"{id_} wrong JSON experiment id or status")
        if not (p.get("plan_only") and not p.get("worker_claim")):
            errors.append(f"{id_} unexpectedly marked active")
        a = set(p.get("split", {}).get("development", {}).get("seeds", []))
        b = set(p.get("split", {}).get("fresh", {}).get("seeds", []))
        if a != {11, 12, 13} or b != {101, 102, 103, 104, 105} or a & b:
            errors.append(f"{id_} dev/fresh seeds missing or overlapping")
        if not p.get("split", {}).get("fresh", {}).get("locked_before_access"):
            errors.append(f"{id_} not fresh-locked")
        if not set(p.get("gates", {})).issuperset({"PASS", "FAIL", "UNCERTAIN"}):
            errors.append(f"{id_} missing decision gates")
        prose = (folder / "README.md").read_text(encoding="utf-8")
        for required in ("## H", "## T", "## D", "## C", "## U", "Variable table",
                         "Mirror insertion", "Minimal worker-free runbook"):
            if required not in prose:
                errors.append(f"{id_} missing {required}")
        if list(folder.glob("RESULTS_CORE.csv")) or list(folder.glob("VERIFICATION.json")):
            errors.append(f"{id_} premature result/verification file")
    for existing in ("MA-338", "MA-692", "MA-1115"):
        if not (HERE / "existing" / f"{existing}.md").is_file():
            errors.append(f"missing existing ID supplement {existing}")
    board = (APPS / "STATUS_BOARD.md").read_text(encoding="utf-8")
    counts = Counter(row["status"] for row in rows)
    priorities = Counter(row["priority"] for row in rows)
    for p, n in priorities.items():
        if f"{p}: **{n}**" not in board:
            errors.append(f"status board missing priority {p}: {n}")
    for s, n in counts.items():
        if f"{n} {s}" not in board:
            errors.append(f"status board missing {s}: {n}")
    if "Registered candidates: **1123**" not in board:
        errors.append("status board candidate count not 1123")
    manifest = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))
    if not manifest.get("quarantined") or manifest.get("worker_queue_change"):
        errors.append("manifest unexpectedly allows worker queue change")
    print(f"Staged registry={len(rows)}; prior={len(headings)}; plans={len(EXACT_NEW)}; "
          f"existing_supplements=3; errors={len(errors)}")
    for error in errors:
        print("ERROR:", error, file=sys.stderr)
    return not errors


if __name__ == "__main__":
    sys.exit(0 if validate() else 1)
