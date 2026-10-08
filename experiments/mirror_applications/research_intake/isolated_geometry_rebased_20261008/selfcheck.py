#!/usr/bin/env python3
"""Read-only isolated registry, source and protocol preflight (stdlib only).
Do not modify the live worker queue, registry or claims.
From a full repo checkout: python experiments/mirror_applications/research_intake/isolated_geometry_rebased_20261008/selfcheck.py
"""
from __future__ import annotations
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
APPS = ROOT / "experiments" / "mirror_applications"
EXPECTED_MA = 1170
EXPECTED_PA = 433
ISOLATED = range(1156, 1171)
REQUIRED = ("id","family","proposal","shared_object","mirror_coordinate",
            "logical_multiplicity","cheapest_control","first_metric",
            "priority","status","prior_art_level","prior_art_refs",
            "mirror_delta","worker_note")


def check():
    errors = []
    def error(msg):
        errors.append(msg)

    with (APPS / "IDEA_REGISTRY.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REQUIRED:
            error("incorrect 14-column header")
        rows = list(reader)
    byid = {r.get("id"): r for r in rows}
    expected = ["MA-%03d" % i for i in range(1, EXPECTED_MA + 1)]
    if len(rows) != EXPECTED_MA or [r["id"] for r in rows] != expected:
        error("MA IDs are not gap-free 001..%d" % EXPECTED_MA)
    if len(byid) != len(rows) or any(None in r or None in r.values() for r in rows):
        error("duplicate IDs or malformed CSV row widths")
    txt = (ROOT / "docs" / "phase2" / "MIRROR_APPLICATION_PRIOR_ART.md").read_text(encoding="utf-8")
    headings = [int(m.group(1)) for m in re.finditer(r"^## PA([0-9]+)\b", txt, re.M)]
    if headings != list(range(1, EXPECTED_PA + 1)):
        error("PA heading sequence not 1..%d" % EXPECTED_PA)
    for row in rows:
        for n in re.findall(r"PA([0-9]+)", row.get("prior_art_refs", "")):
            if int(n) not in headings:
                error("unresolved prior %s for %s" % (n, row["id"]))
    status = Counter(r["status"] for r in rows)
    priority = Counter(r["priority"] for r in rows)
    board = (APPS / "STATUS_BOARD.md").read_text(encoding="utf-8")
    if "Registered candidates: **%d**" % EXPECTED_MA not in board:
        error("board count disagrees with registry")
    for k, n in status.items():
        if "%d %s" % (n,k) not in board:
            error("board status mismatch %s" % k)
    for k,n in priority.items():
        if "%s: **%d**" % (k,n) not in board:
            error("board priority mismatch %s" % k)
    manifest = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))
    if not manifest.get("quarantined") or manifest.get("worker_queue_change"):
        error("isolation/worker queue invariant broken")
    ids = [x.get("id") for x in manifest.get("experiments", [])]
    if ids != ["MA-%d" % i for i in ISOLATED]:
        error("manifest MA mapping incomplete or has collision")
    if manifest.get("existing") != ["MA-338", "MA-692", "MA-1115"]:
        error("existing old-index collision not corrected")
    for num in ISOLATED:
        label = "MA-%d" % num
        row = byid.get(label, {})
        if row.get("status") != "UNTESTED" or "ISOLATED" not in row.get("worker_note", ""):
            error("%s not isolated/UNTESTED" % label)
        folder = HERE / "plans" / label
        for file in ("README.md", "PROTOCOL.json", "STATUS.md"):
            if not (folder / file).is_file():
                error("missing plan file: %s/%s" % (label,file))
        if not (folder / "PROTOCOL.json").is_file():
            continue
        try:
            protocol = json.loads((folder / "PROTOCOL.json").read_text(encoding="utf-8"))
        except (ValueError,OSError) as exc:
            error("malformed protocol %s: %s" % (label,exc))
            continue
        if protocol.get("experiment_id") != label or protocol.get("status") != "UNTESTED":
            error("%s id/status mismatch" % label)
        if protocol.get("plan_only") is not True or protocol.get("worker_claim") is not False:
            error("%s activated without evidence" % label)
        sections = protocol.get("split", {})
        dev = sections.get("development", protocol.get("development", {}))
        fresh = sections.get("fresh", protocol.get("fresh", {}))
        devseeds = dev.get("seeds", dev.get("worlds_or_seeds", []))
        frseeds = fresh.get("seeds", fresh.get("worlds_or_seeds", []))
        if list(devseeds) != [11,12,13] or list(frseeds) != [101,102,103,104,105]:
            error("%s dev/fresh seed mismatch" % label)
        if not fresh.get("locked_before_access", False):
            error("%s fresh not locked" % label)
        if set(protocol.get("gates", {})) != {"PASS","FAIL","UNCERTAIN"}:
            error("%s missing predeclared gates" % label)
        readme = (folder / "README.md").read_text(encoding="utf-8") if (folder / "README.md").exists() else ""
        if not all(x in readme for x in ("## H", "## T", "## D", "## C", "## U", "Mirror insertion")):
            error("%s incomplete H/T/D/C/U or insertion" % label)
        if (folder / "RESULTS_CORE.csv").exists() or (folder / "VERIFICATION.json").exists():
            error("%s staged experimental result inside plan folder" % label)
    for name in ("MA-1097-ALORA_CONTROL.md", "MA-578-KV_INT4_CONTROL.md",
                 "MA-1112-STANDARD_LORA_PREFIX.md"):
        if not (HERE / "existing" / name).exists():
            error("missing native-control supplement " + name)
    with (APPS / "CLAIM_LEDGER.csv").open(encoding="utf-8", newline="") as stream:
        claims=list(csv.DictReader(stream))
    for r in claims:
        if byid.get(r["id"],{}).get("status") != r["status"]:
            error("claim status mismatch %s" % r["id"])
    for r in rows:
        if r["status"] in ("PROMISING","FAIL","REPLICATED","ADOPTED") and not any(c["id"] == r["id"] for c in claims):
            error("completed MA without verified claim: %s" % r["id"])
    print("MA=%d PA=%d P0=%d P1=%d P2=%d UNTESTED=%d claims=%d plans=%d errors=%d" %
          (len(rows),len(headings),priority["P0"],priority["P1"],priority["P2"],
           status["UNTESTED"],len(claims),len(ISOLATED),len(errors)))
    for message in errors:
        print("ERROR:",message,file=sys.stderr)
    return not errors


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
