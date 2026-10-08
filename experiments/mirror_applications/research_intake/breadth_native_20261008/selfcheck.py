#!/usr/bin/env python3
"""Read-only integrity check: isolated 2026-10-08 breadth-method intake.
Run from any working directory in a FULL clone of this isolated branch.
No network, third-party packages, model training, file writes, or worker edits.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import csv
import json
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
APPS = ROOT / "experiments" / "mirror_applications"
IDS = [f"MA-{i}" for i in range(1177, 1183)]
PAPERS = [f"PA{i}" for i in range(440, 453)]
SCHEMA = ("id", "family", "proposal", "shared_object", "mirror_coordinate",
          "logical_multiplicity", "cheapest_control", "first_metric", "priority",
          "status", "prior_art_level", "prior_art_refs", "mirror_delta", "worker_note")


def verify() -> bool:
    err: list[str] = []
    def complain(m: str) -> None:
        err.append(m)

    with (APPS / "IDEA_REGISTRY.csv").open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if tuple(reader.fieldnames or ()) != SCHEMA:
            complain("registry schema changed")
        rows = list(reader)
    idlist = [r.get("id", "") for r in rows]
    if idlist != [f"MA-{i:03d}" for i in range(1, 1183)]:
        complain("expected gap-free ordered 1182 MA rows")
    if len(set(idlist)) != len(rows) or any(None in row for row in rows):
        complain("duplicate IDs or malformed CSV row widths")
    pri = Counter(r.get("priority") for r in rows)
    status = Counter(r.get("status") for r in rows)
    if pri != {"P0": 647, "P1": 432, "P2": 103}:
        complain(f"priorities changed {dict(pri)}")
    if status != {"UNTESTED": 1135, "PROMISING": 29, "FAIL": 18}:
        complain(f"scientific statuses changed {dict(status)}")
    prior = (ROOT / "docs" / "phase2" / "MIRROR_APPLICATION_PRIOR_ART.md").read_text(encoding="utf-8")
    pa_ids = [int(s) for s in re.findall(r"^## PA(\d+)\b", prior, re.MULTILINE)]
    if pa_ids != list(range(1, 453)):
        complain("expected 452 sequential PA records")
    for row in rows:
        for ref in re.findall(r"PA(\d+)\b", row.get("prior_art_refs", "")):
            if int(ref) not in pa_ids:
                complain("orphan paper PA%s on %s" % (ref, row["id"]))
    byid = {r["id"]: r for r in rows}
    manifest = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("new_MA") != IDS or manifest.get("new_PA") != PAPERS:
        complain("isolated manifest missing MA/PA entry")
    if manifest.get("worker_queue_change") or not manifest.get("quarantined") or not manifest.get("no_special_priority"):
        complain("worker isolation/neutral-priority invariant changed")
    with (APPS / "CLAIM_LEDGER.csv").open(encoding="utf-8", newline="") as fp:
        claims = list(csv.DictReader(fp))
    if len(claims) != 47 or any(byid.get(c.get("id"), {}).get("status") != c.get("status") for c in claims):
        complain("47 existing experiment statuses no longer match claims")
    board = (APPS / "STATUS_BOARD.md").read_text(encoding="utf-8")
    if "Registered candidates: **1182**" not in board or "1135 UNTESTED" not in board:
        complain("status board count changed")
    if "MA-1175 is the FIRST" in board:
        complain("the unwanted special priority remains")
    queue = (APPS / "research_intake" / "single_forward_prefetch_20261008" / "RESEARCH_FOCUS_QUEUE.md").read_text(encoding="utf-8")
    if "NO special experiment priority" not in queue:
        complain("the research focus queue was not de-prioritized")
    with (HERE / "EXPERIMENT_MAP.csv").open(encoding="utf-8", newline="") as fp:
        mapping = list(csv.DictReader(fp))
    if [r.get("id") for r in mapping] != IDS:
        complain("experiment map row order/count incorrect")
    for row in mapping:
        if row.get("status") != "UNTESTED":
            complain("map claimed results")
    for id_ in IDS:
        obj = byid.get(id_, {})
        if obj.get("status") != "UNTESTED" or "ISOLATED" not in obj.get("worker_note", ""):
            complain(f"{id_} has false scientific status or missing quarantine flag")
        directory = HERE / "plans" / id_
        for name in ("README.md", "PROTOCOL.json", "STATUS.md"):
            if not (directory / name).is_file():
                complain(f"{id_} missing {name}")
        if not (directory / "PROTOCOL.json").exists():
            continue
        p = json.loads((directory / "PROTOCOL.json").read_text(encoding="utf-8"))
        if p.get("experiment_id") != id_ or p.get("status") != "UNTESTED":
            complain(f"{id_} wrong protocol ID/status")
        if p.get("plan_only") is not True or p.get("worker_claim") is not False:
            complain(f"{id_} unexpectedly activated")
        split = p.get("splits", {})
        if split.get("development", {}).get("seeds") != [11, 12, 13]:
            complain(f"{id_} wrong development seeds")
        fresh = split.get("fresh", {})
        if fresh.get("seeds") != [101, 102, 103, 104, 105] or fresh.get("locked_before_access") is not True:
            complain(f"{id_} unfreezed fresh")
        if set(p.get("gates", {})) != {"PASS", "FAIL", "UNCERTAIN"}:
            complain(f"{id_} not fully preregistered")
        prose = (directory / "README.md").read_text(encoding="utf-8")
        for section in ("## H", "## T", "## D", "## C", "## U", "Mirror insertion"):
            if section not in prose:
                complain(f"{id_} missing {section}")
        if (directory / "RESULTS_CORE.csv").exists() or (directory / "VERIFICATION.json").exists():
            complain(f"{id_} design folder has premature results")
    for obj in manifest.get("supplements", []):
        if not (HERE / obj["file"]).is_file():
            complain("missing native supplemental control: " + obj["file"])
    print(f"MA={len(rows)} PA={len(pa_ids)} new={len(IDS)} supplements={len(manifest.get('supplements', []))} "
          f"UNTESTED={status['UNTESTED']} claims={len(claims)} errors={len(err)}")
    for e in err:
        print("ERROR:", e, file=sys.stderr)
    return not err


if __name__ == "__main__":
    sys.exit(0 if verify() else 1)
