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
EXPECTED_MA = 1176
EXPECTED_PA = 439
ISOLATED = range(1156, 1175)
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
        folder = ((ROOT / "experiments" / "mirror_applications" / "research_intake" / "formula_crossovers_20261008" / "plans" / label) if num >= 1171 else (HERE / "plans" / label))
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
        firewall = protocol.get("data_firewall", {})
        dataset = protocol.get("dataset", {})
        devseeds = dev.get("seeds", dev.get("worlds_or_seeds", firewall.get("dev_seeds", dataset.get("dev_seeds", []))))
        frseeds = fresh.get("seeds", fresh.get("worlds_or_seeds", firewall.get("fresh_seeds", dataset.get("fresh_seeds", []))))
        if list(devseeds) != [11,12,13] or list(frseeds) != [101,102,103,104,105]:
            error("%s dev/fresh seed mismatch" % label)
        if not fresh.get("locked_before_access", firewall.get("fresh_locked_before_access", dataset.get("fresh_frozen", False))):
            error("%s fresh not locked" % label)
        if set(protocol.get("gates", {})) != {"PASS","FAIL","UNCERTAIN"}:
            error("%s missing predeclared gates" % label)
        readme = (folder / "README.md").read_text(encoding="utf-8") if (folder / "README.md").exists() else ""
        if not all(x in readme for x in ("## H", "## T", "## D", "## C", "## U", "Mirror insertion")):
            error("%s incomplete H/T/D/C/U or insertion" % label)
        if (folder / "RESULTS_CORE.csv").exists() or (folder / "VERIFICATION.json").exists():
            error("%s staged experimental result inside plan folder" % label)
    # Two later SFM plans are in their own isolated research subtree, not
    # historical geometry/formula MANIFEST.json and not the canonical worker.
    sfm_root = APPS / "research_intake" / "single_forward_prefetch_20261008"
    for num in (1175, 1176):
        label = "MA-%d" % num
        row = byid.get(label, {})
        if row.get("status") != "UNTESTED" or "ISOLATED" not in row.get("worker_note", ""):
            error("%s not isolated/UNTESTED" % label)
        folder = sfm_root / "plans" / label
        for filename in ("README.md", "PROTOCOL.json", "STATUS.md"):
            if not (folder / filename).is_file():
                error("missing SFM plan %s/%s" % (label, filename))
        proto_file = folder / "PROTOCOL.json"
        if proto_file.is_file():
            protocol = json.loads(proto_file.read_text(encoding="utf-8"))
            if protocol.get("experiment_id") != label or protocol.get("status") != "UNTESTED":
                error("SFM plan ID/status invalid " + label)
            if protocol.get("worker_claim") is not False or protocol.get("plan_only") is not True:
                error("SFM plan incorrectly active " + label)
    import hashlib
    trained = sfm_root / "pilots" / "sfm003_trained_multioutput"
    checksums = {
        "source/run_sfm003.py": "60541467a66369157747987a52c42ee5d16a6383d3e6a3e49c2761d0baa5cf48",
        "results/dev_quality.csv": "912c0251521fcb8dd46cdc05891fdb6841426e50493f9472ed5c4c79c3bc1b2b",
        "results/fresh_quality.csv": "de659664792dacc6c49d23a5ab7fa24dbb924c4f21f0228a2d31f57508183d38",
        "results/dev_timing.csv": "0e305df80d9db11e487366e6cdcd7a719f6da18e253691802567a26b92ddc0cc",
        "results/fresh_timing.csv": "0c535da6590082308a6672341d2927ad0ebd07f5fefaf83605afb0c7f668898b",
    }
    for relative, digest in checksums.items():
        name = trained / relative
        if not name.is_file():
            error("missing Stage-0 source/data " + relative)
        elif hashlib.sha256(name.read_bytes()).hexdigest() != digest:
            error("Stage-0 SHA mismatch " + relative)
    if not (trained / "PROTOCOL.json").is_file():
        error("SFM003 frozen protocol missing")
    if not (sfm_root / "SFM003_REPORT.md").is_file():
        error("SFM003 report missing")
    if not (sfm_root / "RESEARCH_FOCUS_QUEUE.md").is_file():
        error("research-only focus queue missing")
    for name in ("MA-1097-ALORA_CONTROL.md", "MA-578-KV_INT4_CONTROL.md",
                 "MA-1112-STANDARD_LORA_PREFIX.md"):
        if not (HERE / "existing" / name).exists():
            error("missing native-control supplement " + name)
    formula_base = ROOT / "experiments" / "mirror_applications" / "research_intake" / "formula_crossovers_20261008"
    for ma_id in ['MA-690','MA-1114','MA-1099','MA-1156','MA-1160','MA-1165','MA-1166','MA-1081','MA-1146','MA-248','MA-231','MA-196']:
        if not (formula_base / "existing" / (ma_id+".md")).is_file():
            error("missing formula crossover supplement " + ma_id)
    for required in ("FORMULA_LEDGER.md", "FORMULA_STAGE0_PROTOCOL.json",
                     "pilots/formula_stage0/REPORT.md",
                     "pilots/formula_stage0/source/run_formula_audit.py",
                     "pilots/formula_stage0/results/fresh_raw.csv",
                     "pilots/formula_stage0/results/VERIFICATION.json"):
        if not (formula_base / required).is_file():
            error("missing formula audit artifact "+required)
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
