#!/usr/bin/env python3
"""Read-only registry/protocol validation for isolated 2026-10-08 crossdomain sweep.
Runs on full repository checkout of the research branch; no file writes or external modules.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import csv
import json
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
APP = ROOT / "experiments" / "mirror_applications"
IDS = [f"MA-{i}" for i in range(1183, 1189)]
EXPECTED_PA = list(range(1, 463))
EXPECTED_STATUS = {"UNTESTED": 1141, "PROMISING": 29, "FAIL": 18}
EXPECTED_PRIORITY = {"P0": 649, "P1": 436, "P2": 103}

def audit():
    errors=[]
    def need(ok, reason):
        if not ok: errors.append(reason)
    with (APP/"IDEA_REGISTRY.csv").open(encoding="utf-8", newline="") as f:
        rd=csv.DictReader(f)
        header=rd.fieldnames
        rows=list(rd)
    need(len(rows)==1188,"MA count expected 1188")
    need(all(r.get("id")==f"MA-{i:03d}" for i,r in enumerate(rows,1)),"MA ordered contiguous IDs")
    need(all(None not in r for r in rows),"MA row schema/no extra columns")
    need(dict(Counter(r["status"] for r in rows))==EXPECTED_STATUS,"scientific statuses")
    need(dict(Counter(r["priority"] for r in rows))==EXPECTED_PRIORITY,"interest labels")
    byid={r["id"]:r for r in rows}
    pa_file=(ROOT/"docs"/"phase2"/"MIRROR_APPLICATION_PRIOR_ART.md").read_text(encoding="utf-8")
    got_pa=[int(x) for x in re.findall(r"^## PA(\d+)\b",pa_file,re.MULTILINE)]
    need(got_pa==EXPECTED_PA,"PA 1..462 complete")
    for row in rows:
        for a in re.findall(r"PA(\d+)",row.get("prior_art_refs","")):
            need(int(a) in got_pa,f"orphan PA{a} in {row['id']}")
    with (APP/"CLAIM_LEDGER.csv").open(encoding="utf-8",newline="") as f:
        claims=list(csv.DictReader(f))
    need(len(claims)==47,"47 historical claims")
    need(all(byid[c["id"]]["status"]==c["status"] for c in claims),"claimed historic outcomes unchanged")
    manifest=json.loads((HERE/"MANIFEST.json").read_text(encoding="utf-8"))
    need(manifest["new_MA"]==IDS,"manifest new MA")
    need(manifest["new_PA"]==[f"PA{i}" for i in range(453,463)],"manifest new PA")
    need(manifest["isolation"] and not manifest["worker_queue_changed"]
         and manifest["neutral_research_priority"],"worker neutrality")
    board=(APP/"STATUS_BOARD.md").read_text(encoding="utf-8")
    need("Registered candidates: **1188**" in board and "1141 UNTESTED" in board,"status board counts")
    mapping=list(csv.DictReader((HERE/"EXPERIMENT_MAP.csv").open(encoding="utf-8",newline="")))
    need([r["id"] for r in mapping]==IDS,"experiment map")
    for id in IDS:
        folder=HERE/"plans"/id
        need(all((folder/p).is_file() for p in ("README.md","PROTOCOL.json","STATUS.md")),f"{id} 3-file plan")
        try:
            p=json.loads((folder/"PROTOCOL.json").read_text(encoding="utf-8"))
            d=(folder/"README.md").read_text(encoding="utf-8")
            need(p["experiment_id"]==id and p["status"]=="UNTESTED"
                 and p["plan_only"] and p["worker_claim"] is False, f"{id} protocol state")
            need(p["splits"]["development"]["seeds"]==[11,12,13]
                 and p["splits"]["fresh"]["seeds"]==[101,102,103,104,105]
                 and p["splits"]["fresh"]["locked_before_use"], f"{id} train/audit firewall")
            need(set(p["gates"])=={"PASS","FAIL","UNCERTAIN"}, f"{id} complete gates")
            need(len(p["native_and_simple_controls"])>=6,f"{id} six controls")
            need(all(t in d for t in ("## H","## T","## D","## C","## U","Mirror insertion")),f"{id} HTDCU")
            need(not (folder/"RESULTS_CORE.csv").exists(),f"{id} unrun status")
        except Exception as ex: errors.append(f"{id} unreadable: {ex}")
    for id in manifest["supplement_existing"]:
        need((HERE/"existing"/f"{id}_NATIVE_CONTROL.md").is_file(),f"supplement {id}")
    stage=HERE/"pilots"/"mcx_stage0"
    need((stage/"PROTOCOL.json").is_file() and (stage/"REPORT.md").is_file(),"stage0 evidence")
    need((stage/"source"/"algebra_reference.py").is_file(),"stage0 executable")
    need((stage/"results"/"VERIFICATION.json").is_file(),"stage0 summary")
    print(f"MA={len(rows)} PA={len(got_pa)} new={len(IDS)} claims={len(claims)} "
          f"new_controls={len(manifest['supplement_existing'])} errors={len(errors)}")
    for e in errors: print("ERROR:",e,file=sys.stderr)
    return not errors

if __name__=="__main__":
    raise SystemExit(0 if audit() else 1)
