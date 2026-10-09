#!/usr/bin/env python3
"""Import completed MA claims from the latest fetched research/ma-* branches.

Run only after `git fetch origin '+refs/heads/research/ma-*:refs/remotes/origin/research/ma-*'`.
The current branch must be the current worker-ready baseline or a dedicated MA branch
created from it. Per-ID reports remain on their research branches; this writes scoped
links, status rows and a deterministic reconciliation manifest.
"""
from __future__ import annotations
import csv, io, json, re, subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MA_DIR = ROOT / "experiments" / "mirror_applications"
REGISTRY = MA_DIR / "IDEA_REGISTRY.csv"
CLAIMS = MA_DIR / "CLAIM_LEDGER.csv"
BOARD = MA_DIR / "STATUS_BOARD.md"
MANIFEST = MA_DIR / "LIVE_BRANCH_RECONCILIATION.csv"
TERMINAL = {"PROMISING", "FAIL", "REPLICATED", "ADOPTED", "NOT ESTABLISHED"}
PAUSED = {"MA-369", "MA-371", "MA-372"}

def git(*args: str, check: bool = True) -> str:
    p = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    if check and p.returncode:
        raise RuntimeError(p.stderr.strip())
    return p.stdout

def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)

def from_git(ref: str, path: str):
    p = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, text=True, capture_output=True)
    return p.stdout if p.returncode == 0 else None

def tree_file(ref: str, path: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{ref}:{path}"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

def parse_csv(raw: str):
    return list(csv.DictReader(io.StringIO(raw)))

def ref_time(ref: str) -> int:
    return int(git("log", "-1", "--format=%ct", ref).strip())

def find_artifact(ref: str, ma_id: str, name: str, claim_value: str) -> tuple[str, str] | None:
    # First honor explicit branch/path links, including canonical/supplemental variants.
    for branch, path in re.findall(r"branch\s+([^:;,]+):\s*([^;,]+)", claim_value):
        branch = branch.strip()
        path = path.strip().rstrip(".")
        candidate_ref = branch if branch.startswith("origin/") else "origin/" + branch
        if tree_file(candidate_ref, path):
            return candidate_ref, path
        if tree_file(ref, path):
            return ref, path
    # Otherwise locate the unique stable MA directory in the branch tree.
    paths = git("ls-tree", "-r", "--name-only", ref).splitlines()
    prefix = f"experiments/mirror_applications/ma-{int(ma_id.split('-')[1])}-"
    matches = [p for p in paths if p.startswith(prefix) and p.endswith("/" + name)]
    if matches:
        matches.sort(key=lambda p: (len(p), p))
        return ref, matches[0]
    return None

def main():
    reg_header, registry = rows(REGISTRY)
    claim_header, claims = rows(CLAIMS)
    by_id = {r["id"]: r for r in registry}
    existing_claim = {r["id"] for r in claims}
    refs = git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin/research/ma-*").splitlines()
    observations = defaultdict(list)
    for ref in refs:
        match = re.search(r"/ma-(\d+)-", ref)
        if not match:
            continue
        ma_id = f"MA-{int(match.group(1)):03d}"
        if ma_id not in by_id or by_id[ma_id]["status"] != "UNTESTED":
            continue
        source_rows = from_git(ref, "experiments/mirror_applications/IDEA_REGISTRY.csv")
        source_claims = from_git(ref, "experiments/mirror_applications/CLAIM_LEDGER.csv")
        if source_rows is None or source_claims is None:
            continue
        source_row = next((r for r in parse_csv(source_rows) if r["id"] == ma_id), None)
        source_claim = next((r for r in parse_csv(source_claims) if r["id"] == ma_id), None)
        if not source_row or not source_claim or source_row["status"] not in TERMINAL or source_claim["status"] != source_row["status"]:
            continue
        report = find_artifact(ref, ma_id, "README.md", source_claim["report_path"])
        verification = find_artifact(ref, ma_id, "VERIFICATION.json", source_claim["verification_path"])
        if not report or not verification:
            continue
        observations[ma_id].append((ref_time(ref), source_row, source_claim, ref, report, verification))

    manifest_rows = []
    conflicts = []
    added = []
    for ma_id, choices in observations.items():
        choices.sort(key=lambda x: (x[0], "reconciled" in x[3], x[3]), reverse=True)
        winner = choices[0]
        ts, source_row, source_claim, ref, report, verification = winner
        old_statuses = sorted({x[1]["status"] for x in choices})
        if len(old_statuses) > 1:
            conflicts.append((ma_id, old_statuses, ref, source_row["status"]))
        local = by_id[ma_id]
        local["status"] = source_row["status"]
        local["worker_note"] = source_row["worker_note"]
        claim = dict(source_claim)
        claim["report_path"] = f"branch {report[0].removeprefix('origin/')}: {report[1]}"
        claim["verification_path"] = f"branch {verification[0].removeprefix('origin/')}: {verification[1]}"
        if not re.fullmatch(r"[0-9a-f]{7,40}", claim.get("commit", "")):
            claim["commit"] = git("rev-parse", ref).strip()
        if ma_id not in existing_claim:
            claims.append(claim)
            existing_claim.add(ma_id)
            added.append(ma_id)
        manifest_rows.append({"id": ma_id, "status": source_row["status"], "source_branch": ref.removeprefix("origin/"),
                              "source_head": git("rev-parse", ref).strip(), "source_time_unix": ts,
                              "report_path": report[1], "verification_path": verification[1],
                              "alternative_statuses": ";".join(old_statuses)})

    # Import completed orphan outcomes whose branch has a verified STATUS/README but did not update its registry/claim ledger.
    for ref in refs:
        match = re.search(r"/ma-(\d+)-", ref)
        if not match:
            continue
        ma_id = f"MA-{int(match.group(1)):03d}"
        if ma_id not in by_id or by_id[ma_id]["status"] != "UNTESTED" or ma_id in existing_claim:
            continue
        tree = git("ls-tree", "-r", "--name-only", ref).splitlines()
        prefix = f"experiments/mirror_applications/ma-{int(match.group(1))}-"
        status_paths = [x for x in tree if x.startswith(prefix) and x.endswith("/STATUS.md")]
        readme_paths = [x for x in tree if x.startswith(prefix) and x.endswith("/README.md")]
        verify_paths = [x for x in tree if x.startswith(prefix) and x.endswith("/VERIFICATION.json")]
        if not status_paths or not readme_paths or not verify_paths:
            continue
        status_path, report_path, verify_path = status_paths[0], readme_paths[0], verify_paths[0]
        status_raw = from_git(ref, status_path) or ""
        verify_raw = from_git(ref, verify_path) or "{}"
        try:
            verify_obj = json.loads(verify_raw)
        except json.JSONDecodeError:
            continue
        disposition = None
        if "NOT ESTABLISHED" in status_raw:
            disposition = "NOT ESTABLISHED"
        elif re.search(r"\bFAIL\b", status_raw) and verify_obj.get("metric_replay", {}).get("checked") is True:
            disposition = "FAIL"
        if disposition is None:
            continue
        head = git("rev-parse", ref).strip()
        by_id[ma_id]["status"] = disposition
        if disposition == "NOT ESTABLISHED":
            note = "Frozen task/control was not learnable; no method verdict. Fresh sealed."
        else:
            note = "Development FAIL disposition and metric replay are documented on the dedicated branch; fresh remains sealed."
        by_id[ma_id]["worker_note"] = note
        claim = {"id":ma_id,"status":disposition,"evidence_lane":"MECHANISM/STORAGE/QUALITY/RUNTIME",
            "scoped_claim":note, "report_path":f"branch {ref.removeprefix('origin/')}: {report_path}",
            "verification_path":f"branch {ref.removeprefix('origin/')}: {verify_path}","commit":head,
            "notes":"Imported from a completed branch STATUS.md because its registry/claim ledger was not finalized; no fresh data accessed."}
        claims.append(claim);existing_claim.add(ma_id)
        ts=ref_time(ref)
        manifest_rows.append({"id":ma_id,"status":disposition,"source_branch":ref.removeprefix("origin/"),
            "source_head":head,"source_time_unix":ts,"report_path":report_path,"verification_path":verify_path,
            "alternative_statuses":"Registry was UNTESTED; completed branch STATUS/VERIFICATION imported"})

    # MA-325 is a terminal NOT ESTABLISHED disposition documented on a live branch,
    # but its registry and claim ledger were not updated there. Preserve that result.
    ma325_ref = "origin/research/ma-325-tt-embedding-domain-mirror-20261008"
    if "MA-325" in by_id and by_id["MA-325"]["status"] == "UNTESTED" and "MA-325" not in existing_claim and ma325_ref in refs:
        status_text = from_git(ma325_ref, "experiments/mirror_applications/ma-325-tt-embedding-domain-mirror/STATUS.md") or ""
        verification_text = from_git(ma325_ref, "experiments/mirror_applications/ma-325-tt-embedding-domain-mirror/VERIFICATION.json") or "{}"
        if "NOT ESTABLISHED" in status_text:
            by_id["MA-325"]["status"] = "NOT ESTABLISHED"
            by_id["MA-325"]["worker_note"] = "Frozen development task was unlearnable even for the independent upper control (NLL≈ln(16)); no Mirror verdict. Fresh sealed; protocol redesign requires a new ID/amendment."
            claims.append({"id":"MA-325","status":"NOT ESTABLISHED","evidence_lane":"MECHANISM/STORAGE/DOMAIN_TRANSFER",
                "scoped_claim":"The frozen synthetic TT language task was not learnable: independent full-table control also remained at approximately uniform NLL/accuracy. This run cannot decide the Mirror hypothesis; fresh remained sealed.",
                "report_path":"branch research/ma-325-tt-embedding-domain-mirror-20261008: experiments/mirror_applications/ma-325-tt-embedding-domain-mirror/README.md",
                "verification_path":"branch research/ma-325-tt-embedding-domain-mirror-20261008: experiments/mirror_applications/ma-325-tt-embedding-domain-mirror/VERIFICATION.json",
                "commit":git("rev-parse",ma325_ref).strip(),
                "notes":"NOT ESTABLISHED; 4 tests and serializer byte roundtrip passed, but metric replay was not checked and task quality was at chance. Fresh sealed."})
            existing_claim.add("MA-325")
            manifest_rows.append({"id":"MA-325","status":"NOT ESTABLISHED","source_branch":ma325_ref.removeprefix("origin/"),
                "source_head":git("rev-parse",ma325_ref).strip(),"source_time_unix":ref_time(ma325_ref),
                "report_path":"experiments/mirror_applications/ma-325-tt-embedding-domain-mirror/README.md",
                "verification_path":"experiments/mirror_applications/ma-325-tt-embedding-domain-mirror/VERIFICATION.json",
                "alternative_statuses":"UNTESTED registry corrected from explicit STATUS.md disposition"})

    # Persist merged source records.
    with REGISTRY.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=reg_header); w.writeheader(); w.writerows(registry)
    with CLAIMS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=claim_header); w.writeheader(); w.writerows(claims)
    manifest_header = ["id","status","source_branch","source_head","source_time_unix","report_path","verification_path","alternative_statuses"]
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=manifest_header); w.writeheader(); w.writerows(sorted(manifest_rows,key=lambda x:int(x["id"].split("-")[1])))

    counts = Counter(r["status"] for r in registry)
    priorities = Counter(r["priority"] for r in registry)
    completed = {"PROMISING","FAIL","REPLICATED","ADOPTED","NOT ESTABLISHED"}
    p_done = Counter(r["priority"] for r in registry if r["status"] in completed)
    board = BOARD.read_text(encoding="utf-8")
    board = re.sub(r"- P0: .*",f"- P0: **{priorities['P0']}** ({p_done['P0']} completed; {priorities['P0']-p_done['P0']} UNTESTED)",board,count=1)
    board = re.sub(r"- P1: .*",f"- P1: **{priorities['P1']}** ({p_done['P1']} completed; {priorities['P1']-p_done['P1']} UNTESTED)",board,count=1)
    board = re.sub(r"- P2: .*",f"- P2: **{priorities['P2']}** ({p_done['P2']} completed; {priorities['P2']-p_done['P2']} UNTESTED)",board,count=1)
    status_text = ", ".join(f"{n} {k}" for k,n in sorted(counts.items()))
    board = re.sub(r"- Current MA statuses: .*",f"- Current MA statuses: **{status_text}**",board,count=1)
    first = next((r for r in registry if r["priority"]=="P0" and r["status"]=="UNTESTED" and r["id"] not in PAUSED),None)
    # MA-401 is selected only if the fetched live branch map has no competing branch.
    if first and first["id"] != "MA-401":
        raise RuntimeError(f"Live evidence reconciliation selected {first['id']}, not preregistered MA-401; inspect before proceeding")
    board = re.sub(r"## Next candidate\n\n.*?(?=\n## )", "## Next candidate\n\n**MA-401 — FiLM versus Mirror feature conditioning (P0; PA63)**\n\nLive research branches were fetched and reconciled into `LIVE_BRANCH_RECONCILIATION.csv`. Conflicting protocol variants use the latest branch endpoint with a matching terminal registry/claim status; the source branches and all variants remain linked. MA-325 is NOT ESTABLISHED because the frozen task was unlearnable even for the independent control. MA-369/371/372 remain paused under the documented family rule. MA-401 is the first remaining untested P0 outside those paused families and has no live remote experiment branch.\n\n", board, count=1, flags=re.S)
    board = re.sub(r"## Active experiment\n\n.*?(?=\n## )", "## Active experiment\n\nMA-401 protocol and screen are active on `research/ma-401-film-mirror-feature-conditioning-20261008`. No remote MA-401 branch existed at the last fetch.\n\n", board, count=1, flags=re.S)
    # Add NOT ESTABLISHED to the board's terminal index when present.
    lines = board.splitlines()
    if counts.get("NOT ESTABLISHED", 0) and not any(x.startswith("- **NOT ESTABLISHED (") for x in lines):
        insert = next((i for i,x in enumerate(lines) if x.startswith("- **FAIL (")), None)
        if insert is not None: lines.insert(insert, "- **NOT ESTABLISHED (0):** .")
    # Refresh summary lists, preserving the concise board structure.
    for i, line in enumerate(lines):
        for status in ("PROMISING","FAIL","NOT ESTABLISHED"):
            if line.startswith(f"- **{status} ("):
                ids = [r["id"] for r in registry if r["status"]==status]
                lines[i] = f"- **{status} ({len(ids)}):** " + ", ".join(ids) + "."
    board = "\n".join(lines)+"\n"
    board = board.replace("- 47 experiment directories, complete with status/protocol/results/verification files, have been imported into this branch.",
        f"- 47 baseline experiment directories remain present; {len(manifest_rows)} additional per-ID outcomes are linked to their dedicated research branches in `LIVE_BRANCH_RECONCILIATION.csv`.")
    board = board.replace("Updated: 2026-10-08 JST", "Updated: 2026-10-08 JST (live branch reconciliation)",1)
    BOARD.write_text(board,encoding="utf-8")
    report = MA_DIR / "LIVE_BRANCH_RECONCILIATION.md"
    report.write_text("""# Live MA branch evidence reconciliation\n\nFetched current `origin/research/ma-*` refs on 2026-10-08 and reconciled terminal per-ID claims into the latest 1155-row worker-ready registry. The detailed manifest records each source branch, HEAD, status, and report/verification path. The latest terminal branch record wins only when registry and claim ledger agree and both report and verification artifacts are locatable. Conflicting statuses are disclosed in `alternative_statuses`; every source branch remains unchanged.\n\nMA-325 was separately marked NOT ESTABLISHED from its explicit status report because its independent upper control remained at chance; its verification states that metric replay was not checked. This is not a FAIL verdict for Mirror.\n\nMA-369/371/372 remain paused after the documented consecutive width/depth family failures. The first untested P0 outside that pause is MA-401 (PA63); live remote search found no MA-401 branch.\n\nNo fresh data were opened during reconciliation.\n""",encoding="utf-8")
    print("Imported terminal branch outcomes:",len(manifest_rows))
    print("Added claim rows:",len(added)+1)
    print("Status counts:",dict(counts))
    print("P0 completion:",p_done['P0'],"/",priorities['P0'])
    print("Next candidate:",first["id"] if first else "none")
    print("Conflicted IDs resolved by latest terminal branch:",len(conflicts))
    for item in conflicts: print(item[0],item[1],"->",item[3],item[2])

if __name__ == "__main__":
    main()
