#!/usr/bin/env python3
"""Audit the authoritative Mirror registry/claims/PA IDs, including MA-1000+.

Usage from repository root:
  python experiments/mirror_applications/check_registry_integrity.py

Read-only; no external packages, network, model weights, or GPU required.
The status board is an operational cache, not the authoritative scientific record.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REGISTRY = HERE / "IDEA_REGISTRY.csv"
CLAIMS = HERE / "CLAIM_LEDGER.csv"
BOARD = HERE / "STATUS_BOARD.md"
PRIOR_ART = ROOT / "docs" / "phase2" / "MIRROR_APPLICATION_PRIOR_ART.md"

MA_ID = re.compile(r"^MA-([0-9]{3,})$")
PA_HEADING = re.compile(r"^## PA([0-9]+)\b", re.MULTILINE)
PA_REF = re.compile(r"PA([0-9]+)")
STATUS_ALLOWED = {"UNTESTED", "SCREENING", "PROMISING", "FAIL", "REPLICATED", "ADOPTED", "NOT ESTABLISHED"}
FINISHED = {"PROMISING", "FAIL", "REPLICATED", "ADOPTED", "NOT ESTABLISHED"}
REQUIRED = {
    "id", "family", "proposal", "shared_object", "mirror_coordinate",
    "logical_multiplicity", "cheapest_control", "first_metric", "priority",
    "status", "prior_art_level", "prior_art_refs", "mirror_delta", "worker_note",
}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        header = list(reader.fieldnames)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty table")
    for row_number, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}:{row_number}: inconsistent CSV width")
    return header, rows


def audit() -> list[str]:
    errors: list[str] = []
    header, rows = read_rows(REGISTRY)
    if set(header) != REQUIRED or len(header) != len(REQUIRED):
        errors.append("Registry header/width differs from the 14-column contract")
    seen: set[int] = set()
    by_id: dict[str, str] = {}
    statuses: Counter[str] = Counter()
    priorities: Counter[str] = Counter()
    prior_refs: set[int] = set()

    for row in rows:
        key = row["id"]
        match = MA_ID.fullmatch(key)
        if match is None:
            errors.append(f"Invalid MA ID: {key!r}; supports MA-001, MA-999, MA-1000...")
            continue
        idx = int(match.group(1))
        if key != f"MA-{idx:03d}":
            errors.append(f"Noncanonical MA ID: {key!r}")
        if idx in seen:
            errors.append(f"Duplicate MA ID: {key}")
        seen.add(idx)
        status = row["status"]
        if status not in STATUS_ALLOWED:
            errors.append(f"Invalid status {key}: {status}")
        if row["priority"] not in {"P0", "P1", "P2"}:
            errors.append(f"Invalid priority {key}: {row['priority']}")
        statuses[status] += 1
        priorities[row["priority"]] += 1
        by_id[key] = status
        prior_refs.update(int(m.group(1)) for m in PA_REF.finditer(row["prior_art_refs"]))

    if seen and seen != set(range(1, max(seen) + 1)):
        missing = sorted(set(range(1, max(seen) + 1)) - seen)
        errors.append(f"Gap in MA IDs: {missing[:12]}")

    art = PRIOR_ART.read_text(encoding="utf-8")
    pa_numbers = [int(m.group(1)) for m in PA_HEADING.finditer(art)]
    if len(pa_numbers) != len(set(pa_numbers)):
        errors.append("Duplicate PA heading")
    if pa_numbers != list(range(1, max(pa_numbers, default=0) + 1)):
        errors.append("PA headings are not sequential from PA01")
    unresolved = sorted(prior_refs - set(pa_numbers))
    if unresolved:
        errors.append(f"Unresolved PA references: {unresolved}")

    claim_header, claims = read_rows(CLAIMS)
    if not {"id", "status", "evidence_lane", "report_path", "verification_path", "commit"}.issubset(claim_header):
        errors.append("Claim ledger missing mandatory columns")
    claimed: set[str] = set()
    for claim in claims:
        key = claim["id"]
        if key in claimed:
            errors.append(f"Duplicate claim ledger ID: {key}")
        claimed.add(key)
        if key not in by_id:
            errors.append(f"Orphan claim: {key}")
        elif by_id[key] != claim["status"]:
            errors.append(f"Claim/registry mismatch: {key}: {claim['status']} vs {by_id[key]}")
    missing_claims = sorted(key for key, status in by_id.items() if status in FINISHED and key not in claimed)
    if missing_claims:
        errors.append(f"Completed statuses missing claim evidence: {missing_claims[:12]}")

    board = BOARD.read_text(encoding="utf-8")
    marker = re.search(r"Registered candidates: \*\*([0-9]+)\*\*", board)
    if not marker or int(marker.group(1)) != len(rows):
        errors.append("STATUS_BOARD registry count does not match IDEA_REGISTRY")
    for status, count in statuses.items():
        needle = f"{count} {status}"
        if needle not in board:
            errors.append(f"STATUS_BOARD does not show '{needle}'")
    for priority, count in priorities.items():
        if f"{priority}: **{count}**" not in board:
            errors.append(f"STATUS_BOARD priority {priority} does not show count {count}")

    print(f"MA registry: {len(rows)} | P0 {priorities['P0']} P1 {priorities['P1']} P2 {priorities['P2']}")
    print("Status: " + ", ".join(f"{status} {n}" for status, n in sorted(statuses.items())))
    print(f"PA references: {len(pa_numbers)} | Claims: {len(claims)}")
    print(f"Max MA ID: MA-{max(seen, default=0):03d} | Errors: {len(errors)}")
    return errors


if __name__ == "__main__":
    try:
        found = audit()
    except (OSError, ValueError, KeyError) as exc:
        print(f"INTEGRITY ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    for problem in found:
        print(f"INTEGRITY ERROR: {problem}", file=sys.stderr)
    sys.exit(1 if found else 0)
