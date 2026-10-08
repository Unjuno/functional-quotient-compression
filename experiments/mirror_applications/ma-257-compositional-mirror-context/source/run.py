#!/usr/bin/env python3
"""Run the preregistered MA-257 development or fresh mechanism screen."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as e

ROOT = Path(__file__).resolve().parents[1]
DEV_SEEDS = (25701, 25702, 25703)
FRESH_SEEDS = (25711, 25712, 25713)


def run(split: str, out: Path) -> None:
    if split == "development":
        seeds, supports = DEV_SEEDS, (1, 2, 3)
    elif split == "fresh":
        protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
        support = protocol["fresh"]["support_residues"]
        if support not in (1, 2, 3):
            raise RuntimeError("Fresh remains sealed until development selection is frozen in PROTOCOL.json")
        seeds, supports = FRESH_SEEDS, (support,)
    else:
        raise ValueError(split)
    rows = []
    for seed in seeds:
        for condition in ("aligned", "independent"):
            for support in supports:
                rows.extend(e.run_world(seed, condition, support))
    report = {"experiment_id": "MA-257", "split": split, "seeds": seeds,
              "support_residues": supports, "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {len(rows)} method rows to {out}")


def select_dev(report: Path) -> int | None:
    data = json.loads(report.read_text())
    rows = data["rows"]
    for support in (1, 2, 3):
        okay = True
        for seed in DEV_SEEDS:
            for method in ("mirror_f32", "coeff_f16"):
                r = next(x for x in rows if x["seed"] == seed and x["condition"] == "aligned" and
                         x["support_residues"] == support and x["method"] == method)
                if r["mean_val_all_nMSE"] > 1e-5 or r["max_val_all_nMSE"] > 1e-4:
                    okay = False
        if okay:
            return support
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("development", "fresh"), required=True)
    p.add_argument("--out", type=Path, default=ROOT / "source" / "dev_results.json")
    a = p.parse_args()
    run(a.split, a.out)
    if a.split == "development":
        selected = select_dev(a.out)
        print(f"frozen support_residues selection: {selected}")

if __name__ == "__main__":
    main()
