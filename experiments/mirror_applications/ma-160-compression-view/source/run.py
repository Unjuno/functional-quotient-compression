import argparse
import csv
import json
from pathlib import Path

from engine import METHODS, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("development", "fresh"), required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = [row for seed in args.seeds for condition in ("aligned", "independent") for row in run(seed, condition)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"split": args.split, "seeds": args.seeds, "rows": len(rows),
                      "methods": METHODS, "file": str(args.out)}, indent=2))


if __name__ == "__main__": main()
