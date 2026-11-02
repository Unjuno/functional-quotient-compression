from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

from run import WIDTHS, nrmse, predict, teacher_world


def verify_world(directory: Path) -> int:
    rows = list(csv.DictReader((directory / "RESULTS_CORE.csv").open()))
    world_id = int(rows[0]["world"])
    split = rows[0]["split"]
    world = teacher_world(world_id)
    checks = 0
    for method in ("usnet_prefix", "mirror_angle", "direct_coeff", "independent_full"):
        path = directory / f"{method}.npz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as bundle:
            if method == "usnet_prefix":
                U, V = bundle["U"], bundle["V"]
                angle_fit = coeff = None
            elif method == "mirror_angle":
                U, V, angle_fit = bundle["U"], bundle["V"], bundle["angle_fit"]
                coeff = None
            elif method == "direct_coeff":
                U, V, coeff = bundle["U"], bundle["V"], bundle["coefficients"]
                angle_fit = None
            else:
                independent = bundle["operators"]
                U = V = angle_fit = coeff = None
        for row in rows:
            if row["method"] != method:
                continue
            k = int(row["width"])
            _, _, x, y = world["data"][k]
            if method == "independent_full":
                pred = x @ independent[WIDTHS.index(k)]
            else:
                A, _ = predict(U, V, k, method, angle_fit, coeff)
                pred = x @ A
            metric = nrmse(pred, y)
            if abs(metric - float(row["heldout_nrmse"])) > 1e-12:
                raise AssertionError(f"metric mismatch for {world_id}/{method}/{k}: {metric} != {row['heldout_nrmse']}")
            if int(row["payload_bytes"]) != path.stat().st_size or row["payload_sha256"] != digest:
                raise AssertionError(f"payload accounting mismatch for {world_id}/{method}/{k}")
            checks += 1
    return checks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    total = sum(verify_world(path) for path in args.directories)
    print(f"verified metric rows and payload bytes/hashes: {total}")
