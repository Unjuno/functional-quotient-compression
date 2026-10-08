import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

from run import METHODS, evaluate, load_serialized, make_data


def verify(root, result_path):
    root = Path(root)
    result = json.loads(Path(result_path).read_text())
    checks = []
    data = make_data(result["seed"])
    for row in result["summaries"]:
        path = root / "payloads" / f"development_{result['seed']}_{row['method']}.npz"
        raw = path.read_bytes()
        checks.append({
            "method": row["method"],
            "bytes_match": len(raw) == row["serialized_bytes"],
            "sha256_match": hashlib.sha256(raw).hexdigest() == row["payload_sha256"],
            "zip_names": sorted(zipfile.ZipFile(path).namelist()),
            "method_id": METHODS.index(row["method"]),
        })
        with zipfile.ZipFile(path) as archive:
            arrays = {name[:-4]: np.load(archive.open(name), allow_pickle=False)
                      for name in archive.namelist()}
        checks[-1]["metadata_match"] = arrays["meta"].tolist() == [12, 12, 144, 16, 8, METHODS.index(row["method"])]
        checks[-1]["arrays_finite"] = all(np.isfinite(array).all() for array in arrays.values())
        model = load_serialized(row["method"], result["seed"], arrays)
        recomputed = {split: evaluate(model, data[split], data) for split in ("validation", "test")}
        checks[-1]["metrics_match"] = all(
            abs(recomputed[split][metric] - row[split][metric]) < 1e-7
            for split in recomputed for metric in recomputed[split]
        )
    ok = all(c["bytes_match"] and c["sha256_match"] and c["metadata_match"] and c["arrays_finite"] and c["metrics_match"] for c in checks)
    return {"condition": result["condition"], "seed": result["seed"], "payloads": checks,
            "exact_replay": "payloads reloaded and validation/test metrics recomputed from the frozen seed",
            "passed": ok}


if __name__ == "__main__":
    report = verify(sys.argv[1], sys.argv[2])
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
