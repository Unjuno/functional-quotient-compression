"""Reload every inference archive and replay its registered test metrics."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiment as exp  # noqa: E402


def verify(root: Path):
    checks = []
    for family, seed in (("aligned", 38301), ("aligned", 38302),
                         ("unrelated", 38301), ("unrelated", 38302)):
        x, y, task, _, _, _ = exp.dataset(seed, family, "test")
        summary = json.loads((root / f"{family}_{seed}_result.json").read_text())
        for method in ("explicit", "scalar", "mirror", "hyper"):
            path = root / f"{family}_{seed}_{method}.zip"
            with zipfile.ZipFile(path) as archive:
                arrays = {Path(name).stem: np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
                          for name in archive.namelist() if name.endswith(".npy")}
            model = exp.Pool(method)
            state = {key: torch.from_numpy(arrays[key.replace(".", "_")].astype("float32"))
                     for key in model.state_dict()}
            model.load_state_dict(state)
            got = exp.evaluate(model, x, y, task,
                               torch.from_numpy(arrays["keys"].astype("float32")),
                               torch.from_numpy(arrays["classifier"].astype("float32")))
            expected = summary["methods"][method]["test"]
            assert got["accuracy"] == expected["accuracy"]
            assert got["per_task_accuracy"] == expected["per_task_accuracy"]
            nll_delta = abs(got["nll"] - expected["nll"])
            assert nll_delta < 5e-5
            assert "keys" in arrays and "classifier" in arrays
            blob = path.read_bytes()
            checks.append({"file": path.name, "bytes": len(blob),
                           "sha256": hashlib.sha256(blob).hexdigest(),
                           "nll_delta": nll_delta})
    return checks


if __name__ == "__main__":
    result = verify(HERE.parent / "results" / "development")
    print(json.dumps({"payloads_replayed": len(result), "checks": result}, indent=2))
