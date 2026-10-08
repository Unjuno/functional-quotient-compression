import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from run import PromptModel, make_data

torch.set_num_threads(1)


def benchmark(result_path, payload_dir, repeats=20):
    result = json.loads(Path(result_path).read_text())
    data = make_data(result["seed"])
    rows = []
    for record in result["summaries"]:
        method = record["method"]
        payload = Path(payload_dir) / f"{result['condition']}_{result['seed']}_{method}.npz"
        with zipfile.ZipFile(payload) as archive:
            arrays = {name[:-4]: np.lib.format.read_array(io.BytesIO(archive.read(name)), allow_pickle=False)
                      for name in archive.namelist()}
        model = PromptModel(method, result["seed"])
        model.load_state_dict({k: torch.tensor(arrays[k], dtype=v.dtype)
                               for k, v in model.state_dict().items()})
        model.eval()
        keys = torch.tensor(arrays["router_keys"].astype(np.float32))
        batch = data["test"]
        with torch.no_grad():
            for _ in range(3):
                model(batch["key"], batch["features"], keys)
            start = time.perf_counter()
            for _ in range(repeats):
                model(batch["key"], batch["features"], keys)
            elapsed = time.perf_counter() - start
        rows.append({"method": method, "payload_bytes": payload.stat().st_size,
                     "examples": len(batch["labels"]) * repeats, "wall_seconds": elapsed,
                     "examples_per_second": len(batch["labels"]) * repeats / elapsed})
    return {"seed": result["seed"], "condition": result["condition"],
            "repeats": repeats, "runtime": "CPU; one torch thread", "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("payload_dir")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.result, args.payload_dir, args.repeats), indent=2))
