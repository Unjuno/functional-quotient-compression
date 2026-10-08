import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from run import load_serialized, make_data

torch.set_num_threads(1)


def benchmark(result_path, outdir, repeats=50):
    result = json.loads(Path(result_path).read_text())
    data = make_data(result["seed"])
    rows = []
    for row in result["summaries"]:
        method, seed = row["method"], result["seed"]
        path = Path(outdir) / "payloads" / f"{result['condition']}_{seed}_{method}.npz"
        with zipfile.ZipFile(path) as archive:
            arrays = {name[:-4]: np.lib.format.read_array(io.BytesIO(archive.read(name)), allow_pickle=False)
                      for name in archive.namelist()}
        model, indices = load_serialized(method, seed, arrays)
        tokens = data["test"]["tokens"]
        with torch.no_grad():
            for _ in range(3):
                model(tokens, indices)
            start = time.perf_counter()
            for _ in range(repeats):
                model(tokens, indices)
            elapsed = time.perf_counter() - start
        rows.append({"method": method, "payload_bytes": path.stat().st_size,
                     "token_lookups": len(tokens) * repeats, "wall_seconds": elapsed,
                     "lookups_per_second": len(tokens) * repeats / elapsed})
    return {"seed": result["seed"], "repeats": repeats,
            "runtime": "CPU, one torch thread", "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("outdir")
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.result, args.outdir, args.repeats), indent=2))
