import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from run import METHODS, QRModel, load_serialized


def benchmark(root, result_path, repeats=100):
    root, result = Path(root), json.loads(Path(result_path).read_text())
    tokens = torch.arange(144).repeat(64)
    output = []
    for row in result["summaries"]:
        path = root / "payloads" / f"development_{result['seed']}_{row['method']}.npz"
        import zipfile
        with zipfile.ZipFile(path) as archive:
            arrays = {name[:-4]: np.load(archive.open(name), allow_pickle=False) for name in archive.namelist()}
        model = load_serialized(row["method"], result["seed"], arrays)
        with torch.no_grad():
            model(tokens)
            start = time.perf_counter()
            for _ in range(repeats):
                model(tokens)
            elapsed = time.perf_counter() - start
        count = repeats * len(tokens)
        output.append({"method": row["method"], "repeats": repeats, "examples": count,
                       "wall_seconds": elapsed, "examples_per_second": count / elapsed})
    return output


if __name__ == "__main__":
    print(json.dumps(benchmark(sys.argv[1], sys.argv[2]), indent=2))
