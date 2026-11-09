import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "source"))
import run_experiment


def test_rank4_recovers_aligned_functions():
    bank = run_experiment.make_bank(68901, 0.0)
    basis, codes = run_experiment.fit_codes(bank[:24])
    assert run_experiment.nrmse(codes @ basis, bank[:24]) < 1e-6


def test_serialized_payload_replays_and_native_control_matches(tmp_path):
    run_experiment.run(tmp_path)
    manifest = json.loads((tmp_path / "payload_manifest.json").read_text())
    assert manifest
    for item in manifest:
        p = tmp_path / item["file"]
        assert p.stat().st_size == item["bytes"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item["sha256"]
        assert item["replay_max_abs_error"] < 1e-6

    rows = list(csv.DictReader((tmp_path / "RESULTS_CORE.csv").open()))
    assert rows
    # Native PCA and Mirror are the same fitted linear function and operation count.
    for seed in (68901, 68902):
        for rho in (0.0, 0.05, 0.2):
            for key, mirror_name, native_name in (
                (f"rho{rho:g}_before_transfer", "mirror_code", "native_pca"),
                (f"rho{rho:g}_after_episode_store_deleted", "consolidated_mirror", "consolidated_native_pca"),
            ):
                subset = [r for r in rows if r["world_or_seed"] == str(seed) and r["condition"] == key]
                mirror = next(r for r in subset if r["method"] == mirror_name)
                native = next(r for r in subset if r["method"] == native_name)
                assert abs(float(mirror["primary_value"]) - float(native["primary_value"])) < 1e-8
                assert mirror["active_compute_proxy"] == native["active_compute_proxy"]
                assert mirror["serialized_bytes"] == native["serialized_bytes"]
