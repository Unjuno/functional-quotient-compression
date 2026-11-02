import json
import hashlib
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[5]
RUNS=Path(__file__).resolve().parents[1]/"runs/dev"

def test_serialized_method_bytes_match_dev_metrics():
    for seed in (52701,52702):
        d=RUNS/f"seed_{seed}"
        report=json.loads((d/"metrics.json").read_text())
        for row in report["rows"]:
            payload=d/f"{row['method']}.npz"
            assert payload.stat().st_size==row["payload_bytes"]
            with zipfile.ZipFile(payload) as z:
                assert all(x.compress_type==zipfile.ZIP_STORED for x in z.infolist())
                arrays={n:z.read(n) for n in z.namelist()}
                assert arrays
        assert (d/"explicit_fv.npz").stat().st_size==report["rows"][0]["explicit_payload_bytes"]

def test_dev_freeze_records_no_fresh_metrics_and_identity_exposure():
    freeze=json.loads((Path(__file__).resolve().parents[1]/"DEV_FREEZE.json").read_text())
    assert freeze["decision"]=="FAIL_DEV_NO_FRESH_METRICS"
    assert "task ids 14-15" in freeze["fresh_task_identity_exposure"].lower()
    assert not (Path(__file__).resolve().parents[1]/"runs/fresh").exists()

def test_payload_manifest_hashes_match_files():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/"PAYLOAD_MANIFEST.json").read_text())
    for row in manifest["dev_payload_files"]:
        p=root/"runs/dev"/f"seed_{row['seed']}"/row["path"]
        assert p.stat().st_size==row["bytes"]
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row["sha256"]
