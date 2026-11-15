"""Replay MA-1035 serialized model/index artifacts and exact search metrics."""
import hashlib, json, math, pathlib
import numpy as np
import torch

import run_experiment as ma

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def read_model(raw, model):
    head, blob = raw.split(b"\n", 1)
    manifest = json.loads(head)
    state, off = {}, 0
    for t in manifest["tensor_manifest"]:
        shape = tuple(t["shape"]); size = math.prod(shape) * 4
        state[t["name"]] = torch.from_numpy(np.frombuffer(blob[off:off+size], dtype="<f4").copy().reshape(shape))
        off += size
    assert off == len(blob)
    model.load_state_dict(state, strict=True); model.eval()
    return manifest


def main():
    freeze = json.loads((ROOT / "FREEZE.json").read_text())
    for key, path in [("protocol_sha256", ROOT / "PROTOCOL.json"),
                      ("runner_sha256", ROOT / "source/run_experiment.py"),
                      ("development_result_sha256", RUNS / "development_103500.json")]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == freeze[key], key
    ref = ROOT / "source/qinco_reference.py"
    provenance = json.loads((ROOT / "ARTIFACT_PROVENANCE.json").read_text())
    assert hashlib.sha256(ref.read_bytes()).hexdigest() == provenance["vendored_sha256"]
    assert provenance["upstream_commit"] == freeze["upstream_qinco_commit"]
    report = []
    for seed in freeze["fresh_world_seeds"]:
        world = ma.make_world(seed)
        codebooks = ma.initialize_centroids(world["train"], seed + 1)
        models = ma.build_models(codebooks, seed)
        truth = ma.exact_topk(world["queries"], world["database"], 10)
        run = json.loads((RUNS / f"audit_{seed}.json").read_text())
        assert run["world"] == world["generation"]
        assert run["source_sha256"] == freeze["runner_sha256"]
        assert run["model_config"] == {"d":ma.D,"K":ma.K,"M":ma.M,"L":ma.L,"hidden":ma.H,"updates":ma.UPDATES,"batch":ma.BATCH,"lr":ma.LR}
        for method, model in models.items():
            model_raw = (RUNS / f"model_{seed}_{method}.bin").read_bytes()
            index_raw = (RUNS / f"index_{seed}_{method}.bin").read_bytes()
            assert hashlib.sha256(model_raw).hexdigest() == run["results"][method]["model_payload_sha256"]
            assert hashlib.sha256(index_raw).hexdigest() == run["results"][method]["index_payload_sha256"]
            manifest = read_model(model_raw, model)
            assert manifest["method"] == method and manifest["world_seed"] == seed
            index_head, packed = index_raw.split(b"\n", 1)
            im = json.loads(index_head)
            assert im["rows"] == len(world["database"]) and im["d"] == ma.D and im["M"] == ma.M and im["K"] == ma.K
            codes = ma.unpack_codes(packed, im["rows"])
            assert codes.shape == (len(world["database"]), ma.M)
            with torch.no_grad(): recon = model.decode(torch.from_numpy(codes)).cpu().numpy().astype(np.float32)
            pred = ma.exact_topk(world["queries"], recon, 10)
            expected = run["results"][method]
            nmse = float(np.mean((world["database"]-recon)**2)/(np.var(world["database"])+1e-12))
            r1 = float(np.mean(pred[:,0] == truth[:,0]))
            r10 = float(np.mean([len(set(p)&set(t))/10 for p,t in zip(pred,truth)]))
            assert abs(nmse - expected["normalized_mse"]) < 1e-12, (seed,method,"nmse")
            assert abs(r1 - expected["recall_at_1"]) < 1e-12, (seed,method,"r1")
            assert abs(r10 - expected["recall_at_10"]) < 1e-12, (seed,method,"r10")
            assert len(model_raw) + len(index_raw) == expected["total_serialized_bytes"]
            report.append({"seed":seed,"method":method,"nmse":nmse,"r1":r1,"r10":r10})
    print(json.dumps({"freeze_hashes":True,"generated_world_hashes":True,"model_and_index_hashes":True,
                      "packed_code_roundtrip":True,"normalized_mse_replay":True,"exact_recall_replay":True,
                      "actual_byte_arithmetic":True,"worlds":freeze["fresh_world_seeds"],"rows_replayed":len(report)},sort_keys=True))


if __name__ == "__main__":
    main()
