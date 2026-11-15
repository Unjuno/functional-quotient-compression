"""Verify MA-1033 frozen assets, split isolation, payloads and metric replay."""
from __future__ import annotations
import hashlib, importlib.util, json, math, pathlib
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOP = ROOT.parents[2]
RUNS = ROOT / "runs"
src = ROOT / "source" / "run_experiment.py"
spec = importlib.util.spec_from_file_location("ma1033", src)
ma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ma)


def load_payload(raw, method, init, seed):
    head, blob = raw.split(b"\n", 1)
    meta = json.loads(head)
    assert meta["method"] == method and meta["classes"] == ma.CLASSES and meta["domains"] == ma.DOMAINS
    net = ma.PromptState(init, method, seed)
    state, offset = {}, 0
    for item in meta["tensor_manifest"]:
        shape = tuple(item["shape"])
        size = math.prod(shape) * 4
        state[item["name"]] = torch.from_numpy(np.frombuffer(blob[offset:offset+size], dtype="<f4").copy().reshape(shape))
        offset += size
    assert offset == len(blob)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def main():
    freeze = json.loads((ROOT / "FREEZE.json").read_text())
    for field, path in (("protocol_sha256", ROOT / "PROTOCOL.json"),
                        ("source_sha256", src),
                        ("development_result_sha256", RUNS / "development_103300.json")):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == freeze[field], field
    assert freeze["audit_seeds"] == [103301, 103302, 103303]
    model, tok, proc, files, backbone = ma.assets()
    assert not model.training
    model.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(False)
    assert backbone == 608863580
    model_files = {f["name"]: f for f in files}
    assert model_files["pytorch_model.bin"]["sha256"] == freeze["model_sha256"]
    init = ma.initial_context(model, tok)
    train_ds = ma.load_dataset(ma.DATASET, split="train")
    test_ds = ma.load_dataset(ma.DATASET, split="test")
    ex_train = ma.previous_exclusions("train")
    ex_test = ma.previous_exclusions("test")
    prior_train = {c: set(v) for c, v in ex_train.items()}
    prior_test = {c: set(v) for c, v in ex_test.items()}
    payloads = {m: (RUNS / f"payload_{m}.bin").read_bytes() for m in ["shared_maple", "mirror_product", "additive_code", "independent_cell", "zero_shot"]}
    expected_payload_hashes = {}
    for seed in freeze["audit_seeds"]:
        d = json.loads((RUNS / f"audit_{seed}.json").read_text())
        assert d["fresh_opened_after_freeze"] and d["seed"] == seed
        assert d["code_sha256"] == freeze["source_sha256"]
        assert d["backbone_bytes"] == backbone and d["model_files"] == files
        assert d["train_fingerprint"] == train_ds._fingerprint and d["test_fingerprint"] == test_ds._fingerprint
        for c, ids in d["train_indices"].items():
            assert set(ids).isdisjoint(prior_train[c])
        for c, ids in d["audit_indices"].items():
            assert set(ids).isdisjoint(prior_test[c])
            hashes = [hashlib.sha256(test_ds[int(ix)]["img"].tobytes()).hexdigest() for ix in ids]
            assert hashes == d["audit_image_sha256"][c]
        assert len(ma.records(train_ds, d["train_indices"])) == 240
        assert d["train_meta"]["mirror_product"]["updates"] == 20
        assert d["train_meta"]["mirror_product"]["examples"] == 80
        for method, raw in payloads.items():
            digest = hashlib.sha256(raw).hexdigest()
            assert digest == d["payload_sha256"][method]
            assert len(raw) == d["prompt_payload_bytes"][method]
            assert d["full_bundle_bytes"][method] == backbone + len(raw)
            expected_payload_hashes.setdefault(method, digest)
            assert expected_payload_hashes[method] == digest
        replay = {"zero_shot": ma.evaluate("zero_shot", None, model, tok, proc, test_ds, d["audit_indices"])}
        for method in ["shared_maple", "mirror_product", "additive_code", "independent_cell"]:
            net = load_payload(payloads[method], method, init, 103300)
            replay[method] = ma.evaluate(method, net, model, tok, proc, test_ds, d["audit_indices"])
        for method, rr in replay.items():
            for key in ["observed_pair_accuracy", "heldout_pair_accuracy", "seen_heldout_hmean", "all_accuracy"]:
                assert abs(rr[key] - d["results"][method][key]) <= 1e-12, (seed, method, key, rr[key], d["results"][method][key])
            for domain in ma.DOMAINS:
                assert abs(rr["per_domain_accuracy"][domain] - d["results"][method]["per_domain_accuracy"][domain]) <= 1e-12
    print(json.dumps({"freeze_hashes": True, "asset_hashes": True, "prior_train_test_index_exclusions": True,
                      "audit_image_hashes": True, "payload_hashes_and_bundle_arithmetic": True,
                      "serialized_payload_metric_replay": True, "audit_seeds": freeze["audit_seeds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
